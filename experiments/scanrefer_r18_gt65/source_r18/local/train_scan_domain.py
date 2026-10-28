"""Matched 3D/RGB heads; historical validation is dev, new rooms stay sealed."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from category_common_v82 import file_hash, run_lock
from category_v10.run import read_manifest, check_sources
from rgb_classifier import fuse
from run_rgb_experiment import check_frozen, freeze
from run_storage import write_json
import shape_scale

shape_scale.normalized_category = lru_cache(None)(shape_scale.normalized_category)
SEEDS = (42, 43, 44)
WEIGHTS = [i/10 for i in range(11)]
METHODS = ('S0', 'S1', 'S2', 'S3', 'S4')


def make_head(text):
    head = nn.Linear(512, len(text))
    with torch.no_grad():
        head.weight.copy_(10 * text)
        head.bias.zero_()
    return head


def dev_measure(logits, labels):
    known = labels >= 0
    correct = int(((logits.argmax(-1) == labels) & known).sum())
    return {'correct': correct, 'accuracy': correct/len(labels),
            'known_CE': float(F.cross_entropy(logits[known], labels[known]))}


def checkpoint_key(row):
    return (-row['dev']['correct'], row['dev']['known_CE'], row['epoch'])


def train_one(output, modality, seed, x, y, dev, dev_y, text, config_hash):
    directory = output / f'models/seed{seed}/{modality}'
    marker = directory / 'COMPLETE.json'
    if marker.exists():
        value = json.loads(marker.read_text())
        if file_hash(directory / 'head.pt') != value['head_sha256'] or file_hash(directory / 'history.json') != value['history_sha256']:
            raise ValueError('Completed head changed')
        return
    if directory.exists():
        raise ValueError('Incomplete training preserved: choose a fresh model output')
    directory.mkdir(parents=True)
    torch.manual_seed(seed)
    head = make_head(text).to(x.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.003, weight_decay=.01)
    generator = torch.Generator(device=x.device).manual_seed(seed)
    history, best = [], None
    for epoch in range(1, 41):
        head.train()
        total = 0.
        for indices in torch.randperm(len(y), device=x.device, generator=generator).split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(x[indices]), y[indices])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        head.eval()
        with torch.inference_mode():
            measure = dev_measure(head(dev), dev_y)
        row = {'epoch': epoch, 'train_CE': total/len(y), 'dev': measure}
        history.append(row)
        write_json(directory / 'history.json', history)
        if best is None or checkpoint_key(row) < checkpoint_key(best):
            best = row
            torch.save({'state_dict': {k: v.detach().cpu() for k, v in head.state_dict().items()},
                        'epoch': epoch, 'seed': seed, 'modality': modality, 'config_sha256': config_hash}, directory / 'head.tmp')
            (directory / 'head.tmp').replace(directory / 'head.pt')
        print(json.dumps({'stage': 'training', 'modality': modality, 'seed': seed, **row}), flush=True)
    write_json(marker, {'head_sha256': file_hash(directory / 'head.pt'),
                       'history_sha256': file_hash(directory / 'history.json'), 'epochs': 40})


def predict(output, modality, seed, features, text, config_hash):
    path = output / f'models/seed{seed}/{modality}/head.pt'
    record = torch.load(path, map_location='cpu', weights_only=True)
    history = json.loads(path.with_name('history.json').read_text())
    if (len(history) != 40 or record['epoch'] != min(history, key=checkpoint_key)['epoch']
            or record['config_sha256'] != config_hash or record['modality'] != modality or record['seed'] != seed):
        raise ValueError('Checkpoint selection/provenance differs')
    head = make_head(text).to(features.device)
    head.load_state_dict(record['state_dict'], strict=True)
    head.eval()
    with torch.inference_mode():
        return head(features).cpu().numpy()


def hit_count(logits, labels):
    return int(((logits.argmax(-1) == labels) & (labels >= 0)).sum())


def choose(parts, labels):
    grid, selected = {}, {}
    for method, source in (('S3', [parts['base']] * 3), ('S4', parts['shape'])):
        rows = []
        for w in WEIGHTS:
            counts = [hit_count(fuse(s, r, parts['valid'], w), labels) for s, r in zip(source, parts['rgb'])]
            rows.append({'weight': w, 'correct_by_seed': counts, 'total_correct': sum(counts),
                         'mean_accuracy': sum(counts)/(3*len(labels))})
        grid[method] = rows
        selected[method] = min(rows, key=lambda r: (-r['total_correct'], r['weight']))
    candidates = [{'method': 'S0', 'total_correct': 3*hit_count(parts['base'], labels)}]
    for method, scores in (('S1', parts['shape']), ('S2', parts['rgb'])):
        candidates.append({'method': method, 'total_correct': sum(hit_count(s, labels) for s in scores)})
    candidates += [{'method': m, 'total_correct': selected[m]['total_correct']} for m in ('S3', 'S4')]
    primary = min(candidates, key=lambda r: (-r['total_correct'], METHODS.index(r['method'])))['method']
    return {'fusion': selected, 'primary_method': primary, 'method_candidates': candidates}, grid


def all_scores(parts, selection):
    result = {'S0': parts['base']}
    for index, seed in enumerate(SEEDS):
        result[f'S1_seed{seed}'] = parts['shape'][index]
        result[f'S2_seed{seed}'] = parts['rgb'][index]
        for method, base in (('S3', parts['base']), ('S4', parts['shape'][index])):
            result[f'{method}_seed{seed}'] = fuse(base, parts['rgb'][index], parts['valid'], selection['fusion'][method]['weight'])
    return result


def main(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output, feature_root = args.output, args.output / 'features'
    check_frozen(output, 'PROTOCOL_FROZEN.json')
    check_frozen(output, 'FEATURES_FROZEN.json')
    manifests = {s: read_manifest(output / f'manifests/{s}.json') for s in ('train', 'dev')}
    check_sources(manifests['train']['sources'], manifests['dev']['sources'])
    source_names = ('train_scan_domain.py', 'cache_scan_domain.py', 'scan_domain_protocol.py',
                    'rgb_classifier.py', 'shape_scale.py', 'category_common_v82.py', 'run_storage.py',
                    'run_rgb_experiment.py', 'category_v10/run.py', 'category_v10/data.py',
                    'pointcloud_classifier_v82.py')
    cfg = {'version': 'matched-scan-domain-heads-r1', 'seeds': SEEDS, 'epochs': 40, 'batch_size': 256,
           'lr': .003, 'weight_decay': .01, 'classes': 197, 'dimension': 512,
           'initialization': '10 * normalized shared text weight, zero bias',
           'checkpoint': 'max research-dev True, min known CE, earliest epoch',
           'head_parameters': 512*197+197, 'dev_role': manifests['dev']['evaluation_status'],
           'design_sha256': file_hash(args.design), 'features_seal_sha256': file_hash(output / 'FEATURES_FROZEN.json'),
           'protocol_seal_sha256': file_hash(output / 'PROTOCOL_FROZEN.json'),
           'code_sha256': {name: file_hash(Path(__file__).parent / name) for name in source_names},
           'fusion_weights': WEIGHTS, 'primary_tie_order': METHODS, 'torch': str(torch.__version__)}
    cfg = json.loads(json.dumps(cfg))
    with run_lock(output / 'training_lock'):
        config_path = output / 'training_config.json'
        if config_path.exists() and json.loads(config_path.read_text()) != cfg:
            raise ValueError('Training inputs/code/design changed')
        if not config_path.exists():
            write_json(config_path, cfg)
            snapshot = output / 'source_snapshot'
            snapshot.mkdir()
            for name in source_names:
                (snapshot / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(Path(__file__).parent / name, snapshot / name)
            shutil.copyfile(args.design, output / 'design.md')
        config_hash = file_hash(config_path)
        text = torch.from_numpy(np.load(feature_root / 'text.npy')).float()
        split = 'dev' if args.stage == 'train-select' else 'eval'
        if split == 'eval':
            check_frozen(output, 'MODEL_SELECTION_FROZEN.json')
            check_frozen(output, 'EVAL_FEATURES_FROZEN.json')
            manifests[split] = read_manifest(output / 'manifests/eval.json')
            for old in ('train', 'dev'):
                check_sources(manifests[old]['sources'], manifests[split]['sources'])
        with np.load(feature_root / f'shape_{split}.npz') as a:
            shape = torch.from_numpy(a['features']).float().to(args.device)
        rgb_path = feature_root / ('rgb_dev.npz' if split == 'dev' else 'rgb_eval/features.npz')
        with np.load(rgb_path) as a:
            rgb = torch.from_numpy(a['features']).float().to(args.device)
            valid = a['valid']
        if len(shape) != len(manifests[split]['samples']) or len(rgb) != len(shape) or valid.shape != (len(shape),):
            raise ValueError('Feature/manifest length mismatch')
        labels = np.array([r['label'] for r in manifests[split]['samples']], dtype=np.int64)
        if split == 'dev':
            with np.load(feature_root / 'shape_train.npz') as a:
                train_shape = torch.from_numpy(a['features']).float().to(args.device)
            with np.load(feature_root / 'rgb_train/features.npz') as a:
                train_rgb = torch.from_numpy(a['features']).float().to(args.device)
                if not a['valid'].all() or not valid.all():
                    raise ValueError('Matched training requires full RGB coverage')
            y = torch.tensor([r['label'] for r in manifests['train']['samples']], device=args.device)
            if len(y) != len(train_shape) or len(y) != len(train_rgb) or (y < 0).any():
                raise ValueError('Matched training membership mismatch')
            for modality, values, development in (('shape', train_shape, shape), ('rgb', train_rgb, rgb)):
                for seed in SEEDS:
                    train_one(output, modality, seed, values, y, development,
                              torch.from_numpy(labels).to(args.device), text, config_hash)
        base = (10 * shape @ text.to(args.device).T).cpu().numpy()
        parts = {'base': base, 'valid': valid,
                 'shape': np.stack([predict(output, 'shape', seed, shape, text, config_hash) for seed in SEEDS]),
                 'rgb': np.stack([predict(output, 'rgb', seed, rgb, text, config_hash) for seed in SEEDS])}
        parts['rgb'][:, ~valid] = base[~valid]
        destination = output / split
        destination.mkdir(exist_ok=True)
        np.savez_compressed(destination / 'components.npz', **parts)
        if split == 'dev':
            selection, grid = choose(parts, labels)
            write_json(output / 'selection.json', selection)
            write_json(output / 'dev_grid.json', grid)
        else:
            selection = json.loads((output / 'selection.json').read_text())
        records = []
        for name, logits in all_scores(parts, selection).items():
            np.save(destination / f'{name}.npy', logits)
            hit, _ = shape_scale.correctness(logits, manifests[split])
            base_hit, _ = shape_scale.correctness(base, manifests[split])
            records.append({'method': name, 'summary': shape_scale.score(logits, manifests[split]),
                            'corrected': int((hit & ~base_hit).sum()), 'broken': int((~hit & base_hit).sum())})
        write_json(destination / 'results.json', {'primary_method_locked': selection['primary_method'], 'records': records,
            'valid_RGB': int(valid.sum()), 'count': len(valid), 'split_role': manifests[split]['evaluation_status']})
        if split == 'dev':
            names = ['training_config.json', 'design.md', 'selection.json', 'dev_grid.json',
                     'dev/components.npz', 'dev/results.json', 'FEATURES_FROZEN.json', 'PROTOCOL_FROZEN.json']
            names += [f'models/seed{s}/{m}/{f}' for s in SEEDS for m in ('shape', 'rgb') for f in ('head.pt', 'history.json', 'COMPLETE.json')]
            freeze(output, 'MODEL_SELECTION_FROZEN.json', names)
        else:
            freeze(output, 'RESULTS_FROZEN.json', ['eval/components.npz', 'eval/results.json'] +
                   [f'eval/{name}.npy' for name in all_scores(parts, selection)])
        print(json.dumps({'stage': args.stage + '_complete', 'selection': selection,
                          'results': [{'method': r['method'], 'correct': r['summary']['correct']} for r in records]}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('train-select', 'evaluate'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--design', required=True, type=Path)
    parser.add_argument('--device', default='cuda')
    main(parser.parse_args())
