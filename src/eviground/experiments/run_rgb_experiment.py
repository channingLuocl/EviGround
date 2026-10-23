"""Train-only rendering QA, dev-only fusion selection, then fixed evaluation."""
from eviground.paths import source_file
import argparse
from functools import lru_cache
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from eviground.categories.category_common_v82 import file_hash, instance_seed, load_cloud, run_lock
from eviground.classifiers.category_v10.run import check_sources, read_manifest
import eviground.geometry.shape_scale as shape_scale
from eviground.geometry.shape_scale import load_split, load_global, frozen_logits, load_scale, head_scores
from eviground.classifiers.rgb_classifier import RENDER, load_clip, verify_text, render_views, extract, fuse
from eviground.evaluation.run_storage import write_json

shape_scale.normalized_category = lru_cache(None)(shape_scale.normalized_category)
WEIGHTS = [i / 10 for i in range(11)]
SIZE_WEIGHTS = [0., .25, .5, 1., 2.]
SEEDS = [42, 43, 44]


def config(args):
    manifests = {s: read_manifest(args.feature_root / f'manifests/{s}.json') for s in ('train', 'dev', 'eval')}
    for first, second in (('train', 'dev'), ('train', 'eval'), ('dev', 'eval')):
        check_sources(manifests[first]['sources'], manifests[second]['sources'])
    code = {name: file_hash(source_file(name)) for name in
            ('rgb_classifier.py', 'run_rgb_experiment.py', 'shape_scale.py', 'category_common_v82.py',
             'category_v10/run.py', 'category_v10/data.py', 'run_storage.py')}
    return {'version': 'fixed-rgb-fusion-r1', 'renderer': RENDER, 'weights': WEIGHTS, 'size_weights': SIZE_WEIGHTS,
            'fusion': '(1-w)*z_class(3D)+w*z_class(image)+c*size_residual; RGB failure falls back to z_class(3D)',
            'selection': 'max dev True; tie smaller size coefficient then smaller image weight',
            'image_head_training': False, 'image_feature': 'normalize(mean(four normalized image embeddings))',
            'clip_sha256': file_hash(args.clip), 'clip': str(args.clip), 'code_sha256': code,
            'prior_config_sha256': file_hash(args.prior / 'config.json'), 'torch': str(torch.__version__),
            'manifests': {s: file_hash(args.feature_root / f'manifests/{s}.json') for s in manifests},
            'scale_heads': {str(s): file_hash(args.scale_root / f'models/seed{s}/size/head.pt') for s in SEEDS},
            'evaluation_status': 'previously inspected validation, not untouched test'}, manifests


def freeze(output, marker, names):
    write_json(output / marker, {'files': {name: file_hash(output / name) for name in names}})


def check_frozen(output, marker):
    for name, digest in json.loads((output / marker).read_text())['files'].items():
        if file_hash(output / name) != digest:
            raise ValueError(f'Frozen artifact changed: {name}')


def qa(args, manifests, model):
    manifest = manifests['train']
    out = args.output / 'qa'
    out.mkdir()
    records = []
    for number, index in enumerate([0, len(manifest['samples']) // 2, len(manifest['samples']) - 1]):
        row = manifest['samples'][index]
        expected = next(s['cloud_sha256'] for s in manifest['sources'] if s['scene'] == row['scene'])
        if file_hash(row['cloud']) != expected:
            raise ValueError('QA cloud hash mismatch')
        xyz, rgb, ids = load_cloud(row['cloud'])
        selected = ids == row['bbox_id']
        images, coverage = render_views(xyz[selected], rgb[selected] if rgb is not None else None,
            instance_seed(RENDER['seed'], row['scene'], row['bbox_id']))
        if images is None:
            raise ValueError('Fixed QA object cannot be rendered')
        Image.fromarray(np.concatenate(list(images), axis=1)).save(out / f'object{number}.png')
        records.append({'manifest_index': index, 'scene': row['scene'], 'bbox_id': row['bbox_id'],
                        'coverage': coverage, 'points': int(selected.sum())})
    write_json(out / 'records.json', records)


def size_scores(args, data):
    result = []
    for seed in SEEDS:
        scale_args = SimpleNamespace(output=args.scale_root, device='cpu')
        head = load_scale(scale_args, seed, 'size')
        result.append(head_scores(head, data['size'], 'cpu'))
    return np.stack(result)


def select(base, image, valid, sizes, manifest):
    names = np.array([shape_scale.normalized_category(c) for c in manifest['vocabulary']])
    targets = np.array([shape_scale.normalized_category(r['category']) for r in manifest['samples']])
    grids = {}
    for method in ('V2', 'V3'):
        candidates = []
        for weight in WEIGHTS:
            for coefficient in ([0.] if method == 'V2' else SIZE_WEIGHTS):
                accuracies = [float((names[fuse(base, image, valid, weight, s, coefficient).argmax(-1)] == targets).mean())
                              for s in (sizes[:1] if method == 'V2' else sizes)]
                candidates.append({'weights': [weight, coefficient], 'accuracies': accuracies,
                                   'mean_accuracy': float(np.mean(accuracies))})
        grids[method] = candidates
    selection = {m: min(rows, key=lambda r: (-r['mean_accuracy'], r['weights'][1], r['weights'][0]))
                 for m, rows in grids.items()}
    return selection, grids


def main(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with run_lock(args.output):
        cfg, manifests = config(args)
        config_path = args.output / 'config.json'
        if config_path.exists() and json.loads(config_path.read_text()) != cfg:
            raise ValueError('Inputs/code changed: use a fresh experiment output')
        if not config_path.exists():
            write_json(config_path, cfg)
            source = args.output / 'source_snapshot'
            for name in cfg['code_sha256']:
                target = source / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_file(name), target)
            target = args.output / 'manifests'
            target.mkdir()
            for split in manifests:
                shutil.copyfile(args.feature_root / f'manifests/{split}.json', target / f'{split}.json')
        if args.stage != 'qa':
            check_frozen(args.output, 'QA_FROZEN.json')
        if args.stage == 'eval':
            check_frozen(args.output, 'SELECTION_FROZEN.json')
        model, tokenizer = load_clip(args.clip, args.device)
        prior = json.loads((args.prior / 'config.json').read_text())
        text, global_weight = load_global(args, 42, prior)
        alignment = verify_text(model, tokenizer, manifests['train']['vocabulary'], text, args.device)
        alignment['checkpoint_sha256'] = cfg['clip_sha256']
        if args.stage == 'qa':
            write_json(args.output / 'text_alignment.json', alignment)
            qa(args, manifests, model)
            return
        split = 'dev' if args.stage == 'dev' else 'eval'
        data = load_split(args, split, prior)
        base, _ = frozen_logits(data, text, global_weight)
        features, valid = extract(model, manifests[split], args.output / split, args.device)
        image = features @ text.numpy().T
        sizes = size_scores(args, data)
        np.savez_compressed(args.output / split / 'components.npz', base=base, image=image, valid=valid, size=sizes)
        if args.stage == 'dev':
            selection, grids = select(base, image, valid, sizes, manifests[split])
            write_json(args.output / 'dev_grid.json', grids)
            write_json(args.output / 'selection.json', selection)
            freeze(args.output, 'SELECTION_FROZEN.json', ['config.json', 'selection.json', 'dev_grid.json',
                    'QA_FROZEN.json', 'dev/components.npz', 'dev/features.npz'])
            print(json.dumps({'stage': 'selection_frozen', 'selection': selection}), flush=True)
            return
        selection = json.loads((args.output / 'selection.json').read_text())
        predictions = {'V0': base, 'V1': fuse(base, image, valid, 1.)}
        for method in ('V2', 'V3'):
            w, c = selection[method]['weights']
            for index in (range(3) if method == 'V3' else range(1)):
                key = f'V3_seed{SEEDS[index]}' if method == 'V3' else method
                predictions[key] = fuse(base, image, valid, w, sizes[index], c)
        baseline_hit, _ = shape_scale.correctness(base, manifests[split])
        records = []
        for method, logits in predictions.items():
            np.save(args.output / split / f'{method}.npy', logits)
            hit, _ = shape_scale.correctness(logits, manifests[split])
            records.append({'method': method, 'summary': shape_scale.score(logits, manifests[split]),
                            'corrected': int((hit & ~baseline_hit).sum()), 'broken': int((~hit & baseline_hit).sum())})
        image_hit, _ = shape_scale.correctness(predictions['V1'], manifests[split])
        report = {'records': records, 'valid_RGB': int(valid.sum()), 'total': len(valid), 'selection': selection,
                  'diagnostic_oracle_accuracy': float((baseline_hit | image_hit).mean()),
                  'oracle_is_not_a_deployable_result': True}
        write_json(args.output / 'results.json', report)
        names = ['results.json', 'eval/components.npz', 'eval/features.npz'] + [f'eval/{m}.npy' for m in predictions]
        freeze(args.output, 'RESULTS_FROZEN.json', names)
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('feature-root', 'prior', 'scale-root', 'clip', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--stage', choices=('qa', 'dev', 'eval'), required=True)
    main(parser.parse_args())
