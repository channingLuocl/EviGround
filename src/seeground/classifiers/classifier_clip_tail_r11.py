"""Cache frozen CLIP prefix tokens, then adapt its actual final image block."""
from seeground.paths import source_file
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from seeground.experiments.audit_pair_train_test import recount
from seeground.experiments.audit_pair_views import z
from seeground.categories.category_common_v82 import file_hash, instance_seed, load_cloud, run_lock
from seeground.classifiers.classifier_residual_r9 import bases, choose
from seeground.classifiers.clip_tail_r11 import ClipTail, SPECS, objective, prefix_tokens
from seeground.classifiers.rgb_classifier import RENDER, load_clip, render_views, verify_text
from seeground.experiments.run_rgb_experiment import check_frozen, freeze
from seeground.evaluation.run_storage import write_json

SEEDS = (42, 43, 44)
EPOCHS = (3, 6)
WEIGHTS = (0., .2, .4, .6, .8, 1.)
REFERENCES = ('S0', 'r4_M', 'r8_M')


def read(path):
    return json.loads(Path(path).read_text())


def setup(args):
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    manifests = {s: read(args.reference / f'manifests/{s}.json') for s in ('train', 'eval', 'dev')}
    assert [len(m['samples']) for m in manifests.values()] == [28730, 404, 3413]
    rooms = {s: {r['scene'].split('_')[0] for r in m['samples']} for s, m in manifests.items()}
    assert not rooms['train'] & rooms['eval'] and not rooms['train'] & rooms['dev'] and not rooms['eval'] & rooms['dev']
    assert all(m['vocabulary'] == manifests['train']['vocabulary'] for m in manifests.values())
    assert all(r['label'] >= 0 for r in manifests['train']['samples'])
    for marker in ('FEATURES_FROZEN.json', 'EVAL_FEATURES_FROZEN.json'):
        check_frozen(args.reference, marker)
    for ref in (args.r4, args.r8):
        check_frozen(ref, 'TEST_FROZEN.json')
    old = read(args.reference / 'features/config.json')
    assert file_hash(args.clip) == old['clip_sha256'] and RENDER == old['renderer']
    sources = ('classifier_clip_tail_r11.py', 'clip_tail_r11.py', 'rgb_classifier.py', 'category_common_v82.py',
               'classifier_residual_r9.py', 'classifier_backbone_r5.py', 'pointnext_residual_r9.py',
               'audit_pair_train_test.py', 'audit_pair_views.py', 'run_rgb_experiment.py', 'run_storage.py')
    inputs = [args.design, args.clip, args.reference / 'features/text.npy',
              args.reference / 'features/rgb_train/features.npz', args.reference / 'features/rgb_eval/features.npz',
              args.r4 / 'validation/locked_M_v0.npy', args.r8 / 'validation/locked_M.npy']
    inputs += [args.reference / f'manifests/{s}.json' for s in manifests]
    storage_path = root / 'RAM_STORAGE.json'
    if storage_path.exists():
        inputs.append(storage_path)
    config = {'version': 'clip-image-tail-r11', 'seeds': SEEDS, 'specs': SPECS,
              'epochs': 6, 'checkpoints': EPOCHS, 'batch_objects': 32, 'views': 4,
              'warmup': '40 epochs train-only linear head, same paired seed initialization for every arm',
              'head_lr': .001, 'tail_lr': 1e-5, 'weight_decay': .01, 'scheduler': '6 epoch cosine after warmup',
              'loss': 'inverse-sqrt CE; tail_anchor adds squared unit feature distance, coefficient 1',
              'encoder': 'original CLIP ViT-B/16 first11 blocks frozen, final block/norm/projection train only in tail arms',
              'precision': 'float32 prefix storage and training, TF32 disabled; no AMP or quantization',
              'storage': read(storage_path) if storage_path.exists() else {'kind': 'shared_disk'},
              'renderer': RENDER, 'augmentation': 'none; original four deterministic views',
              'reference_scale': 'original stored S0/r4/r8; only new scores get per-object z-score',
              'source': {n: file_hash(source_file(n)) for n in sources},
              'inputs': {str(p): file_hash(p) for p in inputs}, 'torch': str(torch.__version__),
              'limitations': 'revealed validation/historical evaluation, original pretraining exposure unknown; r8 rotation precision limit'}
    config = json.loads(json.dumps(config))
    if (root / 'config.json').exists():
        assert read(root / 'config.json') == config
    else:
        write_json(root / 'config.json', config)
        for folder in ('source_snapshot', 'manifests'):
            (root / folder).mkdir()
        for name in sources:
            shutil.copyfile(source_file(name), root / 'source_snapshot' / name)
        shutil.copyfile(args.design, root / 'source_snapshot/design.md')
        for split in manifests:
            shutil.copyfile(args.reference / f'manifests/{split}.json', root / f'manifests/{split}.json')
        freeze(root, 'PROTOCOL_FROZEN.json', ['config.json'] + [str(p.relative_to(root))
               for folder in ('source_snapshot', 'manifests') for p in (root / folder).iterdir()])
    check_frozen(root, 'PROTOCOL_FROZEN.json')
    return manifests


def load_model(args, vocabulary):
    # Keep the active Torch installation; borrow only missing Python dependencies.
    sys.path.append('/root/paper/reproductions/SeeGround/.venv-v82/lib/python3.12/site-packages')
    model, tokenizer = load_clip(args.clip, 'cuda')
    text = torch.from_numpy(np.load(args.reference / 'features/text.npy')).cuda()
    alignment = verify_text(model, tokenizer, vocabulary, text, 'cuda')
    return model, text, alignment


def rows_by_scene(manifest):
    groups = {}
    for i, row in enumerate(manifest['samples']):
        groups.setdefault(row['scene'], []).append(i)
    return groups


def render_scene_rows(manifest, indices):
    row = manifest['samples'][indices[0]]
    sources = {s['scene']: s['cloud_sha256'] for s in manifest['sources']}
    assert file_hash(row['cloud']) == sources[row['scene']]
    xyz, rgb, ids = load_cloud(row['cloud'])
    assert rgb is not None
    order = np.argsort(ids, kind='stable')
    unique, start, count = np.unique(ids[order], return_index=True, return_counts=True)
    lookup = {int(oid): order[s:s+n] for oid, s, n in zip(unique, start, count)}
    for start in range(0, len(indices), 8):
        chunk = indices[start:start+8]
        images = []
        for i in chunk:
            row = manifest['samples'][i]
            selected = lookup[row['bbox_id']]
            image, coverage = render_views(xyz[selected], rgb[selected],
                                           instance_seed(RENDER['seed'], row['scene'], row['bbox_id']))
            assert image is not None and min(coverage) > 0
            images.append(image)
        yield chunk, np.concatenate(images)


def image_tensor(images):
    x = torch.from_numpy(images).permute(0, 3, 1, 2).cuda().float() / 255
    mean = x.new_tensor([.48145466, .4578275, .40821073])[None, :, None, None]
    std = x.new_tensor([.26862954, .26130258, .27577711])[None, :, None, None]
    return (x - mean) / std


def preflight(args):
    manifests = setup(args)
    assert not (args.output / 'PREFLIGHT.json').exists()
    model, text, alignment = load_model(args, manifests['train']['vocabulary'])
    chunks, original = [], []
    for indices in rows_by_scene(manifests['train']).values():
        for chunk, images in render_scene_rows(manifests['train'], indices):
            x = image_tensor(images)
            with torch.no_grad():
                tokens = prefix_tokens(model.visual, x).reshape(len(chunk), 4, 197, 768)
                expected = F.normalize(model.encode_image(x, normalize=True).reshape(len(chunk), 4, 512).mean(1), dim=-1)
            chunks.append((chunk, tokens)); original.append(expected)
            if sum(len(c) for c, _ in chunks) >= 32:
                break
        if sum(len(c) for c, _ in chunks) >= 32:
            break
    indices = sum([c for c, _ in chunks], [])[:32]
    tokens = torch.cat([t for _, t in chunks])[:32]
    expected = torch.cat(original)[:32]
    tail = ClipTail(model.visual, text)
    with torch.no_grad():
        first = tail.features(tokens)
    torch.testing.assert_close(first, expected, atol=2e-5, rtol=2e-5)
    visual_before = {k: v.detach().clone() for k, v in model.visual.state_dict().items()}
    tail.configure('tail_anchor')
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in tail.named_parameters() if not n.startswith('head.')], 'lr': 1e-5},
        {'params': tail.head.parameters(), 'lr': .001}], weight_decay=.01)
    labels = torch.tensor([manifests['train']['samples'][i]['label'] for i in indices], device='cuda')
    history, gradients = [], {}
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started = time.time()
    for step in range(20):
        optimizer.zero_grad(set_to_none=True)
        logits, features = tail(tokens)
        total, ce, anchor = objective(logits, features, expected, labels, torch.ones(len(text), device='cuda'), 'tail_anchor')
        assert torch.isfinite(total)
        total.backward()
        if step == 0:
            gradients = {n: float(p.grad.norm()) for n, p in tail.named_parameters()}
            assert all(np.isfinite(list(gradients.values())))
            assert sum(v for k, v in gradients.items() if k.startswith('block.')) > 0
        optimizer.step(); history.append(float(total.detach()))
    torch.cuda.synchronize()
    assert history[-1] < history[0]
    assert all(torch.equal(visual_before[k], v) for k, v in model.visual.state_dict().items())
    assert not torch.equal(tail.block.attn.in_proj_weight, model.visual.transformer.resblocks[-1].attn.in_proj_weight)
    write_json(args.output / 'PREFLIGHT.json', {'status': 'passed', 'objects': len(indices), 'steps': 20,
        'original_bridge_max_abs': float((first - expected).abs().max()), 'losses': history, 'gradients': gradients,
        'seconds': time.time() - started, 'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
        'original_visual_state': 'exactly_unchanged', 'tail_parameters': sum(p.numel() for p in tail.parameters()),
        'text_alignment': alignment, 'train_object_indices': indices})
    print(json.dumps(read(args.output / 'PREFLIGHT.json')), flush=True)


@torch.inference_mode()
def cache(args, split):
    manifests = setup(args)
    if split == 'dev':
        check_frozen(args.output, 'SELECTION_FROZEN.json')
    folder = args.output / 'tokens'; folder.mkdir(exist_ok=True)
    assert not (folder / f'{split}.json').exists()
    path = folder / f'{split}.npy'
    if (args.output / 'RAM_STORAGE.json').exists():
        target = read(args.output / 'RAM_STORAGE.json')['targets'][split]
        assert path.is_symlink() and str(path.readlink()) == target and path.stat().st_size == 0
    else:
        assert not path.exists()
    manifest = manifests[split]
    model, text, alignment = load_model(args, manifest['vocabulary'])
    tail = ClipTail(model.visual, text)
    old_path = args.reference / ('features/rgb_dev.npz' if split == 'dev' else f'features/rgb_{split}/features.npz')
    old = np.load(old_path); assert old['valid'].all()
    values = np.lib.format.open_memmap(folder / f'{split}.npy', mode='w+', dtype=np.float32,
                                      shape=(len(manifest['samples']), 4, 197, 768))
    reference = np.zeros((len(values), 512), dtype=np.float32)
    maximum, completed = 0., 0
    for number, indices in enumerate(rows_by_scene(manifest).values(), 1):
        for chunk, images in render_scene_rows(manifest, indices):
            tokens = prefix_tokens(model.visual, image_tensor(images)).reshape(len(chunk), 4, 197, 768)
            features = tail.features(tokens).cpu().numpy()
            difference = float(np.max(np.abs(features - old['features'][chunk])))
            assert difference <= 1e-5, (split, chunk, difference)
            values[chunk] = tokens.cpu().numpy()
            reference[chunk] = features; maximum = max(maximum, difference); completed += len(chunk)
        if number % 10 == 0 or number == 1:
            values.flush()
            print(json.dumps({'stage': 'prefix_cache', 'split': split, 'scenes': number,
                              'objects': completed, 'total': len(values), 'rgb_bridge_max_abs': maximum}), flush=True)
    values.flush(); del values
    np.save(folder / f'{split}_reference.npy', reference)
    write_json(folder / f'{split}.json', {'objects': len(reference), 'shape': [len(reference), 4, 197, 768],
        'dtype': 'float32', 'sha256': file_hash(folder / f'{split}.npy'),
        'reference_sha256': file_hash(folder / f'{split}_reference.npy'), 'old_rgb_sha256': file_hash(old_path),
        'manifest_sha256': file_hash(args.output / f'manifests/{split}.json'),
        'rgb_bridge_max_abs': maximum, 'text_alignment': alignment})
    print(json.dumps({'stage': 'cache_complete', 'split': split}), flush=True)


def token_bank(args, split):
    folder = args.output / 'tokens'; metadata = read(folder / f'{split}.json')
    assert file_hash(folder / f'{split}.npy') == metadata['sha256']
    assert file_hash(folder / f'{split}_reference.npy') == metadata['reference_sha256']
    return np.load(folder / f'{split}.npy', mmap_mode='r'), torch.from_numpy(np.load(folder / f'{split}_reference.npy')).cuda()


def batch_tokens(bank, indices):
    # A writable minibatch avoids mutating the immutable memory-mapped cache.
    return torch.from_numpy(np.array(bank[indices], copy=True)).cuda()


@torch.inference_mode()
def predict(model, bank, reference):
    if model.spec == 'frozen_head':
        return model.head(reference).cpu().numpy()
    return np.concatenate([model(batch_tokens(bank, slice(start, start+32)))[0].cpu().numpy()
                           for start in range(0, len(bank), 32)])


def warm_head(text, teacher, labels, weights, seed):
    torch.manual_seed(seed)
    head = nn.Linear(text.shape[1], len(text), device='cuda')
    with torch.no_grad():
        head.weight.copy_(10 * text); head.bias.zero_()
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.01)
    generator = torch.Generator(device='cuda').manual_seed(seed)
    history = []
    for _ in range(40):
        total = 0.
        for indices in torch.randperm(len(labels), device='cuda', generator=generator).split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(teacher[indices]), labels[indices], weight=weights)
            loss.backward(); optimizer.step(); total += float(loss.detach()) * len(indices)
        history.append(total / len(labels))
    return copy.deepcopy(head.state_dict()), history


def make_grid(raw, base, manifest):
    rows = []
    for (spec, epoch), (arrays, parameters) in raw.items():
        for name, reference in base.items():
            for weight in WEIGHTS:
                metrics = [recount((1-weight)*reference + weight*z(v.astype(float)), manifest)[0] for v in arrays]
                rows.append({'name': spec, 'epoch': epoch, 'base': name, 'weight': weight, 'parameters': parameters,
                             'correct_by_seed': [m['correct'] for m in metrics],
                             'mean_correct': float(np.mean([m['correct'] for m in metrics])),
                             'mean_macro': float(np.mean([m['class_mean_accuracy'] for m in metrics]))})
    for name, reference in base.items():
        metric = recount(reference, manifest)[0]
        rows.append({'name': name, 'epoch': 0, 'base': name, 'weight': 0., 'parameters': 0,
                     'correct_by_seed': [metric['correct']]*3, 'mean_correct': float(metric['correct']),
                     'mean_macro': metric['class_mean_accuracy']})
    ref = next(r for r in rows if r['name'] == 'r8_M')
    return rows, {'locked': {m: choose(rows, ref, m) for m in ('P', 'B', 'M')},
                  'per_candidate': {s: choose([r for r in rows if r['name'] == s], ref, 'P') for s in SPECS}}


def train_select(args):
    manifests = setup(args)
    assert read(args.output / 'PREFLIGHT.json')['status'] == 'passed'
    assert not (args.output / 'SELECTION_FROZEN.json').exists()
    train, teacher = token_bank(args, 'train'); validation, reference = token_bank(args, 'eval')
    original, text, _ = load_model(args, manifests['train']['vocabulary'])
    template = ClipTail(original.visual, text)
    del original; torch.cuda.empty_cache()
    labels = torch.tensor([r['label'] for r in manifests['train']['samples']], device='cuda')
    weights = (torch.bincount(labels, minlength=len(text)).float()+1).rsqrt(); weights /= weights.mean()
    (args.output / 'validation_raw').mkdir()
    raw = {}
    for seed in SEEDS:
        head_state, warm_history = warm_head(text, teacher, labels, weights, seed)
        for spec in SPECS:
            torch.manual_seed(seed)
            model = copy.deepcopy(template); model.configure(spec); model.head.load_state_dict(head_state)
            parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
            groups = [{'params': model.head.parameters(), 'lr': .001}]
            if spec != 'frozen_head':
                groups += [{'params': [p for n, p in model.named_parameters() if not n.startswith('head.')], 'lr': 1e-5}]
            optimizer = torch.optim.AdamW(groups, weight_decay=.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6)
            generator = torch.Generator(device='cuda').manual_seed(seed+1000)
            directory = args.output / f'models/{spec}/seed{seed}'; directory.mkdir(parents=True)
            torch.save({k: v.cpu() for k, v in head_state.items()}, directory / 'warm_head.pt')
            write_json(directory / 'warm_history.json', warm_history)
            history = []
            for epoch in range(1, 7):
                started = time.time(); totals = np.zeros(3); seen = 0
                for indices in torch.randperm(len(labels), device='cuda', generator=generator).split(32):
                    optimizer.zero_grad(set_to_none=True)
                    if spec == 'frozen_head':
                        features = teacher[indices]; logits = model.head(features)
                    else:
                        logits, features = model(batch_tokens(train, indices.cpu().numpy()))
                    total, ce, anchor = objective(logits, features, teacher[indices], labels[indices], weights, spec)
                    assert torch.isfinite(total)
                    total.backward()
                    if epoch == 1 and seen == 0:
                        grad = {n: float(p.grad.norm()) for n, p in model.named_parameters() if p.requires_grad}
                        assert all(np.isfinite(list(grad.values()))) and sum(grad.values()) > 0
                        write_json(directory / 'gradients.json', grad)
                    optimizer.step(); totals += np.array([float(v.detach()) for v in (total, ce, anchor)])*len(indices); seen += len(indices)
                scheduler.step()
                changed = [n for n, v in model.state_dict().items() if not torch.equal(v, initial[n])]
                if spec == 'frozen_head':
                    assert changed and all(n.startswith('head.') for n in changed)
                else:
                    assert any(n.startswith('block.') for n in changed) and 'projection' in changed
                row = {'spec': spec, 'seed': seed, 'epoch': epoch, 'objects': seen,
                       'loss': float(totals[0]/seen), 'ce': float(totals[1]/seen), 'anchor': float(totals[2]/seen),
                       'changed_state': changed, 'seconds': time.time()-started}
                history.append(row); print(json.dumps(row), flush=True)
                if epoch in EPOCHS:
                    record = {'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                              'spec': spec, 'seed': seed, 'epoch': epoch, 'parameters': parameters,
                              'train_objects': len(labels), 'config_sha256': file_hash(args.output / 'config.json')}
                    path = directory / f'e{epoch}.pt'; torch.save(record, path)
                    write_json(path.with_suffix('.json'), {'sha256': file_hash(path), 'parameters': parameters})
                    scores = predict(model, validation, reference)
                    np.save(args.output / f'validation_raw/{spec}_e{epoch}_seed{seed}.npy', scores)
                    raw.setdefault((spec, epoch), ([], parameters))[0].append(scores)
            write_json(directory / 'history.json', history)
            del model, optimizer
    grid, selection = make_grid(raw, bases(args, 'validation', 'eval'), manifests['eval'])
    write_json(args.output / 'validation_grid.json', grid); write_json(args.output / 'selection.json', selection)
    paths = ['validation_grid.json', 'selection.json', 'tokens/train.json', 'tokens/eval.json']
    paths += [str(p.relative_to(args.output)) for folder in ('models', 'validation_raw') for p in (args.output / folder).rglob('*') if p.is_file()]
    freeze(args.output, 'SELECTION_FROZEN.json', paths)
    print(json.dumps({'stage': 'selection_frozen', 'selection': selection}), flush=True)


def evaluate(args):
    manifests = setup(args); check_frozen(args.output, 'SELECTION_FROZEN.json')
    assert not (args.output / 'TEST_FROZEN.json').exists()
    original, text, _ = load_model(args, manifests['train']['vocabulary'])
    template = ClipTail(original.visual, text); del original; torch.cuda.empty_cache()
    selection = read(args.output / 'selection.json'); records = []
    selections = {n: {'name': n, 'epoch': 0, 'base': n, 'weight': 0., 'parameters': 0} for n in REFERENCES}
    selections.update(selection['per_candidate']); selections.update({'locked_'+m: s for m, s in selection['locked'].items()})
    for role, split in (('validation', 'eval'), ('historical_test', 'dev')):
        folder = args.output / role; folder.mkdir()
        bank, reference = token_bank(args, split)
        base = bases(args, role, split); raw = {}
        for spec in SPECS:
            for seed in SEEDS:
                for epoch in EPOCHS:
                    model = copy.deepcopy(template); model.configure(spec)
                    record = torch.load(args.output / f'models/{spec}/seed{seed}/e{epoch}.pt', weights_only=True)
                    model.load_state_dict(record['state_dict'], strict=True)
                    scores = predict(model, bank, reference)
                    np.save(folder / f'{spec}_e{epoch}_seed{seed}_raw.npy', scores)
                    if split == 'eval':
                        np.testing.assert_array_equal(scores, np.load(args.output / f'validation_raw/{spec}_e{epoch}_seed{seed}.npy'))
                    raw[(spec, epoch, seed)] = scores; del model
        for name, selected in selections.items():
            for seed in SEEDS:
                scores = base[selected['name']] if selected['name'] in base else (
                    (1-selected['weight'])*base[selected['base']] + selected['weight']*z(raw[(selected['name'], selected['epoch'], seed)].astype(float)))
                np.save(folder / f'{name}_seed{seed}.npy', scores)
                row = {'role': role, 'method': name, 'seed': seed, 'selection': selected,
                       'summary': recount(scores, manifests[split])[0]}
                records.append(row)
        print(json.dumps({'stage': role+'_scored', 'records': [r for r in records if r['role'] == role]}), flush=True)
    write_json(args.output / 'results.json', {'records': records})
    freeze(args.output, 'TEST_FROZEN.json', ['results.json', 'tokens/dev.json'] + [str(p.relative_to(args.output))
           for folder in ('validation', 'historical_test') for p in (args.output / folder).iterdir()])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('preflight', 'cache', 'train-select', 'evaluate'))
    for name in ('output', 'reference', 'r4', 'r8', 'clip', 'design'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'eval', 'dev'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    with run_lock(args.output):
        if args.stage == 'cache':
            assert args.split is not None; cache(args, args.split)
        else:
            {'preflight': preflight, 'train-select': train_select, 'evaluate': evaluate}[args.stage](args)
