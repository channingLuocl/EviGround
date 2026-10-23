"""Fixed-protocol residual scale experiments on frozen PointNeXt features."""
from eviground.paths import source_file
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from eviground.categories.category_common_v82 import ROOT, file_hash, normalized_category, run_lock
from eviground.classifiers.category_v10.model import adaptation_loss
from eviground.classifiers.category_v10.run import check_sources, read_manifest
from eviground.evaluation.run_storage import write_json


SEEDS = (42, 43, 44)
METHODS = ('E0', 'E1', 'E2', 'E2-R', 'E3')
GLOBAL_WEIGHTS = (0., .25, .5, .75, 1.)
SCALE_WEIGHTS = (0., .25, .5, 1., 2.)


def geometry_features(boxes, mode):
    """Use observed extents only; no centre/scene identity enters the head."""
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 6 or not np.isfinite(boxes).all():
        raise ValueError('Expected finite bbox[N,6]')
    if (boxes[:, 3:] < 0).any() or mode not in ('size', 'ratio'):
        raise ValueError('Invalid extents or geometry mode')
    x, y, h = np.maximum(boxes[:, 3:], .001).T
    length, width = np.maximum(x, y), np.minimum(x, y)
    values = (np.stack([np.log(length), np.log(width), np.log(h)], 1) if mode == 'size'
              else np.stack([np.log(length / width), np.log(h / np.sqrt(length * width))], 1))
    return torch.from_numpy(values.astype(np.float32))


class ScaleHead(nn.Module):
    def __init__(self, mean, std, categories=197):
        super().__init__()
        self.register_buffer('mean', mean.clone())
        self.register_buffer('std', std.clamp_min(1e-6).clone())
        self.net = nn.Sequential(nn.Linear(len(mean), 32), nn.GELU(), nn.Linear(32, categories))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, values):
        return self.net((values - self.mean) / self.std)


def choose_weights(candidates):
    """Maximum mean dev accuracy, then smallest total correction, then (a,c)."""
    return min(candidates, key=lambda r: (-r['mean_accuracy'], sum(r['weights']), *r['weights']))


def grid(method):
    if method == 'E0':
        return [(0., 0.)]
    if method == 'E1':
        return [(a, 0.) for a in GLOBAL_WEIGHTS]
    if method in ('E2', 'E2-R'):
        return [(0., c) for c in SCALE_WEIGHTS]
    return [(a, c) for a in GLOBAL_WEIGHTS for c in SCALE_WEIGHTS]


def combine(base, residual, size, ratio, method, weights):
    a, c = weights
    return base + a * residual + c * (ratio if method == 'E2-R' else size)


def correctness(logits, manifest):
    order = np.argsort(-np.asarray(logits), axis=1, kind='stable')
    names = np.array([normalized_category(c) for c in manifest['vocabulary']])
    targets = np.array([normalized_category(r['category']) for r in manifest['samples']])
    return names[order[:, 0]] == targets, (names[order[:, :5]] == targets[:, None]).any(1)


def score(logits, manifest):
    logits = np.asarray(logits)
    if logits.shape != (len(manifest['samples']), len(manifest['vocabulary'])) or not np.isfinite(logits).all():
        raise ValueError('Invalid prediction scores')
    hit, top5 = correctness(logits, manifest)
    order = np.argsort(-logits, axis=1, kind='stable')
    groups = defaultdict(list)
    for row, value in zip(manifest['samples'], hit):
        groups[normalized_category(row['category'])].append(bool(value))
    labels = np.array([r['label'] for r in manifest['samples']])
    known = labels >= 0
    return {'count': len(hit), 'correct': int(hit.sum()), 'accuracy': float(hit.mean()),
            'exact_accuracy': float(np.mean([manifest['vocabulary'][int(i)] == r['category']
                for i, r in zip(order[:, 0], manifest['samples'])])),
            'top5_accuracy': float(top5.mean()),
            'class_mean_accuracy': float(np.mean([np.mean(v) for v in groups.values()])),
            'class_mean_accuracy_in_vocabulary': float(np.mean([hit[labels == c].mean() for c in np.unique(labels[known])])),
            'category_loss': float(F.cross_entropy(torch.from_numpy(logits[known]), torch.from_numpy(labels[known]))),
            'targets_outside_vocabulary': int((~known).sum()), 'errors': 0}


def load_split(args, split, prior):
    path = args.feature_root / f'features/{split}/features.npz'
    if file_hash(path) != prior['cache_sha256'][split]:
        raise ValueError(f'Frozen cache hash differs: {split}')
    manifest = read_manifest(args.feature_root / f'manifests/{split}.json')
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive['metadata']))
        features = torch.from_numpy(archive['global_features']).float()
        labels = torch.from_numpy(archive['labels']).long()
    expected = [(r['scene'], r['bbox_id'], r['category']) for r in manifest['samples']]
    actual = [(r['scene'], r['bbox_id'], r['category']) for r in meta['records']]
    if (expected != actual or labels.tolist() != [r['label'] for r in manifest['samples']]
            or meta['code_sha256'] != prior['code_sha256'] or meta['split'] != split
            or meta['sources'] != manifest['sources'] or meta['vocabulary'] != manifest['vocabulary']
            or features.shape != (len(labels), 2 if split == 'train' else 1, 512)
            or not torch.isfinite(features).all() or (features.norm(dim=-1) <= 0).any()):
        raise ValueError(f'Cache/manifest mismatch: {split}')
    boxes = [r['bbox_3d'] for r in meta['records']]
    return {'features': F.normalize(features, dim=-1), 'labels': labels,
            'size': geometry_features(boxes, 'size'), 'ratio': geometry_features(boxes, 'ratio'),
            'manifest': manifest, 'meta': meta}


def load_global(args, seed, prior):
    path = args.prior / f'models/seed{seed}/global/head.pt'
    report = json.loads((args.prior / f'predictions/seed{seed}/global/results.json').read_text())
    if file_hash(path) != report['head_sha256']:
        raise ValueError('Old global checkpoint changed')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    if (config['train_cache_sha256'] != prior['cache_sha256']['train']
            or config['dev_cache_sha256'] != prior['cache_sha256']['dev']
            or config['code_sha256'] != prior['code_sha256'] or config['seed'] != seed
            or config['mode'] != 'global'):
        raise ValueError('Old global head provenance mismatch')
    history = json.loads(path.with_name('history.json').read_text())
    if len(history) != 40 or checkpoint['best_epoch'] != min(history, key=lambda r: r['dev']['category_loss'])['epoch']:
        raise ValueError('Old checkpoint selection mismatch')
    state = checkpoint['state_dict']
    if set(state) != {'category_text', 'global_adapter.weight'}:
        raise ValueError('Unexpected global state')
    return F.normalize(state['category_text'], dim=-1), state['global_adapter.weight']


def frozen_logits(data, text, weight):
    features = data['features'][:, 0]
    return (10 * features @ text.T).numpy(), (features @ weight.T).numpy()


def head_scores(head, values, device):
    head.eval()
    with torch.inference_mode():
        return torch.cat([head(v.to(device)).cpu() for v in values.split(256)]).numpy()


def train_head(args, train, dev, text, mode, seed):
    out = args.output / f'models/seed{seed}/{mode}'
    complete = out / 'COMPLETE.json'
    if complete.exists():
        marker = json.loads(complete.read_text())
        if file_hash(out / 'head.pt') != marker['sha256']:
            raise ValueError('Completed model changed')
        return
    if out.exists():
        raise ValueError(f'Incomplete model: preserve and use a fresh output root: {out}')
    out.mkdir(parents=True)
    torch.manual_seed(seed)
    values = train[mode]
    head = ScaleHead(values.mean(0), values.std(0, unbiased=False)).to(args.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.01)
    global_features = train['features'].to(args.device)
    base = 10 * global_features @ text.to(args.device).T
    labels = train['labels'].to(args.device)
    geometry = values.to(args.device)
    dev_base, _ = frozen_logits(dev, text, torch.zeros_like(text))
    best, history = float('inf'), []
    for epoch in range(1, 41):
        head.train()
        total = 0.
        for indices in torch.randperm(len(labels)).split(64):
            optimizer.zero_grad()
            residual = head(geometry[indices])
            loss = adaptation_loss({'logits': base[indices, 0] + residual},
                                   {'logits': base[indices, 1] + residual}, labels[indices], .1)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        report = score(dev_base + head_scores(head, dev[mode], args.device), dev['manifest'])
        history.append({'epoch': epoch, 'train_loss': total / len(labels), 'dev': report})
        write_json(out / 'history.json', history)
        if report['category_loss'] < best:
            best = report['category_loss']
            torch.save({'state_dict': {k: v.detach().cpu() for k, v in head.state_dict().items()},
                        'epoch': epoch, 'mode': mode, 'seed': seed}, out / 'head.tmp')
            (out / 'head.tmp').replace(out / 'head.pt')
        print(json.dumps({'stage': 'training', 'mode': mode, 'seed': seed, **history[-1]}), flush=True)
    write_json(complete, {'epochs': 40, 'sha256': file_hash(out / 'head.pt')})


def load_scale(args, seed, mode):
    path = args.output / f'models/seed{seed}/{mode}/head.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    state = checkpoint['state_dict']
    head = ScaleHead(state['mean'], state['std']).to(args.device)
    head.load_state_dict(state, strict=True)
    history = json.loads(path.with_name('history.json').read_text())
    if (len(history) != 40 or checkpoint['epoch'] != min(history, key=lambda r: r['dev']['category_loss'])['epoch']
            or checkpoint['mode'] != mode or checkpoint['seed'] != seed):
        raise ValueError('Scale head selection differs')
    return head


def input_config(args):
    prior = json.loads((args.prior / 'config.json').read_text())
    design = json.loads(args.design.read_text())
    if design['version'] != 'classifier-shape-scale-design-r1':
        raise ValueError('Wrong design')
    for split in ('train', 'dev', 'eval'):
        if file_hash(args.feature_root / f'manifests/{split}.json') != design['provenance']['manifests'][split]['sha256']:
            raise ValueError(f'Manifest differs from approved design: {split}')
    return {'design_sha256': file_hash(args.design), 'source_sha256': file_hash(Path(__file__)),
            'implementation_settings': design['phase1'],
            'dependency_sha256': {name: file_hash(source_file(name)) for name in
                ('category_common_v82.py', 'category_v10/model.py', 'category_v10/run.py', 'run_storage.py')},
            'prior_config_sha256': file_hash(args.prior / 'config.json'), 'torch': str(torch.__version__),
            'device': args.device, 'prior': str(args.prior.resolve()), 'feature_root': str(args.feature_root.resolve()),
            'manifest_sha256': {s: file_hash(args.feature_root / f'manifests/{s}.json') for s in ('train', 'dev', 'eval')},
            'global_head_sha256': {str(s): file_hash(args.prior / f'models/seed{s}/global/head.pt') for s in SEEDS}}, prior


def train_select(args, prior):
    train, dev = load_split(args, 'train', prior), load_split(args, 'dev', prior)
    check_sources(train['meta']['sources'], dev['meta']['sources'])
    # Use eval provenance for split isolation, never evaluation features or scores here.
    evaluation = read_manifest(args.feature_root / 'manifests/eval.json')
    for data in (train, dev):
        check_sources(data['meta']['sources'], evaluation['sources'])
    text0, _ = load_global(args, 42, prior)
    for seed in SEEDS:
        text, _ = load_global(args, seed, prior)
        if not torch.equal(text, text0):
            raise ValueError('Text vectors differ across old heads')
        for mode in ('size', 'ratio'):
            train_head(args, train, dev, text, mode, seed)
    scores = []
    for seed in SEEDS:
        text, weight = load_global(args, seed, prior)
        base, residual = frozen_logits(dev, text, weight)
        size = head_scores(load_scale(args, seed, 'size'), dev['size'], args.device)
        ratio = head_scores(load_scale(args, seed, 'ratio'), dev['ratio'], args.device)
        folder = args.output / f'dev/seed{seed}'
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / 'scores.npz', base=base, residual=residual, size=size, ratio=ratio)
        scores.append((base, residual, size, ratio))
    selection, grids = {}, {}
    for method in METHODS:
        rows = []
        for weights in grid(method):
            accuracies = [float(correctness(combine(*s, method, weights), dev['manifest'])[0].mean()) for s in scores]
            rows.append({'weights': list(weights), 'seed_accuracies': accuracies, 'mean_accuracy': float(np.mean(accuracies))})
        grids[method] = rows
        selection[method] = choose_weights(rows)
    write_json(args.output / 'dev_grid.json', grids)
    write_json(args.output / 'selection.json', selection)
    freeze = {str(p.relative_to(args.output)): file_hash(p) for p in args.output.rglob('*')
              if p.is_file() and p.name in ('head.pt', 'history.json', 'COMPLETE.json', 'config.json',
                                            'selection.json', 'dev_grid.json', 'scores.npz')}
    write_json(args.output / 'SELECTION_FROZEN.json', {'files': freeze, 'eval_features_read': False,
               'time': time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    print(json.dumps({'stage': 'selection_frozen', 'selection': selection}), flush=True)


def bootstrap(delta, scenes):
    rooms = sorted({s.split('_')[0] for s in scenes})
    groups = [np.array([i for i, s in enumerate(scenes) if s.split('_')[0] == room]) for room in rooms]
    sums = np.array([delta[g].sum() for g in groups])
    counts = np.array([len(g) for g in groups])
    rng = np.random.RandomState(20261019)
    samples = rng.randint(len(groups), size=(10000, len(groups)))
    differences = sums[samples].sum(1) / counts[samples].sum(1)
    return np.quantile(differences, [.025, .975]).tolist()


def evaluate(args, prior):
    marker = json.loads((args.output / 'SELECTION_FROZEN.json').read_text())
    for name, digest in marker['files'].items():
        if file_hash(args.output / name) != digest:
            raise ValueError(f'Frozen selection asset changed: {name}')
    folder = args.output / 'eval'
    if folder.exists():
        raise ValueError('Evaluation already exists; preserve frozen results')
    selection = json.loads((args.output / 'selection.json').read_text())
    data = load_split(args, 'eval', prior)
    manifest, records, hits = data['manifest'], [], defaultdict(list)
    baseline_hit = None
    for seed in SEEDS:
        text, weight = load_global(args, seed, prior)
        base, residual = frozen_logits(data, text, weight)
        old = np.load(args.prior / 'frozen_cache_control/logits.npy', allow_pickle=False)
        if not np.allclose(base, old, rtol=1e-5, atol=1e-5) or score(base, manifest)['correct'] != 2262:
            raise ValueError('Original baseline not reproduced')
        baseline_hit, baseline_top5 = correctness(base, manifest)
        size = head_scores(load_scale(args, seed, 'size'), data['size'], args.device)
        ratio = head_scores(load_scale(args, seed, 'ratio'), data['ratio'], args.device)
        parts = folder / f'seed{seed}'
        parts.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(parts / 'components.npz', base=base, residual=residual, size=size, ratio=ratio)
        for method in METHODS:
            logits = combine(base, residual, size, ratio, method, selection[method]['weights'])
            summary = score(logits, manifest)
            hit, _ = correctness(logits, manifest)
            hits[method].append(hit)
            flips = {'corrected': int((~baseline_hit & hit).sum()), 'broken': int((baseline_hit & ~hit).sum()),
                     'net': int(hit.sum() - baseline_hit.sum()), 'both_wrong': int((~baseline_hit & ~hit).sum()),
                     'corrected_inside_baseline_top5': int((~baseline_hit & hit & baseline_top5).sum()),
                     'corrected_outside_baseline_top5': int((~baseline_hit & hit & ~baseline_top5).sum())}
            output = parts / method
            output.mkdir()
            np.save(output / 'logits.npy', logits, allow_pickle=False)
            order = np.argsort(-logits, axis=1, kind='stable')
            predictions = [{'scene': r['scene'], 'bbox_id': r['bbox_id'], 'status': 'completed',
                            'predicted_object': manifest['vocabulary'][int(indices[0])],
                            'top5': [manifest['vocabulary'][int(i)] for i in indices[:5]]}
                           for r, indices in zip(manifest['samples'], order)]
            write_json(output / 'predictions.json', {'results': predictions})
            record = {'seed': seed, 'method': method, 'weights': selection[method]['weights'],
                      'summary': summary, 'flips': flips}
            write_json(output / 'results.json', record)
            records.append(record)
    aggregates = {}
    for method in METHODS:
        rows = [r for r in records if r['method'] == method]
        aggregates[method] = {k: {'mean': float(np.mean([r['summary'][k] for r in rows])),
                                  'std': float(np.std([r['summary'][k] for r in rows], ddof=1))}
                              for k in ('accuracy', 'exact_accuracy', 'top5_accuracy', 'class_mean_accuracy')}
        delta = np.mean(hits[method], 0) - baseline_hit.astype(float)
        aggregates[method]['delta_accuracy_95ci_room_bootstrap'] = bootstrap(delta, [r['scene'] for r in manifest['samples']])
    write_json(args.output / 'results.json', {'records': records, 'aggregate': aggregates,
               'evaluation_status': 'previously_inspected_validation_not_unseen_test'})
    table = ['method\tseed\tcount\tcorrect\tTrue\tExact\tTop5\tClassMean\tErrors\tcorrected\tbroken\tnet']
    for r in records:
        s, f = r['summary'], r['flips']
        table.append(f"{r['method']}\t{r['seed']}\t{s['count']}\t{s['correct']}\t{s['accuracy']:.2%}\t{s['exact_accuracy']:.2%}\t{s['top5_accuracy']:.2%}\t{s['class_mean_accuracy']:.2%}\t0\t{f['corrected']}\t{f['broken']}\t{f['net']}")
    (args.output / 'table.tsv').write_text('\n'.join(table) + '\n')
    write_json(args.output / 'RESULTS_FROZEN.json', {'files': {str(p.relative_to(args.output)): file_hash(p)
        for p in [*folder.rglob('*'), args.output / 'results.json', args.output / 'table.tsv'] if p.is_file()}})
    print(json.dumps({'stage': 'evaluated', 'aggregate': aggregates}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('train-select', 'evaluate'))
    for name in ('feature-root', 'prior', 'output', 'design'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(8)
    config, prior = input_config(args)
    with run_lock(args.output):
        path = args.output / 'config.json'
        if path.exists():
            if json.loads(path.read_text()) != config:
                raise ValueError('Configuration changed; use a fresh output')
        else:
            if args.stage != 'train-select':
                raise ValueError('Train and freeze selection first')
            write_json(path, config)
        if args.stage == 'train-select':
            if (args.output / 'SELECTION_FROZEN.json').exists():
                raise ValueError('Selection already frozen')
            train_select(args, prior)
        else:
            evaluate(args, prior)


if __name__ == '__main__':
    main()
