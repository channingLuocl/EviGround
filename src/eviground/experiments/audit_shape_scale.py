"""Independently recount scale exports and verify development-only selection."""
import argparse
from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import torch

from eviground.categories.category_common_v82 import file_hash, load_cloud, normalized_category
from eviground.evaluation.run_storage import write_json


normalized_category = lru_cache(maxsize=None)(normalized_category)


def recount(logits, manifest):
    order = np.argsort(-logits, axis=-1, kind='stable')
    vocabulary = manifest['vocabulary']
    hits, exact, top5, groups = [], [], [], defaultdict(list)
    for indices, row in zip(order, manifest['samples']):
        hit = normalized_category(vocabulary[indices[0]]) == normalized_category(row['category'])
        hits.append(hit)
        exact.append(vocabulary[indices[0]] == row['category'])
        top5.append(any(normalized_category(vocabulary[i]) == normalized_category(row['category']) for i in indices[:5]))
        groups[normalized_category(row['category'])].append(hit)
    return {'count': len(hits), 'correct': sum(hits), 'accuracy': float(np.mean(hits)),
            'exact_accuracy': float(np.mean(exact)), 'top5_accuracy': float(np.mean(top5)),
            'class_mean_accuracy': float(np.mean([np.mean(v) for v in groups.values()]))}, np.array(hits)


def audit(args):
    torch.set_num_threads(2)
    config = json.loads((args.output / 'config.json').read_text())
    for marker in ('SELECTION_FROZEN.json', 'RESULTS_FROZEN.json'):
        for name, digest in json.loads((args.output / marker).read_text())['files'].items():
            assert file_hash(args.output / name) == digest, name
    manifests = {s: json.loads((args.feature_root / f'manifests/{s}.json').read_text()) for s in ('train', 'dev', 'eval')}
    for s in manifests:
        assert file_hash(args.feature_root / f'manifests/{s}.json') == config['manifest_sha256'][s]
    prior = json.loads((Path(config['prior']) / 'config.json').read_text())
    geometry_checks = []
    for split in ('train', 'dev', 'eval'):
        cache = args.feature_root / f'features/{split}/features.npz'
        assert file_hash(cache) == prior['cache_sha256'][split]
        with np.load(cache, allow_pickle=False) as archive:
            meta = json.loads(str(archive['metadata']))
        for index in np.linspace(0, len(meta['records']) - 1, 3, dtype=int):
            row = meta['records'][index]
            source = manifests[split]['samples'][index]
            xyz, _, ids = load_cloud(source['cloud'])
            points = xyz[ids == row['bbox_id']]
            np.testing.assert_allclose(np.ptp(points, axis=0), row['bbox_3d'][3:], atol=1e-6, rtol=1e-6)
            geometry_checks.append([split, row['scene'], row['bbox_id']])
        if split != 'train':
            continue
        extents = np.maximum(np.array([r['bbox_3d'][3:] for r in meta['records']], dtype=np.float64), .001)
        x, y, h = extents.T
        length, width = np.maximum(x, y), np.minimum(x, y)
        features = {'size': np.stack([np.log(length), np.log(width), np.log(h)], 1),
                    'ratio': np.stack([np.log(length / width), np.log(h / np.sqrt(length * width))], 1)}
        for seed in (42, 43, 44):
            for mode, values in features.items():
                path = args.output / f'models/seed{seed}/{mode}/head.pt'
                checkpoint = torch.load(path, map_location='cpu', weights_only=True)
                state = checkpoint['state_dict']
                np.testing.assert_allclose(state['mean'].numpy(), values.mean(0), atol=2e-6)
                np.testing.assert_allclose(state['std'].numpy(), np.maximum(values.std(0), 1e-6), atol=2e-6)
                history = json.loads(path.with_name('history.json').read_text())
                assert len(history) == 40
                assert checkpoint['epoch'] == min(history, key=lambda r: r['dev']['category_loss'])['epoch']
    grids = json.loads((args.output / 'dev_grid.json').read_text())
    selection = json.loads((args.output / 'selection.json').read_text())
    for method, candidates in grids.items():
        for candidate in candidates:
            accuracies = []
            for seed in (42, 43, 44):
                with np.load(args.output / f'dev/seed{seed}/scores.npz') as parts:
                    a, c = candidate['weights']
                    logits = parts['base'] + a * parts['residual'] + c * parts['ratio' if method == 'E2-R' else 'size']
                summary, _ = recount(logits, manifests['dev'])
                accuracies.append(summary['accuracy'])
            np.testing.assert_allclose(accuracies, candidate['seed_accuracies'], atol=1e-12)
            assert abs(np.mean(accuracies) - candidate['mean_accuracy']) < 1e-12
        best = sorted(candidates, key=lambda r: (-r['mean_accuracy'], sum(r['weights']), *r['weights']))[0]
        assert best == selection[method]
    results = json.loads((args.output / 'results.json').read_text())
    for record in results['records']:
        seed, method = record['seed'], record['method']
        path = args.output / f'eval/seed{seed}/{method}'
        logits = np.load(path / 'logits.npy')
        with np.load(path.parent / 'components.npz') as parts:
            a, c = selection[method]['weights']
            expected = parts['base'] + a * parts['residual'] + c * parts['ratio' if method == 'E2-R' else 'size']
            baseline_summary, baseline_hit = recount(parts['base'], manifests['eval'])
        assert baseline_summary['correct'] == 2262
        np.testing.assert_array_equal(logits, expected)
        summary, hit = recount(logits, manifests['eval'])
        for key, value in summary.items():
            assert abs(record['summary'][key] - value) < 1e-12
        assert record['flips']['corrected'] == int((~baseline_hit & hit).sum())
        assert record['flips']['broken'] == int((baseline_hit & ~hit).sum())
        predictions = json.loads((path / 'predictions.json').read_text())['results']
        assert len(predictions) == len(manifests['eval']['samples']) == 3413
        for prediction, row, indices in zip(predictions, manifests['eval']['samples'], np.argsort(-logits, axis=-1, kind='stable')):
            assert set(prediction) == {'scene', 'bbox_id', 'status', 'predicted_object', 'top5'}
            assert (prediction['scene'], prediction['bbox_id']) == (row['scene'], row['bbox_id'])
            assert prediction['top5'] == [manifests['eval']['vocabulary'][i] for i in indices[:5]]
    report = {'status': 'passed', 'models': 6, 'evaluations': len(results['records']),
              'auditor_sha256': file_hash(Path(__file__)),
              'geometry_spotchecks': geometry_checks,
              'checks': ['frozen hashes', 'train-only normalization', '40 epochs and minimum-dev-CE checkpoints',
                         'all dev grid scores and shared weight selection', 'all evaluation metrics and flips',
                         'full logits composition and prediction identities']}
    write_json(args.output / 'audit.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--feature-root', type=Path, required=True)
    audit(parser.parse_args())
