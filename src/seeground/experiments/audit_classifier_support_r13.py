"""Independently recount support weights, residual forwards, selections and metrics."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from seeground.experiments.audit_classifier_balance_r4 import paired_bootstrap
from seeground.experiments.audit_classifier_residual_r9 import pick
from seeground.experiments.audit_pair_train_test import recount
from seeground.experiments.audit_pair_views import z
from seeground.categories.category_common_v82 import file_hash
from seeground.classifiers.classifier_support_r13 import REFERENCES, SEEDS, SPECS, bases, read
from seeground.experiments.run_rgb_experiment import check_frozen
from seeground.evaluation.run_storage import write_json


def independent_head(record, values):
    state = {k: v.numpy().astype(float) for k, v in record['state_dict'].items()}
    x = values.astype(float)
    residual = (x-x.mean(-1, keepdims=True))/np.sqrt(x.var(-1, keepdims=True)+1e-5)
    residual = residual*state['residual.0.weight']+state['residual.0.bias']
    residual = residual @ state['residual.1.weight'].T+state['residual.1.bias']
    residual = F.gelu(torch.from_numpy(residual)).numpy()
    residual = residual @ state['residual.4.weight'].T+state['residual.4.bias']
    return 10*x@state['text'].T + state['gains']*residual


def verify_support(manifest, support):
    samples = manifest['samples']; labels = np.array([r['label'] for r in samples])
    cells = Counter((r['label'], r['scene'].split('_')[0]) for r in samples)
    counts = np.bincount(labels, minlength=197).astype(float)
    rooms = np.array([sum(c == i for c, _ in cells) for i in range(197)], dtype=float)
    root_mass = np.array([sum(n**.5 for (c, _), n in cells.items() if c == i) for i in range(197)])
    factor = np.array([counts[r['label']] / (cells[(r['label'], r['scene'].split('_')[0])]**.5*root_mass[r['label']]) for r in samples])
    gain = np.sqrt(counts/(counts+16)*rooms/(rooms+4))
    weight = (counts+1)**-.5; weight /= weight.mean()
    expected = {'labels': labels, 'counts': counts, 'rooms': rooms, 'gains': gain,
                'object_weight': weight[labels], 'room_factor': factor}
    for name, values in expected.items():
        np.testing.assert_allclose(support[name], values, atol=2e-6, rtol=3e-7)
    for c in np.flatnonzero(counts):
        np.testing.assert_allclose(support['room_factor'][labels==c].sum(dtype=float), counts[c], atol=1e-3, rtol=0)
    return expected


def audit(args):
    root = args.output
    for marker in ('PROTOCOL_FROZEN.json', 'SELECTION_FROZEN.json', 'TEST_FROZEN.json'):
        check_frozen(root, marker)
    config = read(root / 'config.json')
    for name, digest in config['inputs'].items():
        assert file_hash(name) == digest, name
    for name, digest in config['source'].items():
        assert file_hash(root / 'source_snapshot' / name) == digest
    manifests = {s: read(root / f'manifests/{s}.json') for s in ('train', 'eval', 'dev')}
    assert [len(m['samples']) for m in manifests.values()] == [28730, 404, 3413]
    rooms = {s: {r['scene'].split('_')[0] for r in m['samples']} for s, m in manifests.items()}
    assert all(not rooms[a] & rooms[b] for a, b in (('train', 'eval'), ('train', 'dev'), ('eval', 'dev')))
    assert read(root / 'PREFLIGHT.json')['status'] == 'passed'
    support = np.load(root / 'train_support.npz'); expected_support = verify_support(manifests['train'], support)
    grid = read(root / 'validation_grid.json'); assert len(grid) == 75
    base_validation = bases(args, 'validation', 'eval')
    for row in grid:
        arrays = [base_validation[row['name']]]*3 if row['name'] in base_validation else [
            (1-row['weight'])*base_validation[row['base']] + row['weight']*z(np.load(root /
            f'validation_raw/{row["name"]}_e40_seed{seed}.npy').astype(float)) for seed in SEEDS]
        metrics = [recount(v, manifests['eval'])[0] for v in arrays]
        assert [m['correct'] for m in metrics] == row['correct_by_seed']
        assert np.mean([m['correct'] for m in metrics]) == row['mean_correct']
        assert abs(np.mean([m['class_mean_accuracy'] for m in metrics])-row['mean_macro']) < 1e-12
    selection = read(root / 'selection.json'); reference = next(r for r in grid if r['name'] == 'r8_M')
    for mode in ('P', 'B', 'M'):
        assert pick(grid, reference, mode) == selection['locked'][mode]
    for spec in SPECS:
        assert pick([r for r in grid if r['name'] == spec], reference, 'P') == selection['per_candidate'][spec]
    results = read(root / 'results.json'); records = results['records']; assert len(records) == 84
    for name, digest in results['feature_inputs'].items():
        assert file_hash(name) == digest
    text = np.load(args.reference / 'features/text.npy')
    comparisons = {}; max_head = 0.; forwards = 0
    for role, split in (('validation', 'eval'), ('historical_test', 'dev')):
        path = args.reference / ('features/rgb_dev.npz' if split == 'dev' else f'features/rgb_{split}/features.npz')
        rgb = np.load(path); assert rgb['valid'].all(); data = rgb['features']
        assert data.shape == (len(manifests[split]['samples']), 512) and data.dtype == np.float32
        base, raw, hits = bases(args, role, split), {}, {}
        for spec in SPECS:
            for seed in SEEDS:
                path = root / f'models/{spec}/seed{seed}/e40.pt'
                assert file_hash(path) == read(path.with_suffix('.json'))['sha256']
                record = torch.load(path, weights_only=True, map_location='cpu')
                assert (record['spec'], record['seed'], record['epoch'], record['train_objects']) == (spec, seed, 40, 28730)
                assert record['config_sha256'] == file_hash(root / 'config.json') and record['parameter_change_l2'] > 0
                state = record['state_dict']; assert all(torch.isfinite(v).all() for v in state.values())
                assert sum(v.numel() for k, v in state.items() if k.startswith('residual.')) == record['parameters']
                np.testing.assert_array_equal(state['text'].numpy(), text)
                gain = support['gains'] if spec.startswith('support') else np.ones(197, dtype=np.float32)
                np.testing.assert_array_equal(state['gains'].numpy(), gain)
                np.testing.assert_allclose(gain, expected_support['gains'] if spec.startswith('support') else 1., atol=2e-7, rtol=0)
                history = read(path.parent / 'history.json'); assert len(history) == 40 and np.isfinite(history).all()
                expected = independent_head(record, data); saved = np.load(root / role / f'{spec}_e40_seed{seed}_raw.npy')
                np.testing.assert_allclose(expected, saved, atol=5e-5, rtol=2e-5)
                np.testing.assert_array_equal(np.argsort(-expected, axis=1, kind='stable')[:, :5],
                                              np.argsort(-saved, axis=1, kind='stable')[:, :5])
                assert recount(expected, manifests[split])[0] == recount(saved, manifests[split])[0]
                max_head = max(max_head, float(np.max(np.abs(expected-saved)))); forwards += 1
                raw[(spec, seed)] = saved
                if split == 'eval':
                    np.testing.assert_array_equal(saved, np.load(root / f'validation_raw/{spec}_e40_seed{seed}.npy'))
        for row in [r for r in records if r['role'] == role]:
            selected, seed = row['selection'], row['seed']
            if row['method'].endswith('_pure'):
                saved = raw[(selected['name'], seed)]
            else:
                expected = base[selected['name']] if selected['name'] in base else (
                    (1-selected['weight'])*base[selected['base']] + selected['weight']*z(raw[(selected['name'], seed)].astype(float)))
                saved = np.load(root / role / f'{row["method"]}_seed{seed}.npy')
                np.testing.assert_array_equal(saved, expected)
            summary, hit, _, _ = recount(saved, manifests[split]); assert summary == row['summary']
            hits.setdefault(row['method'], []).append(hit)
        for mode in ('P', 'B', 'M'):
            for reference in REFERENCES:
                comparisons[f'{role}:locked_{mode}-{reference}'] = paired_bootstrap(
                    np.array(hits['locked_'+mode]), np.array(hits[reference]), manifests[split])
    result = {'status': 'passed', 'independent_head_forwards': forwards, 'maximum_head_abs': max_head,
        'training_support_objects_recounted': 28730, 'validation_grid_rows': len(grid),
        'prediction_records_recounted': len(records), 'comparisons': comparisons,
        'scope': 'All artifact/input hashes, TRAIN support NumPy recount, frozen text/gains,24 NumPy head forwards/exact Top5,75-grid selection,all metrics and room bootstrap; no optimizer/renderer replay',
        'inherited_reference_limit': 'r8 rotation precision limit'}
    write_json(root / 'AUDIT.json', result); print(__import__('json').dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'reference', 'r4', 'r8'):
        parser.add_argument('--'+name, type=Path, required=True)
    torch.set_num_threads(4)
    audit(parser.parse_args())
