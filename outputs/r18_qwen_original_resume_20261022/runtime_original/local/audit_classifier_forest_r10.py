"""Independently accumulate leaf distributions for every sealed r10 forest."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from audit_classifier_balance_r4 import paired_bootstrap
from audit_pair_train_test import recount
from audit_pair_views import z
from category_common_v82 import file_hash
from run_rgb_experiment import check_frozen
from run_storage import write_json


def read(path):
    return json.loads(Path(path).read_text())


def pick(rows, ref, mode):
    eligible = [r for r in rows if (mode == 'P' or (
                r['mean_macro'] >= ref['mean_macro'] - 1e-10 if mode == 'B' else r['mean_correct'] >= ref['mean_correct']))]
    key = lambda r: ((-r['mean_macro'], -r['mean_correct']) if mode == 'M' else (-r['mean_correct'], -r['mean_macro'])) + (
        r['nodes'], r['name'], r['base'], r['weight'])
    return sorted(eligible, key=key)[0]


def audit(args):
    import joblib
    root = args.output
    for marker in ('PROTOCOL_FROZEN.json', 'SELECTION_FROZEN.json', 'TEST_FROZEN.json'):
        check_frozen(root, marker)
    config = read(root / 'config.json')
    for path, digest in config['inputs'].items():
        assert file_hash(path) == digest
    for name, digest in config['source'].items():
        assert file_hash(root / 'source_snapshot' / name) == digest
    manifests = {s: read(root / f'manifests/{s}.json') for s in ('train', 'eval', 'dev')}
    assert [len(m['samples']) for m in manifests.values()] == [28730, 404, 3413]
    rooms = {s: {r['scene'].split('_')[0] for r in m['samples']} for s, m in manifests.items()}
    assert not rooms['train'] & rooms['eval'] and not rooms['train'] & rooms['dev'] and not rooms['eval'] & rooms['dev']
    def inputs(split):
        path = args.reference / ('features/rgb_dev.npz' if split == 'dev' else f'features/rgb_{split}/features.npz')
        record = np.load(path)
        assert record['valid'].all()
        rgb = record['features'].astype(np.float32)
        shape = np.load(args.reference / f'features/shape_{split}.npz')['features'].astype(np.float32)
        return {'rgb': rgb, 'joint': np.concatenate((shape, rgb), axis=1)}
    expected_classes = np.unique([r['label'] for r in manifests['train']['samples']])
    checked = 0
    maximum = 0.
    for spec in ('rgb_plain', 'rgb_balanced', 'joint_plain', 'joint_balanced'):
        for seed in (42, 43, 44):
            folder = root / f'models/{spec}/seed{seed}'
            record = read(folder / 'model.json')
            assert file_hash(folder / 'model.joblib') == record['sha256']
            assert record['config_sha256'] == file_hash(root / 'config.json')
            assert record['train_objects'] == 28730 and record['fit_n_jobs'] == 8
            model = joblib.load(folder / 'model.joblib')
            assert model.n_jobs == 1 and len(model.estimators_) == 256 and model.max_leaf_nodes == 1024
            assert model.get_params() == record['parameters']
            np.testing.assert_array_equal(model.classes_, expected_classes)
            assert sum(t.tree_.node_count for t in model.estimators_) == record['nodes']
            for role, split in (('validation', 'eval'), ('historical_test', 'dev')):
                x = inputs(split)[spec.split('_')[0]]
                summed = np.zeros((len(x), len(model.classes_)))
                for tree in model.estimators_:
                    leaf = tree.apply(x)
                    counts = tree.tree_.value[leaf, 0].copy()
                    counts /= counts.sum(1, keepdims=True)
                    summed += counts
                    checked += 1
                actual = np.zeros((len(x), 197))
                actual[:, model.classes_.astype(int)] = summed / len(model.estimators_)
                saved = np.load(root / role / f'{spec}_seed{seed}_probability.npy')
                np.testing.assert_allclose(saved, actual, atol=1e-13, rtol=1e-13)
                maximum = max(maximum, float(np.max(np.abs(saved - actual))))
                np.testing.assert_array_equal(np.argsort(-saved, axis=1, kind='stable')[:, :5],
                                              np.argsort(-actual, axis=1, kind='stable')[:, :5])
                if split == 'eval':
                    np.testing.assert_array_equal(saved, np.load(root / f'validation_raw/{spec}_seed{seed}.npy'))
            print(json.dumps({'stage': 'forest_replayed', 'spec': spec, 'seed': seed}), flush=True)
    text = np.load(args.reference / 'features/text.npy').astype(float)
    def refs(role, split):
        shape = np.load(args.reference / f'features/shape_{split}.npz')['features'].astype(float)
        return {'S0': z(10 * shape @ text.T), 'r4_M': np.load(args.r4 / f'{role}/locked_M_v0.npy').astype(float),
                'r8_M': np.load(args.r8 / f'{role}/locked_M.npy').astype(float)}
    base = refs('validation', 'eval')
    grid = read(root / 'validation_grid.json')
    assert len(grid) == 87
    for row in grid:
        values = [base[row['name']]] * 3 if row['name'] in base else [(1 - row['weight']) * base[row['base']] + row['weight'] * z(
            np.log(np.maximum(np.load(root / f'validation_raw/{row["name"]}_seed{seed}.npy'), 1e-6))) for seed in (42, 43, 44)]
        summary = [recount(v, manifests['eval'])[0] for v in values]
        assert [s['correct'] for s in summary] == row['correct_by_seed']
        assert np.mean([s['correct'] for s in summary]) == row['mean_correct']
        assert abs(np.mean([s['class_mean_accuracy'] for s in summary]) - row['mean_macro']) < 1e-12
    selection = read(root / 'selection.json')
    reference = next(r for r in grid if r['name'] == 'r8_M')
    for mode in ('P', 'B', 'M'):
        assert pick(grid, reference, mode) == selection['locked'][mode]
    for name in ('rgb_plain', 'rgb_balanced', 'joint_plain', 'joint_balanced'):
        assert pick([r for r in grid if r['name'] == name], reference, 'P') == selection['per_candidate'][name]
    records = read(root / 'results.json')['records']
    comparisons = {}
    for role, split in (('validation', 'eval'), ('historical_test', 'dev')):
        base = refs(role, split)
        hits = {}
        for row in [r for r in records if r['role'] == role]:
            sel, seed = row['selection'], row['seed']
            if sel is None:
                expected = np.log(np.maximum(np.load(root / role / f'{row["method"][4:]}_seed{seed}_probability.npy'), 1e-6))
                summary = recount(expected, manifests[split])[0]
            else:
                expected = base[sel['name']] if sel['name'] in base else (1 - sel['weight']) * base[sel['base']] + sel['weight'] * z(
                    np.log(np.maximum(np.load(root / role / f'{sel["name"]}_seed{seed}_probability.npy'), 1e-6)))
                saved = np.load(root / role / f'{row["method"]}_seed{seed}.npy')
                np.testing.assert_array_equal(saved, expected)
                summary, hit, _, _ = recount(saved, manifests[split])
                hits.setdefault(row['method'], []).append(hit)
            assert summary == row['summary']
        for mode in ('P', 'B', 'M'):
            for name in ('S0', 'r4_M', 'r8_M'):
                comparisons[f'{role}:locked_{mode}-{name}'] = paired_bootstrap(np.array(hits['locked_' + mode]), np.array(hits[name]), manifests[split])
    result = {'status': 'passed', 'tree_forwards_replayed': checked, 'maximum_probability_abs': maximum,
              'validation_grid_rows': len(grid), 'prediction_records_recounted': len(records), 'comparisons': comparisons,
              'scope': 'Every tree leaf distribution and class mapping, exact Top-5, full validation grid/choices and all metrics/bootstrap; not a refit of the forests.',
              'inherited_reference_limit': 'r8 rotation precision limit'}
    write_json(root / 'AUDIT.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for n in ('output', 'reference', 'r4', 'r8'):
        parser.add_argument('--' + n, type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.output / 'deps_existing'))
    audit(args)
