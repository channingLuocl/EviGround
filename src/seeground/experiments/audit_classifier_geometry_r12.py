"""Independent frozen-feature samples, NumPy heads, selections and metrics for r12."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from seeground.experiments.audit_classifier_balance_r4 import paired_bootstrap
from seeground.experiments.audit_classifier_residual_r9 import pick
from seeground.experiments.audit_pair_train_test import recount
from seeground.experiments.audit_pair_views import z
from seeground.categories.category_common_v82 import file_hash
from seeground.classifiers.classifier_geometry_r12 import (DIMENSIONS, REFERENCES, SEEDS, SPECS, bases,
    embedding, instances, make_classifier, read)
from seeground.classifiers.geometry_classifier_r12 import MODALITIES
from seeground.experiments.run_rgb_experiment import check_frozen
from seeground.evaluation.run_storage import write_json


def independent_head(record, values):
    state = {k: v.numpy().astype(float) for k, v in record['state_dict'].items()}
    x = np.concatenate([values[k] for k in MODALITIES[record['spec']]], axis=-1).astype(float)
    if record['spec'].endswith('_linear'):
        return x @ state['head.weight'].T + state['head.bias']
    x = (x-x.mean(-1, keepdims=True))/np.sqrt(x.var(-1, keepdims=True)+1e-5)
    x = x*state['head.0.weight']+state['head.0.bias']
    x = x @ state['head.1.weight'].T+state['head.1.bias']
    x = F.gelu(torch.from_numpy(x)).numpy()
    return x @ state['head.4.weight'].T+state['head.4.bias']


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
    feature_max = 0.; samples = 0
    for name in ('uni3d', 'ulip2'):
        classifier = make_classifier(args, name)
        for split, manifest in manifests.items():
            path = root / f'features/{name}_{split}.npy'; seal = read(path.with_suffix('.json'))
            assert file_hash(path) == seal['sha256']
            assert file_hash(root / f'manifests/{split}.json') == seal['manifest_sha256']
            assert file_hash(root / f'features/{name}_text.npy') == seal['text_sha256']
            assert seal['encoder_state'] == 'exactly_unchanged'
            values = np.load(path)
            assert values.shape == (len(manifest['samples']), DIMENSIONS[name]) and values.dtype == np.float32
            assert np.isfinite(values).all()
            np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1., atol=2e-6, rtol=0)
            np.testing.assert_array_equal(classifier.text_embeddings.cpu().numpy(), np.load(root / f'features/{name}_text.npy'))
            for i, row, xyz, rgb in instances(manifest, [0, len(values)//2, len(values)-1]):
                from seeground.categories.category_common_v82 import instance_seed
                actual = embedding(classifier, name, xyz, rgb, instance_seed(42, row['scene'], row['bbox_id'])).cpu().numpy()
                np.testing.assert_allclose(actual, values[i], atol=2e-5, rtol=2e-5)
                feature_max = max(feature_max, float(np.max(np.abs(actual-values[i])))); samples += 1
        del classifier; torch.cuda.empty_cache()
    base_validation = bases(args, 'validation', 'eval')
    grid = read(root / 'validation_grid.json'); assert len(grid) == 111
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
    records = read(root / 'results.json')['records']; comparisons = {}; max_head = 0.; forwards = 0
    for role, split in (('validation', 'eval'), ('historical_test', 'dev')):
        data = {n: np.load(root / f'features/{n}_{split}.npy') for n in ('uni3d', 'ulip2')}
        data['shape'] = np.load(args.reference / f'features/shape_{split}.npz')['features']
        rgb_path = args.reference / ('features/rgb_dev.npz' if split == 'dev' else f'features/rgb_{split}/features.npz')
        rgb = np.load(rgb_path); assert rgb['valid'].all(); data['rgb'] = rgb['features']
        base, raw, hits = bases(args, role, split), {}, {}
        for spec in SPECS:
            for seed in SEEDS:
                path = root / f'models/{spec}/seed{seed}/e40.pt'
                assert file_hash(path) == read(path.with_suffix('.json'))['sha256']
                record = torch.load(path, weights_only=True, map_location='cpu')
                assert (record['spec'], record['seed'], record['epoch'], record['train_objects']) == (spec, seed, 40, 28730)
                assert record['config_sha256'] == file_hash(root / 'config.json') and record['parameter_change_l2'] > 0
                history = read(path.parent / 'history.json'); assert len(history) == 40 and np.isfinite(history).all()
                expected = independent_head(record, data)
                saved = np.load(root / role / f'{spec}_e40_seed{seed}_raw.npy')
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
    result = {'status': 'passed', 'sampled_frozen_geometry_replays': samples, 'maximum_feature_abs': feature_max,
        'independent_head_forwards': forwards, 'maximum_head_abs': max_head, 'validation_grid_rows': len(grid),
        'prediction_records_recounted': len(records), 'comparisons': comparisons,
        'scope': 'All artifact/input/cache hashes,18 sampled geometry forwards,36 NumPy head forwards/exact Top5,111-grid selection,all metrics and room bootstrap; not full backbone/optimizer replay',
        'inherited_reference_limit': 'r8 rotation precision limit'}
    write_json(root / 'AUDIT.json', result); print(__import__('json').dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'reference', 'r4', 'r8', 'uni3d', 'ulip2'):
        parser.add_argument('--'+name, type=Path, required=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    audit(parser.parse_args())
