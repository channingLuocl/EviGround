"""Verify every r9 frozen checkpoint, forward, validation choice and reported metric."""
import argparse
from pathlib import Path
import json

import numpy as np
import torch
from torch.nn import functional as F

from eviground.experiments.audit_classifier_balance_r4 import paired_bootstrap
from eviground.experiments.audit_pair_train_test import recount
from eviground.experiments.audit_pair_views import z
from eviground.categories.category_common_v82 import file_hash, instance_seed, load_cloud
from eviground.classifiers.classifier_backbone_r5 import make_encoder
from eviground.classifiers.pointcloud_classifier_v82 import prepare_points
from eviground.classifiers.pointnext_residual_r9 import ADAPTER, STEM, ResidualPointNext
from eviground.experiments.run_rgb_experiment import check_frozen
from eviground.evaluation.run_storage import write_json


def read(path):
    return json.loads(Path(path).read_text())


def pick(rows, ref, mode):
    eligible = [r for r in rows if (mode == 'P' or
                (r['mean_macro'] >= ref['mean_macro'] - 1e-10 if mode == 'B' else r['mean_correct'] >= ref['mean_correct']))]
    key = lambda r: ((-r['mean_macro'], -r['mean_correct']) if mode == 'M' else (-r['mean_correct'], -r['mean_macro'])) + (
        r['parameters'], r['name'], r['epoch'], r['base'], r['weight'])
    return sorted(eligible, key=key)[0]


def audit(args):
    root = args.output
    for marker in ('PROTOCOL_FROZEN.json', 'SELECTION_FROZEN.json', 'TEST_FROZEN.json'):
        check_frozen(root, marker)
    config = read(root / 'config.json')
    for path, digest in config['inputs'].items():
        assert file_hash(path) == digest, path
    for name, digest in config['source'].items():
        assert file_hash(root / 'source_snapshot' / name) == digest
    assert read(root / 'PREFLIGHT.json')['status'] == 'passed'
    manifests = {s: read(root / f'manifests/{s}.json') for s in ('train', 'eval', 'dev')}
    assert [len(m['samples']) for m in manifests.values()] == [28730, 404, 3413]
    rooms = {s: {r['scene'].split('_')[0] for r in m['samples']} for s, m in manifests.items()}
    assert not rooms['train'] & rooms['eval'] and not rooms['train'] & rooms['dev'] and not rooms['eval'] & rooms['dev']
    original = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    expected = {(STEM + '.base.' + n[len(STEM) + 1:] if n.startswith(STEM + '.') else n): v
                for n, v in original.items()}
    paths = {}
    for split in manifests:
        path = args.color / f'points/{split}.npy'
        assert file_hash(path) == read(path.with_suffix('.json'))['sha256']
        paths[split] = path
        points = np.load(path, mmap_mode='r')
        for i in (0, len(points) // 2, len(points) - 1):
            row = manifests[split]['samples'][i]
            xyz, rgb, ids = load_cloud(row['cloud'])
            obj, color = xyz[ids == row['bbox_id']], rgb[ids == row['bbox_id']]
            seed = instance_seed(42, row['scene'], row['bbox_id'])
            np.testing.assert_array_equal(points[i, :, :4], prepare_points(obj, None, 'pointnext', seed))
            indices = np.random.RandomState(seed).choice(len(obj), 2048, replace=len(obj) < 2048)
            np.testing.assert_array_equal(points[i, :, 4:], color[indices].astype(np.float32) / 255 - .5)
    text = np.load(args.reference / 'features/text.npy').astype(float)
    counts = {'checkpoints': 0, 'frozen_arrays': 0, 'raw_forwards': 0}
    maximum = 0.
    for spec in ('geometry_kd', 'color_ce', 'color_kd'):
        for seed in (42, 43, 44):
            directory = root / f'models/{spec}/seed{seed}'
            history = read(directory / 'history.json')
            assert [r['epoch'] for r in history] == list(range(1, 9))
            for row in history:
                assert row['objects'] == 28730 and row['frozen_state'] == 'exactly_unchanged'
                assert np.isfinite([row['loss'], row['ce'], row['kd']]).all()
                assert row['kd'] >= -1e-6
                assert abs(row['loss'] - row['ce'] - (0 if spec == 'color_ce' else row['kd'])) < 1e-6
            gradient = read(directory / 'gradients.json')
            assert all(k.startswith(ADAPTER) for k in gradient['all'])
            assert np.isfinite(list(gradient['all'].values())).all() and sum(gradient['all'].values()) > 0
            assert (gradient['rgb_norm'] == 0) if spec == 'geometry_kd' else (gradient['rgb_norm'] > 0)
            for epoch in (4, 8):
                state = torch.load(directory / f'e{epoch}.pt', map_location='cpu', weights_only=True)
                assert (state['spec'], state['seed'], state['epoch']) == (spec, seed, epoch)
                assert state['config_sha256'] == file_hash(root / 'config.json')
                for name, value in expected.items():
                    assert torch.equal(value, state['encoder'][name]), name
                    counts['frozen_arrays'] += 1
                assert all(k in expected or k.startswith(ADAPTER) for k in state['encoder'])
                assert state['encoder'][ADAPTER + '2.weight'].norm() > 0
                model = ResidualPointNext(make_encoder(args), spec)
                model.load_state_dict(state['encoder'], strict=True)
                assert sum(p.numel() for p in model.parameters() if p.requires_grad) == state['parameters'] == 2592
                counts['checkpoints'] += 1
                for role, split in (('validation', 'eval'), ('historical_test', 'dev')):
                    cache = np.load(paths[split], mmap_mode='r')
                    values = []
                    with torch.inference_mode():
                        for start in range(0, len(cache), 32):
                            x = torch.from_numpy(np.array(cache[start:start + 32])).cuda()
                            embedding = F.normalize(model(x).float(), dim=-1).cpu().numpy().astype(float)
                            values.append(10 * embedding @ text.T)
                    actual = np.concatenate(values)
                    saved = np.load(root / role / f'{spec}_e{epoch}_seed{seed}_raw.npy')
                    np.testing.assert_allclose(actual, saved, atol=2e-5, rtol=2e-5)
                    np.testing.assert_array_equal(np.argsort(-actual, axis=1, kind='stable')[:, :5],
                                                  np.argsort(-saved, axis=1, kind='stable')[:, :5])
                    maximum = max(maximum, float(np.max(np.abs(actual - saved))))
                    assert recount(actual, manifests[split])[0] == recount(saved, manifests[split])[0]
                    if split == 'eval':
                        np.testing.assert_array_equal(saved, np.load(root / f'validation_raw/{spec}_e{epoch}_seed{seed}.npy'))
                    counts['raw_forwards'] += 1
                del model
                print(json.dumps({'stage': 'checkpoint_replayed', 'spec': spec, 'seed': seed, 'epoch': epoch}), flush=True)
    def refs(role, split):
        shape = np.load(args.reference / f'features/shape_{split}.npz')['features'].astype(float)
        return {'S0': z(10 * shape @ text.T),
                'r4_M': np.load(args.r4 / f'{role}/locked_M_v0.npy').astype(float),
                'r8_M': np.load(args.r8 / f'{role}/locked_M.npy').astype(float)}
    base = refs('validation', 'eval')
    grid = read(root / 'validation_grid.json')
    assert len(grid) == 111
    for row in grid:
        arrays = [base[row['name']]] * 3 if row['name'] in base else [
            (1 - row['weight']) * base[row['base']] + row['weight'] * z(np.load(root /
                f'validation_raw/{row["name"]}_e{row["epoch"]}_seed{seed}.npy').astype(float)) for seed in (42, 43, 44)]
        measures = [recount(v, manifests['eval'])[0] for v in arrays]
        assert [m['correct'] for m in measures] == row['correct_by_seed']
        assert np.mean([m['correct'] for m in measures]) == row['mean_correct']
        assert abs(np.mean([m['class_mean_accuracy'] for m in measures]) - row['mean_macro']) < 1e-12
    reference = next(r for r in grid if r['name'] == 'r8_M')
    selection = read(root / 'selection.json')
    for mode in ('P', 'B', 'M'):
        assert pick(grid, reference, mode) == selection['locked'][mode]
    for spec in ('geometry_kd', 'color_ce', 'color_kd'):
        assert pick([r for r in grid if r['name'] == spec], reference, 'P') == selection['per_candidate'][spec]
    records = read(root / 'results.json')['records']
    comparisons = {}
    for role, split in (('validation', 'eval'), ('historical_test', 'dev')):
        base = refs(role, split)
        hits = {}
        for row in [r for r in records if r['role'] == role]:
            sel, seed = row['selection'], row['seed']
            actual = np.load(root / role / f'{row["method"]}_seed{seed}.npy')
            expected_scores = base[sel['name']] if sel['name'] in base else (1 - sel['weight']) * base[sel['base']] + sel['weight'] * z(
                np.load(root / role / f'{sel["name"]}_e{sel["epoch"]}_seed{seed}_raw.npy').astype(float))
            np.testing.assert_array_equal(actual, expected_scores)
            summary, hit, _, _ = recount(actual, manifests[split])
            assert summary == row['summary']
            hits.setdefault(row['method'], []).append(hit)
        for mode in ('P', 'B', 'M'):
            for name in ('S0', 'r4_M', 'r8_M'):
                comparisons[f'{role}:locked_{mode}-{name}'] = paired_bootstrap(
                    np.array(hits['locked_' + mode]), np.array(hits[name]), manifests[split])
    result = {'status': 'passed', 'counts': counts, 'maximum_forward_abs': maximum,
              'validation_grid_rows': len(grid), 'prediction_arrays_recounted': len(records),
              'comparisons': comparisons, 'inherited_reference_limit': 'r8 rotation precision limit',
              'scope': 'All frozen original state, every checkpoint/raw forward, independent NumPy text projection and exact Top-5, full grid/selection/fusion/metrics/bootstrap; not a full optimization trajectory replay.'}
    write_json(root / 'AUDIT.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for n in ('output', 'reference', 'r4', 'r8', 'color', 'checkpoint'):
        parser.add_argument('--' + n, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    audit(args)
