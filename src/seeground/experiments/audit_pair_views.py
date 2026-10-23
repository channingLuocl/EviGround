"""Recount room-CV outcomes and independently reconstruct all fusion outputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from seeground.categories.category_common_v82 import file_hash
from seeground.evaluation.run_storage import write_json


METHODS = ('S0', 'A', 'B', 'C', 'D', 'E')


def z(x):
    return (x-x.mean(-1, keepdims=True))/np.maximum(x.std(-1, keepdims=True), 1e-6)


def softmax(x):
    e = np.exp(x-x.max(-1, keepdims=True))
    return e/e.sum(-1, keepdims=True)


def ent(x):
    p = softmax(x)
    return -(p*np.log(p)).sum(-1)/np.log(x.shape[-1])


def gap(x):
    top = np.sort(x, axis=-1)[..., -2:]
    return top[..., 1]-top[..., 0]


def features(base, rgb, view, mode):
    base, rgb, view = z(base), z(rgb), z(view)
    bp, rp = base.argmax(-1), rgb.argmax(-1)
    d = bp != rp
    generic = np.stack([base.max(-1), rgb.max(-1), gap(base), gap(rgb), ent(base), ent(rgb),
                        np.abs(base-rgb).mean(-1), d.astype(float)], -1)
    if mode == 'C':
        pb, pr = softmax(base), softmax(rgb)
        mid = (pb+pr)/2
        js = .5*((pb*np.log(pb/mid)).sum(-1)+(pr*np.log(pr/mid)).sum(-1))
        tb, tr = np.sort(base, axis=-1)[:, -3:], np.sort(rgb, axis=-1)[:, -3:]
        cosine = (base*rgb).sum(-1)/(np.linalg.norm(base, axis=-1)*np.linalg.norm(rgb, axis=-1))
        extra = np.stack([pb.max(-1), pr.max(-1), gap(pb), gap(pr), tb[:, 2]-tb[:, 0], tr[:, 2]-tr[:, 0], js, cosine], -1)
    elif mode == 'D':
        vm, ve, vx = gap(view), ent(view), view.max(-1)
        supported = view[np.arange(len(view))[:, None], np.arange(4)[None, :], rp[:, None]]
        extra = np.stack([vm.mean(1), vm.std(1), ve.mean(1), ve.std(1), vx.mean(1), vx.std(1),
                          (view.argmax(-1) == rp[:, None]).mean(1), supported.std(1)], -1)
    else:
        rows, angles = np.arange(len(view))[:, None], np.arange(4)[None, :]
        delta = view[rows, angles, rp[:, None]]-view[rows, angles, bp[:, None]]
        rows = np.arange(len(base))
        extra = np.stack([delta.mean(1), delta.std(1), delta.min(1), delta.max(1), (delta > 0).mean(1),
                          base[rows, rp]-base[rows, bp], rgb[rows, rp]-rgb[rows, bp], np.abs(delta).mean(1)], -1)
    return np.concatenate([generic, extra], -1), d


def measure(scores, y):
    prediction = scores.argmax(-1)
    hit = prediction == y
    counts = np.bincount(y, minlength=scores.shape[-1])
    correct = np.bincount(y[hit], minlength=scores.shape[-1])
    return {'count': len(y), 'correct': int(hit.sum()), 'accuracy': float(hit.mean()),
            'class_mean': float((correct[counts > 0]/counts[counts > 0]).mean()),
            'classes_present': int((counts > 0).sum())}, hit


def main(args):
    torch.set_num_threads(2)
    out, reference = args.output, args.reference
    config = json.loads((out/'config.json').read_text())
    for marker in ('EXPERTS_COMPLETE.json', 'RESULTS_COMPLETE.json'):
        seal = json.loads((out/marker).read_text())
        assert seal['config_sha256'] == file_hash(out/'config.json')
        for name, digest in seal['files'].items():
            assert file_hash(out/name) == digest, name
    for name, digest in config['reference_files'].items():
        assert file_hash(reference/name) == digest
    for name, digest in config['sources'].items():
        assert file_hash(out/'source_snapshot'/name) == digest
    assert file_hash(out/'source_snapshot/design.md') == config['design_sha256']
    manifest = json.loads((reference/'manifests/train.json').read_text())
    y = np.array([s['label'] for s in manifest['samples']])
    rooms = np.array([s['scene'].split('_')[0] for s in manifest['samples']])
    folds = json.loads((out/'folds.json').read_text())
    outer = np.array(folds['outer'])
    independent = lambda salt: np.array([int.from_bytes(hashlib.sha256(f'{salt}:{room}'.encode()).digest()[:4], 'little') % 3 for room in rooms])
    np.testing.assert_array_equal(outer, independent('pair-outer-20261019'))
    rgb = np.load(reference/'features/rgb_train/features.npz')['features'].astype(np.float64)
    shape = np.load(reference/'features/shape_train.npz')['features'].astype(np.float64)
    text = np.load(reference/'features/text.npy').astype(np.float64)
    view_seal = json.loads((args.views/'COMPLETE.json').read_text())
    assert file_hash(args.views/'features.npz') == view_seal['features_sha256']
    views = np.load(args.views/'features.npz')['features'].astype(np.float64)
    merged = views.mean(1)
    merged /= np.linalg.norm(merged, axis=1, keepdims=True)
    np.testing.assert_allclose(merged, rgb, atol=1e-5, rtol=0)
    base = z(10*shape @ text.T)
    rows = json.loads((out/'results.json').read_text())['records']
    by_record = {(r['outer'], r['seed'], r['method']): r for r in rows}
    all_hits = {m: np.zeros((3, len(y)), dtype=bool) for m in METHODS}
    all_top5 = {m: np.zeros((3, len(y)), dtype=bool) for m in METHODS}
    max_gate_difference = 0.
    max_macro_rounding_difference = 0.
    for fold in range(3):
        train, test = np.flatnonzero(outer != fold), np.flatnonzero(outer == fold)
        assert not set(rooms[train]) & set(rooms[test])
        inner = np.array(folds['inner'][str(fold)])
        np.testing.assert_array_equal(inner, independent(f'pair-inner-{fold}-20261019'))
        counts = np.bincount(y[train], minlength=197)+1
        p = counts/counts.sum()
        selected = json.loads((out/f'outer{fold}/selection.json').read_text())
        for si, seed in enumerate((42, 43, 44)):
            raw_oof = {}
            raw_test = {}
            for kind in ('ce', 'balanced'):
                folder = out/f'outer{fold}/seed{seed}/{kind}'
                a = np.load(folder/'scores.npz')
                np.testing.assert_array_equal(a['train_indices'], train)
                np.testing.assert_array_equal(a['test_indices'], test)
                expected = np.zeros((len(y), 197))
                for f in range(3):
                    fit, cal = train[inner[train] != f], train[inner[train] == f]
                    assert not set(rooms[fit]) & set(rooms[cal])
                    head = torch.load(folder/f'inner{f}.pt', weights_only=True, map_location='cpu')
                    assert head['train_objects'] == len(fit) and head['epochs'] == 40
                    expected[cal] = rgb[cal] @ head['weight'].numpy().T+head['bias'].numpy()
                np.testing.assert_allclose(expected[train], a['oof'], rtol=1e-5, atol=2e-5)
                head = torch.load(folder/'outer.pt', weights_only=True, map_location='cpu')
                assert head['train_objects'] == len(train) and head['epochs'] == 40
                test_logits = rgb[test] @ head['weight'].numpy().T+head['bias'].numpy()
                np.testing.assert_allclose(test_logits, a['test'], rtol=1e-5, atol=2e-5)
                for history in folder.glob('*.history.json'):
                    history_rows = json.loads(history.read_text())
                    assert len(history_rows) == 40 and np.isfinite(history_rows).all()
                raw_oof[kind], raw_test[kind] = a['oof'].astype(float), a['test'].astype(float)
            dest = out/f'outer{fold}/seed{seed}/comparison'
            expected = {'S0': base[test]}
            for method in ('A', 'B'):
                sel = selected[method]
                expected[method] = (1-sel['weight'])*base[test]+sel['weight']*z(raw_test[sel['kind']]-sel['tau']*np.log(p))
            sel = selected['B']
            folder = out/f'outer{fold}/seed{seed}/{sel["kind"]}'
            head = torch.load(folder/'outer.pt', weights_only=True, map_location='cpu')
            view = views[test] @ head['weight'].numpy().T+head['bias'].numpy()-sel['tau']*np.log(p)
            r = raw_test[sel['kind']]-sel['tau']*np.log(p)
            for method in ('C', 'D', 'E'):
                x, d = features(base[test], r, view, method)
                checkpoint = torch.load(dest/f'{method}.pt', weights_only=True, map_location='cpu')
                state = {k: v.numpy().astype(float) for k, v in checkpoint['state_dict'].items()}
                h = np.clip((x-state['mean'])/state['std'], -5, 5) @ state['network.0.weight'].T+state['network.0.bias']
                h = F.gelu(torch.from_numpy(h)).numpy()
                delta = .5*np.tanh((h @ state['network.2.weight'].T+state['network.2.bias']).ravel())
                weights = np.clip(sel['weight']+delta*d, 0, 1)
                saved = np.load(dest/f'{method}_weights.npy')
                max_gate_difference = max(max_gate_difference, float(np.max(np.abs(weights-saved))))
                np.testing.assert_allclose(weights, saved, atol=2e-4, rtol=1e-4)
                expected[method] = (1-weights[:, None])*base[test]+weights[:, None]*z(r)
            for method in METHODS:
                actual = np.load(dest/f'{method}.npy')
                np.testing.assert_allclose(expected[method], actual, rtol=2e-4, atol=1e-3)
                row, hit = measure(actual, y[test])
                stored = by_record[(fold, seed, method)]
                for key, value in row.items():
                    # GPU float32 reductions can differ from NumPy float64 by >1e-7.
                    if isinstance(value, int):
                        assert value == stored[key], (method, key, value, stored[key])
                    else:
                        assert abs(value-stored[key]) < 2e-7, (method, key, value, stored[key])
                all_hits[method][si, test] = hit
                all_top5[method][si, test] = (np.argpartition(actual, -5, axis=1)[:, -5:] == y[test, None]).any(1)
        for method in ('A', 'B'):
            grid = selected[f'{method}_grid']
            rebuilt = []
            for row in grid:
                values = []
                for seed in (42, 43, 44):
                    a = np.load(out/f'outer{fold}/seed{seed}/{row["kind"]}/scores.npz')['oof']
                    values.append(measure((1-row['weight'])*base[train]+row['weight']*z(a-row['tau']*np.log(p)), y[train])[0])
                assert row['total_correct'] == sum(v['correct'] for v in values)
                macro = float(np.mean([v['class_mean'] for v in values]))
                rounding_difference = abs(row['mean_macro']-macro)
                max_macro_rounding_difference = max(max_macro_rounding_difference, rounding_difference)
                assert rounding_difference < 2e-7
                rebuilt.append({**row, 'mean_macro': macro})
            baseline_macro = measure(base[train], y[train])[0]['class_mean']
            eligible = [r for r in rebuilt if r['mean_macro'] >= baseline_macro-1e-7]
            best = min(eligible, key=lambda r: (-r['total_correct'], -r['mean_macro'], r['kind'] != 'ce', r['tau'], r['weight']))
            assert all(best[k] == selected[method][k] for k in ('kind', 'tau', 'weight', 'total_correct'))
    unique_rooms, room_index = np.unique(rooms, return_inverse=True)
    room_counts = np.bincount(room_index)
    draws = np.random.default_rng(20261019).integers(len(unique_rooms), size=(10000, len(unique_rooms)))
    comparisons = {}
    for control in ('B', 'C', 'D'):
        difference = all_hits['E'].mean(0)-all_hits[control].mean(0)
        sums = np.bincount(room_index, weights=difference)
        delta = 100*sums[draws].sum(1)/room_counts[draws].sum(1)
        comparisons[control] = {'delta_pp': float(100*difference.mean()),
            'room_95CI_pp': np.quantile(delta, [.025, .975]).tolist(),
            'corrected_by_seed': ((~all_hits[control]) & all_hits['E']).sum(1).tolist(),
            'broken_by_seed': (all_hits[control] & ~all_hits['E']).sum(1).tolist()}
    groups = [y == c for c in np.unique(y)]
    summary = {}
    for method in METHODS:
        hit = all_hits[method]
        summary[method] = {'correct_by_seed': hit.sum(1).tolist(),
            'accuracy_by_seed': hit.mean(1).tolist(), 'top5_by_seed': all_top5[method].mean(1).tolist(),
            'macro_by_seed': [float(np.mean([h[g].mean() for g in groups])) for h in hit]}
    np.savez_compressed(out/'review_outcomes.npz', **{f'{m}_hit': a for m, a in all_hits.items()},
                        **{f'{m}_top5': a for m, a in all_top5.items()}, labels=y, room_index=room_index, outer=outer)
    write_json(out/'audited_summary.json', {'status': 'passed', 'objects': len(y), 'rooms': len(unique_rooms),
        'scans': len({r['scene'] for r in manifest['samples']}),
        'classes_present': len(groups), 'methods': summary, 'E_comparisons': comparisons,
        'maximum_independent_gate_weight_difference': max_gate_difference,
        'maximum_macro_float32_rounding_difference': max_macro_rounding_difference,
        'auditor_sha256': file_hash(Path(__file__)), 'checks': ['source/model/result hashes',
            'room-isolated outer and inner folds', '72 heads, final epochs and OOF outputs',
            'full OOF fixed-fusion grids', 'view-mean bridge', 'independent numpy gate features and forward pass',
            'all outer scores, metrics and paired bootstrap']})
    print(json.dumps(json.loads((out/'audited_summary.json').read_text())), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'output', 'views'):
        parser.add_argument('--'+name, type=Path, required=True)
    main(parser.parse_args())
