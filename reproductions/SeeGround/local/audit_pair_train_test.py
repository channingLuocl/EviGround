"""Independently reconstruct frozen validation/test predictions and recount metrics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from audit_pair_views import features, z
from category_common_v82 import file_hash, normalized_category
from run_rgb_experiment import check_frozen
from run_storage import write_json

SEEDS = (42, 43, 44)
METHODS = ('S0', 'A', 'B', 'C', 'D', 'E')


def read(path):
    return json.loads(path.read_text())


def recount(values, manifest):
    assert values.shape == (len(manifest['samples']), len(manifest['vocabulary']))
    assert np.isfinite(values).all()
    order = np.argsort(-values, axis=1, kind='stable')
    names = np.array([normalized_category(n) for n in manifest['vocabulary']])
    target = np.array([normalized_category(s['category']) for s in manifest['samples']])
    hit = names[order[:, 0]] == target
    top5 = (names[order[:, :5]] == target[:, None]).any(1)
    exact = np.array([manifest['vocabulary'][i] == s['category']
                      for i, s in zip(order[:, 0], manifest['samples'])])
    macro = np.mean([hit[target == name].mean() for name in np.unique(target)])
    return {'count': len(hit), 'correct': int(hit.sum()), 'accuracy': float(hit.mean()),
            'exact_accuracy': float(exact.mean()), 'top5_accuracy': float(top5.mean()),
            'class_mean_accuracy': float(macro), 'errors': 0}, hit, top5, order


def linear(path, x):
    record = torch.load(path, weights_only=True, map_location='cpu')
    return x @ record['weight'].numpy().astype(float).T+record['bias'].numpy().astype(float)


def gate(path, x, different):
    record = torch.load(path, weights_only=True, map_location='cpu')
    s = {k: v.numpy().astype(float) for k, v in record['state_dict'].items()}
    h = np.clip((x-s['mean'])/s['std'], -5, 5) @ s['network.0.weight'].T+s['network.0.bias']
    h = F.gelu(torch.from_numpy(h)).numpy()
    delta = .5*np.tanh((h @ s['network.2.weight'].T+s['network.2.bias']).ravel())
    return np.clip(record['weight']+delta*different, 0, 1)


def bootstrap(hits, manifest, selected):
    rooms, ri = np.unique([s['scene'].split('_')[0] for s in manifest['samples']], return_inverse=True)
    draws = np.random.default_rng(20261019).integers(len(rooms), size=(10000, len(rooms)))
    counts = np.bincount(ri)
    result = {}
    for left, right in [('E', m) for m in ('B', 'C', 'D')]+[(selected, 'S0')]:
        delta = hits[left].mean(0)-hits[right].mean(0)
        sums = np.bincount(ri, weights=delta)
        distribution = 100*sums[draws].sum(1)/counts[draws].sum(1)
        result[f'{left}-{right}'] = {'delta_pp': float(100*delta.mean()),
            'room_95CI_pp': np.quantile(distribution, [.025, .975]).tolist(),
            'corrected_by_seed': ((~hits[right]) & hits[left]).sum(1).tolist(),
            'broken_by_seed': (hits[right] & ~hits[left]).sum(1).tolist()}
    return result


def audit(args):
    out, ref = args.output, args.reference
    for marker in ('SELECTION_FROZEN.json', 'TEST_FROZEN.json'):
        check_frozen(out, marker)
    config = read(out/'config.json')
    for name, digest in config['inputs'].items():
        assert file_hash(name) == digest, name
    for name, digest in config['sources'].items():
        assert file_hash(out/'source_snapshot'/name) == digest, name
    manifests = {s: read(ref/f'manifests/{s}.json') for s in ('train', 'eval', 'dev')}
    roomsets = {s: {r['scene'].split('_')[0] for r in m['samples']} for s, m in manifests.items()}
    for a, b in (('train', 'eval'), ('train', 'dev'), ('eval', 'dev')):
        assert not roomsets[a] & roomsets[b]
    y = np.array([s['label'] for s in manifests['train']['samples']])
    assert len(y) == 28730 and (y >= 0).all()
    r1 = next(Path(p).parent for p in config['inputs'] if p.endswith('/experiment/config.json'))
    for marker in ('EXPERTS_COMPLETE.json', 'RESULTS_COMPLETE.json'):
        for name, digest in read(r1/marker)['files'].items():
            assert file_hash(r1/name) == digest, name
    rooms = [s['scene'].split('_')[0] for s in manifests['train']['samples']]
    folds = np.array(read(r1/'folds.json')['outer'])
    expected_folds = [int.from_bytes(hashlib.sha256(f'pair-outer-20261019:{room}'.encode()).digest()[:4], 'little')%3 for room in rooms]
    np.testing.assert_array_equal(folds, expected_folds)
    for fold in range(3):
        fit = folds != fold
        assert not set(np.array(rooms)[fit]) & set(np.array(rooms)[~fit])
        for seed in SEEDS:
            for kind in ('ce', 'balanced'):
                ckpt = torch.load(r1/f'outer{fold}/seed{seed}/{kind}/outer.pt', weights_only=True, map_location='cpu')
                assert ckpt['train_objects'] == int(fit.sum()) and ckpt['epochs'] == 40
    counts = np.bincount(y, minlength=197)+1
    logp = np.log(counts/counts.sum())
    for seed in SEEDS:
        for name in ('ce', 'balanced', 'C', 'D', 'E'):
            path = out/f'models/seed{seed}/{name}.pt'
            ckpt = torch.load(path, weights_only=True, map_location='cpu')
            assert ckpt['seed'] == seed and ckpt['epochs'] == 40
            if name in ('ce', 'balanced'):
                assert ckpt['train_objects'] == 28730 and ckpt['loss'] == name
            history = read(path.with_suffix('.history.json'))
            assert len(history) == 40 and np.isfinite(history).all()
    selection = read(out/'fusion_selection.json')
    deployment = read(out/'deployment_selection.json')['selected']
    text = np.load(ref/'features/text.npy').astype(float)
    summary = {}
    max_weight_difference = 0.
    for role, split, view_folder in (('validation', 'eval', 'validation_views'),
                                     ('historical_test', 'dev', 'historical_test_views')):
        manifest = manifests[split]
        shape = np.load(ref/f'features/shape_{split}.npz')['features'].astype(float)
        rgb_path = ref/('features/rgb_dev.npz' if split == 'dev' else 'features/rgb_eval/features.npz')
        rgb = np.load(rgb_path)['features'].astype(float)
        views = np.load(out/view_folder/'features.npz')['features'].astype(float)
        seal = read(out/view_folder/'COMPLETE.json')
        assert file_hash(out/view_folder/'features.npz') == seal['features_sha256']
        merged = views.mean(1)
        merged /= np.linalg.norm(merged, axis=1, keepdims=True)
        np.testing.assert_allclose(merged, rgb, atol=1e-5, rtol=0)
        base = z(10*shape @ text.T)
        stored = {(r['method'], r['seed']): r for r in read(out/role/'results.json')['records']}
        hits = {m: [] for m in METHODS}
        rows = []
        raw_heads = {(seed, kind): linear(out/f'models/seed{seed}/{kind}.pt', rgb)
                     for seed in SEEDS for kind in ('ce', 'balanced')}
        if role == 'validation':
            # Full independent float64 grid, including the common three-seed selection.
            baseline_macro = recount(base, manifest)[0]['class_mean_accuracy']
            for method in ('A', 'B'):
                grid = []
                for row in selection[method+'_grid']:
                    metrics = [recount((1-row['weight'])*base+row['weight']*z(
                        raw_heads[(seed, row['kind'])]-row['tau']*logp), manifest)[0] for seed in SEEDS]
                    rebuilt = {**row, 'total_correct': sum(v['correct'] for v in metrics),
                               'mean_macro': float(np.mean([v['class_mean_accuracy'] for v in metrics]))}
                    assert row['total_correct'] == rebuilt['total_correct']
                    assert abs(row['mean_macro']-rebuilt['mean_macro']) < 2e-7
                    grid.append(rebuilt)
                eligible = [r for r in grid if r['mean_macro'] >= baseline_macro-1e-7]
                best = min(eligible, key=lambda r: (-r['total_correct'], -r['mean_macro'],
                                                   r['kind'] != 'ce', r['tau'], r['weight']))
                assert all(best[k] == selection[method][k] for k in ('kind', 'tau', 'weight', 'total_correct'))
        with (out/role/'objects.jsonl').open('w') as objects:
            for seed in SEEDS:
                expected = {'S0': base}
                for method in ('A', 'B'):
                    s = selection[method]
                    expected[method] = (1-s['weight'])*base+s['weight']*z(
                        raw_heads[(seed, s['kind'])]-s['tau']*logp)
                s = selection['B']
                raw = raw_heads[(seed, s['kind'])]-s['tau']*logp
                view = linear(out/f'models/seed{seed}/{s["kind"]}.pt', views)-s['tau']*logp
                for method in ('C', 'D', 'E'):
                    x, different = features(base, raw, view, method)
                    weight = gate(out/f'models/seed{seed}/{method}.pt', x, different)
                    saved = np.load(out/role/f'{method}_seed{seed}_weights.npy')
                    max_weight_difference = max(max_weight_difference, float(np.abs(weight-saved).max()))
                    np.testing.assert_allclose(weight, saved, atol=2e-4, rtol=1e-4)
                    expected[method] = (1-weight[:, None])*base+weight[:, None]*z(raw)
                for method in METHODS:
                    actual = np.load(out/role/f'{method}_seed{seed}.npy')
                    np.testing.assert_allclose(expected[method], actual, atol=1e-3, rtol=2e-4)
                    metrics, hit, top5, order = recount(actual, manifest)
                    for key, value in metrics.items():
                        assert abs(value-stored[(method, seed)]['summary'][key]) < 2e-7, (role, method, key)
                    hits[method].append(hit)
                    rows.append({'method': method, 'seed': seed, **metrics})
                    for i, sample in enumerate(manifest['samples']):
                        objects.write(json.dumps({'method': method, 'seed': seed, 'scene': sample['scene'],
                            'bbox_id': sample['bbox_id'], 'target': sample['category'],
                            'predicted': manifest['vocabulary'][int(order[i, 0])], 'correct': bool(hit[i]),
                            'top5': [manifest['vocabulary'][int(j)] for j in order[i, :5]]})+'\n')
        hits = {m: np.array(a) for m, a in hits.items()}
        summary[role] = {'records': rows, 'comparisons': bootstrap(hits, manifest, deployment)}
        np.savez_compressed(out/role/'audited_hits.npz', **hits)
        if role == 'validation':
            by = {m: [r for r in rows if r['method'] == m] for m in METHODS}
            better = sum(r['correct'] for r in by['E']) > sum(r['correct'] for r in by['B'])
            macro_ok = np.mean([r['class_mean_accuracy'] for r in by['E']]) >= np.mean([r['class_mean_accuracy'] for r in by['B']])
            assert deployment == ('E' if better and macro_ok else 'B')
        else:
            assert all(r['correct'] == 2262 for r in rows if r['method'] == 'S0')
    evaluation_hashes = {str(p): file_hash(p) for p in (
        ref/'features/shape_dev.npz', ref/'features/rgb_dev.npz', ref/'manifests/dev.json')}
    summary.update(status='passed', deployment=deployment, maximum_gate_weight_difference=max_weight_difference,
                   historical_test_inputs_sha256=evaluation_hashes,
                   auditor_sha256=file_hash(Path(__file__)), limitations=config['limitation'])
    write_json(out/'audited_summary.json', summary)
    report = ['# 分类训练、验证与历史测试 r2', '',
              '参数训练28730对象；验证404对象（含13个词表外对象）；历史测试3413对象。',
              '404和3413此前均已揭示；不能视为独立盲测，冻结3D主干预训练暴露范围未知。',
              f'验证选择A：{selection["A"]}；B：{selection["B"]}。锁定部署方案：{deployment}。', '',
              'True为既有同义词/复数归一化准确率；Exact Top-1为原标签字符串精确匹配。',
              'Class Mean按归一化目标类别平均，保留词表外对象计错；全部模型保留三个种子。', '']
    for role in ('validation', 'historical_test'):
        report += ['## '+role, '', '| 模型/种子 | 评估对象数 | True正确数 | True准确率 | Exact Top-1 | Top-5 | Class Mean | Errors |',
                   '|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in summary[role]['records']:
            report.append(f'| {r["method"]}/{r["seed"]} | {r["count"]} | {r["correct"]} | '+
                ' | '.join(f'{100*r[k]:.2f}%' for k in ('accuracy', 'exact_accuracy', 'top5_accuracy', 'class_mean_accuracy'))+' | 0 |')
        report += ['', '房间配对bootstrap10000次（种子20261019），差值单位为百分点：', '',
                   '| 比较 | 平均差值 | 95% CI | 纠正/种子 | 改错/种子 |', '|---|---:|---|---|---|']
        for name, r in summary[role]['comparisons'].items():
            report.append(f'| {name} | {r["delta_pp"]:.3f} | {r["room_95CI_pp"]} | {r["corrected_by_seed"]} | {r["broken_by_seed"]} |')
        report += ['']
    report += ['## 核查', '', '源码/输入/模型/选型哈希、房间隔离、15模型40轮记录、四视图均值桥接、完整验证网格、独立线性头与门控前向、全部预测和指标重算均通过。',
               '训练OOF专家与r1封存模型哈希一致，物理房间分折独立重算通过。配对区间仅反映固定模型的房间抽样，不消除历史暴露或主干预训练范围未知的限制。',
               '训练侧r1的97.83%不作为本轮测试性能；本轮只训练RGB分类头和融合门控，未微调点云/CLIP主干。']
    (out/'report.md').write_text('\n'.join(report)+'\n')
    print(json.dumps({'status': 'passed', 'deployment': deployment,
                      'historical_test': summary['historical_test']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    torch.set_num_threads(2)
    audit(parser.parse_args())
