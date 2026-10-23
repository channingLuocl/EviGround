"""Locally verify exported predictions and write the scale experiment report."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from audit_shape_scale import recount
from category_common_v82 import file_hash, normalized_category
from shape_scale import bootstrap
from run_storage import write_json


NAMES = {'E0': '原始PointNeXt+CLIP', 'E1': '校准全局残差', 'E2': '米制尺寸残差',
         'E2-R': '仅比例残差', 'E3': '全局残差＋米制尺寸'}


def summarize(output, manifest_path):
    for marker in ('SELECTION_FROZEN.json', 'RESULTS_FROZEN.json'):
        for name, digest in json.loads((output / marker).read_text())['files'].items():
            assert file_hash(output / name) == digest, name
    assert json.loads((output / 'audit.json').read_text())['status'] == 'passed'
    manifest = json.loads(manifest_path.read_text())
    config = json.loads((output / 'config.json').read_text())
    assert file_hash(manifest_path) == config['manifest_sha256']['eval']
    results = json.loads((output / 'results.json').read_text())
    selection = json.loads((output / 'selection.json').read_text())
    hits, per_class = defaultdict(list), []
    for record in results['records']:
        seed, method = record['seed'], record['method']
        logits = np.load(output / f'eval/seed{seed}/{method}/logits.npy')
        summary, hit = recount(logits, manifest)
        for key, value in summary.items():
            assert abs(value - record['summary'][key]) < 1e-12
        assert summary['count'] == 3413
        hits[method].append(hit)
        groups = defaultdict(list)
        for row, correct in zip(manifest['samples'], hit):
            groups[normalized_category(row['category'])].append(bool(correct))
        per_class.extend({'method': method, 'seed': seed, 'category': name, 'support': len(values),
                          'correct': sum(values), 'accuracy': float(np.mean(values))}
                         for name, values in sorted(groups.items()))
    analysis = output / 'analysis'
    analysis.mkdir(exist_ok=True)
    write_json(analysis / 'per_class.json', per_class)
    baseline = np.mean(hits['E0'], 0)
    scenes = [r['scene'] for r in manifest['samples']]
    table = ['模型\tTrue准确率（均值±标准差）\tExact Top-1\tTop-5\tClass Mean\t相对基线提升（百分点）\t95%区间（百分点）']
    report = ['# 分类修正与真实尺度：第一阶段结果', '',
              '六个小头均完成40 epoch，内部开发集选checkpoint及共享融合权重后统一评估；远端独立审计及本地重算通过。', '',
              '固定3413对象、197类；157个词表外目标保留分母。三种子42/43/44，均值±样本标准差。该集合是此前多次查看的验证集。', '',
              '| 方法 | True | Exact | Top-5 | Class Mean | 相对基线（百分点） |',
              '|---|---:|---:|---:|---:|---:|']
    for method in NAMES:
        aggregate = results['aggregate'][method]
        for metric in ('accuracy', 'exact_accuracy', 'top5_accuracy', 'class_mean_accuracy'):
            values = [r['summary'][metric] for r in results['records'] if r['method'] == method]
            assert abs(np.mean(values) - aggregate[metric]['mean']) < 1e-12
            assert abs(np.std(values, ddof=1) - aggregate[metric]['std']) < 1e-12
        delta = np.mean(hits[method], 0) - baseline
        interval = bootstrap(delta, scenes)
        np.testing.assert_allclose(interval, aggregate['delta_accuracy_95ci_room_bootstrap'], atol=1e-12)
        cells = [f"{aggregate[k]['mean']:.2%} ± {aggregate[k]['std']*100:.2f}%" for k in
                 ('accuracy', 'exact_accuracy', 'top5_accuracy', 'class_mean_accuracy')]
        table.append('\t'.join([NAMES[method], *cells, f'{delta.mean()*100:+.2f}', f'[{interval[0]*100:+.2f}, {interval[1]*100:+.2f}]']))
        report.append('| ' + ' | '.join([f'{method} {NAMES[method]}', *cells, f'{delta.mean()*100:+.2f}']) + ' |')
    pair = np.mean(hits['E2'], 0) - np.mean(hits['E2-R'], 0)
    contrast = {'comparison': 'E2_minus_E2-R', 'accuracy_delta': float(pair.mean()),
                '95ci_room_bootstrap': bootstrap(pair, scenes),
                'interpretation': 'development-selected fixed fusion; previously inspected validation; no multiplicity adjustment'}
    write_json(analysis / 'size_vs_ratio.json', contrast)
    report.extend(['', '## 开发集选择与纠错', '', '| 方法 | 全局权重a | 几何权重c | 三种子纠正数 | 三种子改错数 |', '|---|---:|---:|---|---|'])
    for method in NAMES:
        rows = [r for r in results['records'] if r['method'] == method]
        a, c = selection[method]['weights']
        report.append(f"| {method} | {a} | {c} | " + '/'.join(str(r['flips']['corrected']) for r in rows)
                      + ' | ' + '/'.join(str(r['flips']['broken']) for r in rows) + ' |')
    report.extend(['', '区间按物理房间配对bootstrap重采样10000次，先对每对象三种子正确率取均值；训练种子波动另报。区间未作多重比较校正，不能作为挑选最高结果后的确认性证据。', '',
                   f"米制尺寸相对仅比例的平均差值：{pair.mean()*100:+.2f}个百分点；95%区间 [{contrast['95ci_room_bootstrap'][0]*100:+.2f}, {contrast['95ci_room_bootstrap'][1]*100:+.2f}]。", '',
                   '若E2和E2-R的开发集选择均为零权重，两者最终输出都退回基线，上述零差值不能证明绝对尺度没有价值；它只表示本轮小残差头及选模协议没有获得有效的几何修正。', '',
                   f"开发集原始基线True为{selection['E0']['mean_accuracy']:.2%}，当前验证集为{results['aggregate']['E0']['accuracy']['mean']:.2%}，差异较大。原因尚未查明，后续需要核查输入/标签分布与骨干预训练数据覆盖，不能把它直接当作已证实的数据泄漏。", '',
                   '第一阶段只验证分类修正与尺度，尚未实现RGB多视图分支或新的学习式门控，尚未运行新的Nr3D完整定位。', '',
                   '逐种子整数正确数见table.tsv；均值表见analysis/table_mean_std.tsv；每类支持数见analysis/per_class.json。原始结果与旧缓存未被覆盖。'])
    (analysis / 'table_mean_std.tsv').write_text('\n'.join(table) + '\n')
    (analysis / 'report.md').write_text('\n'.join(report) + '\n')
    write_json(analysis / 'local_audit.json', {'status': 'passed', 'evaluations': len(results['records']),
        'source_sha256': file_hash(Path(__file__)),
        'recount_source_sha256': file_hash(Path(__file__).with_name('audit_shape_scale.py')),
        'checks': ['all frozen file hashes', 'independent metric recount',
                'three-seed aggregates', 'room bootstrap intervals'], 'contrast': contrast})
    print((analysis / 'report.md').read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.output, args.manifest)
