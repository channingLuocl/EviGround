"""Export audited development/holdout metrics and paired class/room diagnostics."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from eviground.experiments.audit_shape_scale import recount, normalized_category
from eviground.categories.category_common_v82 import file_hash
from eviground.evaluation.run_storage import write_json


NAMES = {'S0': 'PointNeXt+CLIP 原基线', 'S1': '监督3D线性头',
         'S2': '监督RGB线性头', 'S3': '原3D＋监督RGB（锁定主方法）',
         'S4': '监督3D＋监督RGB'}
METRICS = ('accuracy', 'exact_accuracy', 'top5_accuracy', 'class_mean_accuracy')


def summarize(output):
    audit = json.loads((output / 'audit.json').read_text())
    assert audit['status'] == 'passed' and audit['stage'] == 'full'
    selection = json.loads((output / 'selection.json').read_text())
    analysis = output / 'analysis'
    analysis.mkdir(exist_ok=True)
    summaries = {}
    diagnostics = []
    for split in ('dev', 'eval'):
        manifest = json.loads((output / f'manifests/{split}.json').read_text())
        result = json.loads((output / split / 'results.json').read_text())
        table = ['模型\t评估对象数\tTrue正确数\tTrue准确率\tExact Top-1\tTop-5\tClass Mean\tErrors']
        rows, hits = {}, {}
        for record in result['records']:
            name = record['method']
            logits = np.load(output / split / f'{name}.npy')
            metrics, hit = recount(logits, manifest)
            for key, value in metrics.items():
                assert abs(value-record['summary'][key]) < 1e-12
            hits[name] = hit
            rows[name] = metrics
            label = NAMES[name.split('_')[0]] + (f"（{name.split('_')[1]}）" if '_' in name else '')
            table.append('\t'.join([label, str(metrics['count']), str(metrics['correct']),
                                   *[f'{metrics[k]:.2%}' for k in METRICS], '0']))
            if name.split('_')[0] == selection['primary_method']:
                order = np.argsort(-logits, axis=1, kind='stable')
                predictions = [{'scene': sample['scene'], 'bbox_id': sample['bbox_id'],
                    'status': 'ok', 'predicted_object': manifest['vocabulary'][indices[0]],
                    'top5': [manifest['vocabulary'][i] for i in indices[:5]]}
                    for sample, indices in zip(manifest['samples'], order)]
                write_json(analysis / f'{split}_{name}_predictions.json', {'results': predictions})
        (analysis / f'{split}_table_8cols.tsv').write_text('\n'.join(table)+'\n')
        grouped = {}
        for method in NAMES:
            names = ['S0'] if method == 'S0' else [f'{method}_seed{s}' for s in (42, 43, 44)]
            grouped[method] = {'correct_by_seed': [rows[n]['correct'] for n in names]}
            for key in METRICS:
                values = [rows[n][key] for n in names]
                grouped[method][key] = {'mean': float(np.mean(values)),
                                        'sd': float(np.std(values, ddof=1)) if len(values)>1 else None}
            average_hit = np.mean([hits[n] for n in names], axis=0)
            groups = {'class': defaultdict(list), 'room': defaultdict(list)}
            for index, sample in enumerate(manifest['samples']):
                groups['class'][normalized_category(sample['category'])].append(index)
                groups['room'][sample['scene'].split('_')[0]].append(index)
            for kind, buckets in groups.items():
                for label, indices in sorted(buckets.items()):
                    base = hits['S0'][indices]
                    current = average_hit[indices]
                    diagnostics.append({'split': split, 'method': method, 'group': kind,
                        'name': label, 'support': len(indices), 'base_correct': int(base.sum()),
                        'mean_correct': float(current.sum()),
                        'delta_pp': float((current-base).mean()*100),
                        'mean_corrected': float(current[~base].sum()),
                        'mean_broken': float((1-current[base]).sum())})
        summaries[split] = {'count': result['count'], 'valid_RGB': result['valid_RGB'],
            'oov': sum(s['label']<0 for s in manifest['samples']),
            'rooms': len(set(s['scene'].split('_')[0] for s in manifest['samples'])),
            'scans': len(manifest['scenes']), 'methods': grouped}
    write_json(analysis / 'summary.json', summaries)
    write_json(analysis / 'class_room_changes.json', diagnostics)
    lines = ['# 扫描域监督分类头与双分支融合', '',
        '训练采用全部1201个官方训练扫描中的28730个词表内对象，固定197类。两模态各训练512→197线性头，'
        '冻结编码器，相同初始化、40轮、AdamW、学习率0.003、batch256；种子42/43/44。', '',
        f"主方法在预留集特征提取前锁定为{selection['primary_method']}；S3的RGB权重"
        f"{selection['fusion']['S3']['weight']}，S4为{selection['fusion']['S4']['weight']}。"
        '逐对象将197类分数标准化后加权。表中±为三个真实训练种子的样本标准差；S0为确定性结果。', '']
    for split, title in (('dev', '研究开发集（已多次查看）'), ('eval', '预留验证房间（本轮仅用于最终评估）')):
        info = summaries[split]
        lines += [f'## {title}', '',
            f"{info['rooms']}物理房间、{info['scans']}扫描、{info['count']}对象；RGB覆盖{info['valid_RGB']}/{info['count']}；"
            f"词表外{info['oov']}对象保留在分母并计错。", '',
            '| 方法 | True正确数（逐种子） | True | Exact Top-1 | Top-5 | Class Mean |',
            '|---|---:|---:|---:|---:|---:|']
        for method, row in info['methods'].items():
            values = []
            for key in METRICS:
                v = row[key]
                values.append(f"{v['mean']*100:.2f}%" + (f" ± {v['sd']*100:.2f}" if v['sd'] is not None else ''))
            lines.append('| '+' | '.join([NAMES[method], '/'.join(map(str,row['correct_by_seed'])), *values])+' |')
        lines.append('')
    primary = selection['primary_method']
    if primary != 'S0':
        value = audit['intervals'][primary]
        low, high = value['paired_room_95CI_pp']
        lines += [f"锁定主方法相对S0的预留集增益为{value['accuracy_delta_pp']:+.2f}个百分点；"
            f"物理房间配对bootstrap（10000次，seed20261019）95%区间[{low:+.2f}, {high:+.2f}]个百分点。"
            + ('区间跨0，不能据此宣称已确认泛化提升。' if low <= 0 <= high else '该区间仅反映这12个房间的抽样不确定性。'), '']
        before = summaries['dev']['methods']['S0']['class_mean_accuracy']['mean']
        after = summaries['dev']['methods'][primary]['class_mean_accuracy']['mean']
        lines += [f'类别均衡表现：研究开发集Class Mean由{before:.2%}变为{after:.2%}。'
            '本轮按True选择checkpoint和融合，不能宣称所有指标都提升；后续应在开发集研究长尾类别问题。'
            'S4的预留均值虽更高，仍保留为对照，不能按已揭示的预留表现替换本轮主方法。', '']
        rooms = [r for r in diagnostics if r['split']=='eval' and r['method']==primary and r['group']=='room']
        best_room = max(rooms, key=lambda r: r['mean_correct']-r['base_correct'])
        gain = best_room['mean_correct']-best_room['base_correct']
        total_gain = sum(r['mean_correct']-r['base_correct'] for r in rooms)
        lines += [f"房间分布：{sum(r['delta_pp']>1e-9 for r in rooms)}个房间改善，"
            f"{sum(abs(r['delta_pp'])<=1e-9 for r in rooms)}个不变，{sum(r['delta_pp']< -1e-9 for r in rooms)}个退步。"
            f"最大增益来自{best_room['name']}（{best_room['support']}对象，平均净增{gain:.2f}个正确对象）；"
            f"全部房间平均净增{total_gain:.2f}个，增益存在集中性。", '']
    lines += ['远端与本地独立审计通过：重算六个头的线性输出、checkpoint选择、完整开发融合网格、'
        '全部结果与纠正/误改，核对冻结源码/模型/特征哈希及物理房间隔离。原CUDA算子重编译后，先通过三个历史对象的特征一致性核查再提取预留集。', '',
        '预留资产修复：scene0217_00有31对类别和segment集合完全相同的重复标注；性能计算前保留每对首个ID，'
        '去除重复项，原始资产不变。修复清单、原始/派生标注及哈希已保存，其他21个扫描无此冲突。', '',
        '限制：旧3413对象是研究开发数据；新集合只有12个未用于本轮开发的验证房间，并非官方测试集。'
        '原发布PointNeXt主干是否见过这些房间尚未证实。本轮训练的是闭集分类头，不应描述为零样本。'
        '尚未运行Nr3D完整定位，不能将类别准确率增益等同于定位增益。', '',
        '可填表文件：dev_table_8cols.tsv与eval_table_8cols.tsv，两者必须分表注明数据角色；'
        '保留逐种子整数正确数，避免把平均正确数写成一次实际评估。']
    (analysis / 'report.md').write_text('\n'.join(lines)+'\n')
    write_json(analysis / 'summary_verification.json', {'status': 'passed',
        'summarizer_sha256': file_hash(Path(__file__)), 'splits': ['dev','eval'],
        'audit_sha256': file_hash(output / 'audit.json')})
    print((analysis / 'report.md').read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    summarize(parser.parse_args().output)
