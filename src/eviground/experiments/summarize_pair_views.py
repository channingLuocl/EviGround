"""Write a compact report from the independently audited room-CV pilot."""
import argparse
import json
from pathlib import Path

import numpy as np

from eviground.categories.category_common_v82 import file_hash
from eviground.evaluation.run_storage import write_json


NAMES = {'S0': '原冻结3D', 'A': '普通CE＋固定融合', 'B': '长尾校正／Balanced＋固定融合',
         'C': 'B＋普通置信度门控', 'D': 'B＋普通逐视图质量门控', 'E': 'B＋类别对视图支持门控（主候选）'}


def main(args):
    p = args.output
    audit = json.loads((p/'audited_summary.json').read_text())
    assert audit['status'] == 'passed'
    outcomes = np.load(p/'review_outcomes.npz')
    y = outcomes['labels']
    outer = outcomes['outer']
    methods = audit['methods']
    averages = {m: float(np.mean(v['accuracy_by_seed'])) for m, v in methods.items()}
    control = max(('A', 'B', 'C', 'D'), key=lambda m: averages[m])
    e, b = methods['E'], methods[control]
    net = np.array(e['correct_by_seed'])-np.array(b['correct_by_seed'])
    meets = (averages['E']-averages[control] >= .005 and
             np.mean(e['macro_by_seed']) >= np.mean(b['macro_by_seed']) and (net > 0).sum() >= 2)
    decision = {'primary': 'E', 'strongest_control_by_outer_accuracy': control,
        'delta_pp': 100*(averages['E']-averages[control]),
        'macro_delta_pp': 100*(np.mean(e['macro_by_seed'])-np.mean(b['macro_by_seed'])),
        'net_correct_by_seed': net.tolist(), 'engineering_threshold_met': bool(meets),
        'qualification': 'exploratory budget decision; no claim of method novelty or independent backbone generalization'}
    analysis = p/'analysis'
    analysis.mkdir(exist_ok=True)
    lines = ['# 类别对视图证据门控：训练侧三外折实验', '',
        f"{audit['rooms']}物理房间、{audit['scans']}扫描、{audit['objects']}个词表内对象、{audit['classes_present']}个实际出现类别。"
        '各房间的所有重扫描分在同一外折，外折内三折交叉拟合训练门控；种子42/43/44。', '',
        '本表是官方训练对象上的新分类头／融合模块验证，不是Nr3D定位结果。原PointNeXt主干可能见过这些房间；'
        '没有词表外对象，因此不能与旧3413开发集或404对象验证集的数字直接比较。', '',
        '| 方法 | 正确数42/43/44 | 类别Top-1 | Top-5 | 出现类别Macro |', '|---|---:|---:|---:|---:|']
    tsv = ['方法\t对象数\t正确数seed42\t正确数seed43\t正确数seed44\tTop1均值\tTop1种子SD\tTop5均值\tMacro均值\tMacro种子SD']
    for m, value in methods.items():
        fields = []
        for key in ('accuracy_by_seed', 'top5_by_seed', 'macro_by_seed'):
            values = np.array(value[key])*100
            fields.append(f'{values.mean():.2f}% ± {values.std(ddof=1):.2f}')
        lines.append('| '+' | '.join([NAMES[m], '/'.join(map(str, value['correct_by_seed'])), *fields])+' |')
        tsv.append('\t'.join([NAMES[m], str(audit['objects']), *map(str, value['correct_by_seed']),
            f"{np.mean(value['accuracy_by_seed']):.6%}", f"{np.std(value['accuracy_by_seed'],ddof=1):.6%}",
            f"{np.mean(value['top5_by_seed']):.6%}", f"{np.mean(value['macro_by_seed']):.6%}",
            f"{np.std(value['macro_by_seed'],ddof=1):.6%}"]))
    lines += ['', '表中±为三个训练种子的样本标准差。每个种子的三个外折预测先合并，再计算上述指标；'
              '不是将三个外折Macro直接平均。S0三个种子共用相同冻结输出。', '',
              '## 主候选相对各对照', '',
              '| E减去对照 | Top-1差（百分点） | 房间配对95%区间 | 纠正数42/43/44 | 改错数42/43/44 |',
              '|---|---:|---:|---:|---:|']
    for m, row in audit['E_comparisons'].items():
        low, high = row['room_95CI_pp']
        lines.append(f"| {m} | {row['delta_pp']:+.2f} | [{low:+.2f}, {high:+.2f}] | "
                     +'/'.join(map(str,row['corrected_by_seed']))+' | '+'/'.join(map(str,row['broken_by_seed']))+' |')
    lines += ['', '区间按物理房间bootstrap10000次；三种子先逐对象平均。区间针对固定分折模型的房间抽样，'
              '没有重训整个嵌套流程，也没有消除预训练暴露或多重探索影响。', '',
              f"按外折平均Top-1最高的对照为{control}；E的差值{decision['delta_pp']:+.2f}个百分点，"
              f"Macro差值{decision['macro_delta_pp']:+.2f}个百分点，三种子净正确数变化{net.tolist()}。", '',
              ('达到预先设定的继续投入门槛；仍需完整多视图attention／容量对照、机制消融及新的评估策略，不能据此宣布创新成立。'
               if meets else '没有达到预先设定的继续投入门槛。本轮不把类别对门控升级为有效的新方法；保留全部负结果，并先检查强基线与误差来源。'), '',
              '## 分折与设置', '', '| 外折 | 房间数 | 对象数 | A设置 | B设置 |', '|---|---:|---:|---|---|']
    for f in range(3):
        selection = json.loads((p/f'outer{f}/selection.json').read_text())
        describe = lambda r: f"{r['kind']}, tau={r['tau']}, w={r['weight']}"
        mask = outer == f
        lines.append(f"| {f} | {len(set(outcomes['room_index'][mask]))} | {mask.sum()} | {describe(selection['A'])} | {describe(selection['B'])} |")
    lines += ['', 'A/B的设置只由该外折训练部分的OOF分数选择；C/D/E使用同一个B设置。'
              '72个RGB线性头固定40epoch最后一轮；27个门控固定40epoch。全部编码器冻结。', '',
              'C/D/E均为16→16→1，共289参数；C只看聚合分数置信度，D看普通跨视图质量，E看候选类别对的跨视图支持。'
              'D不是完整view-attention，尚不能声称超过所有多视图学习方案。', '',
              f"四视图均值与旧RGB缓存桥接检查通过。独立NumPy特征及网络重算的门控权重最大差"
              f"{audit['maximum_independent_gate_weight_difference']:.3g}；模型、数据、分折、OOF网格、外折分数与指标审计通过。", '',
              '本轮没有改动旧冻结结果、没有用404对象调参、没有接入Nr3D完整定位，也没有证明方法首创性。']
    (analysis/'report.md').write_text('\n'.join(lines)+'\n')
    (analysis/'table.tsv').write_text('\n'.join(tsv)+'\n')
    write_json(analysis/'decision.json', decision)
    frequency = np.zeros(len(y), dtype=int)
    for f in range(3):
        counts = np.bincount(y[outer != f], minlength=197)
        frequency[outer == f] = counts[y[outer == f]]
    bins = {}
    for name, mask in [('zero',frequency == 0), ('tail_1_19',(frequency > 0) & (frequency < 20)),
                        ('mid_20_99',(frequency >= 20) & (frequency < 100)), ('head_100_plus',frequency >= 100)]:
        bins[name] = {'objects':int(mask.sum()), 'correct_by_seed':
            {m: outcomes[f'{m}_hit'][:,mask].sum(1).tolist() for m in methods}}
    write_json(analysis/'frequency_groups.json', bins)
    write_json(analysis/'report_provenance.json', {'audit_sha256': file_hash(p/'audited_summary.json'),
        'outcomes_sha256':file_hash(p/'review_outcomes.npz'), 'summarizer_sha256':file_hash(Path(__file__)),
        'report_sha256':file_hash(analysis/'report.md')})
    print((analysis/'report.md').read_text(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
