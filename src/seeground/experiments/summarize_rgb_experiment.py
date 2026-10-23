"""Write reviewed metrics, eight-column table, and class-level diagnostics."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from seeground.experiments.audit_shape_scale import recount, normalized_category
from seeground.categories.category_common_v82 import file_hash
from seeground.evaluation.run_storage import write_json


NAMES = {'V0': 'SeeGround PointNeXt+CLIP（原基线）',
         'V1': '冻结CLIP ViT-B/16（固定4视角RGB）',
         'V2': '自研固定融合（PointNeXt＋冻结CLIP RGB）',
         'V3_seed42': '融合＋尺寸残差（seed42）',
         'V3_seed43': '融合＋尺寸残差（seed43）',
         'V3_seed44': '融合＋尺寸残差（seed44）'}


def main(output):
    audit = json.loads((output / 'audit.json').read_text())
    assert audit['status'] == 'passed'
    report = json.loads((output / 'results.json').read_text())
    manifest = json.loads((output / 'manifests/eval.json').read_text())
    analysis = output / 'analysis'
    analysis.mkdir(exist_ok=True)
    table = ['模型\t评估对象数\tTrue正确数\tTrue准确率\tExact Top-1\tTop-5\tClass Mean\tErrors']
    lines = ['# 冻结CLIP多视角RGB融合：第二阶段结果', '',
             '沿用固定3413对象、197类候选；157个词表外目标保留在分母。图像分支与文本空间均为OpenAI CLIP ViT-B/16，无新图像头训练。', '',
             '| 模型 | 正确数 | True | Exact | Top-5 | Class Mean | 纠正/误改 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    hits, per_class = {}, []
    for row in report['records']:
        method, claimed = row['method'], row['summary']
        logits = np.load(output / f'eval/{method}.npy')
        summary, hit = recount(logits, manifest)
        for key, value in summary.items():
            assert abs(value - claimed[key]) < 1e-12
        hits[method] = hit
        values = [f"{summary[key]:.2%}" for key in ('accuracy', 'exact_accuracy', 'top5_accuracy', 'class_mean_accuracy')]
        table.append('\t'.join([NAMES[method], str(summary['count']), str(summary['correct']), *values, '0']))
        lines.append('| ' + ' | '.join([NAMES[method], str(summary['correct']), *values,
                                      f"{row['corrected']}/{row['broken']}"]) + ' |')
        groups = defaultdict(list)
        for sample, value in zip(manifest['samples'], hit):
            groups[normalized_category(sample['category'])].append(bool(value))
        per_class.extend({'method': method, 'category': name, 'support': len(values),
                          'correct': sum(values), 'accuracy': float(np.mean(values))} for name, values in sorted(groups.items()))
        if method in ('V0', 'V1', 'V2'):
            order = np.argsort(-logits, axis=1, kind='stable')
            predictions = [{'scene': sample['scene'], 'bbox_id': sample['bbox_id'], 'status': 'ok',
                            'predicted_object': manifest['vocabulary'][indices[0]],
                            'top5': [manifest['vocabulary'][i] for i in indices[:5]]}
                           for sample, indices in zip(manifest['samples'], order)]
            write_json(analysis / f'{method}_predictions.json', {'results': predictions})
    w, c = report['selection']['V2']['weights']
    w3, c3 = report['selection']['V3']['weights']
    interval = audit['intervals']['V2']
    ci = interval['paired_room_bootstrap_95CI_pp']
    lines += ['', f'开发集唯一选择：V2图像权重w={w}；V3图像权重w={w3}、尺寸权重c={c3}。使用每对象197类分数的均值/标准差标准化后融合。开发集提取、网格选择及冻结文件均在eval图像提取之前完成。', '',
              f"V2相对基线为{interval['accuracy_delta_percentage_points']:+.2f}个百分点；按{audit['physical_rooms']}个物理房间配对bootstrap（10000次，seed20261019）的95%区间为[{ci[0]:+.2f}, {ci[1]:+.2f}]个百分点。此区间仅反映抽样不确定性，未作多重比较校正。", '',
              f"图像覆盖{report['valid_RGB']}/{report['total']}个对象；不丢弃缺失样本。V0/V1/V2是冻结确定性结果，不能将重复推理当作独立训练种子。V3复用三个真实训练尺寸头；当尺寸系数为0时三个输出相同，不代表训练方差被完整估计。", '',
              f"图像与3D二选一、由GT事后挑正确项的oracle为{report['diagnostic_oracle_accuracy']:.2%}，只说明互补性潜力，不能填作可部署模型准确率。", '',
              '数据核查：全部32256个训练/开发/评估对象有RGB，缓存bbox与点云一致、现有GT标签一致。train/dev分别25672/3171对象额外核对原始aggregation；eval未找到原始aggregation。本轮未证实主干预训练场景范围。开发集93.35%与评估66.28%的差距不能仅由类别频率解释，详见数据核查记录。', '',
              '本轮输入是真实扫描对象的点云渲染，未使用原始相机RGB帧、可学习门控或新图像分类头；不等同完整多模态方法上限。此eval此前已多次查看，不能称为独立测试集；尚未运行新的Nr3D完整定位，不能宣称定位指标已提升。', '',
              '远端与本地独立重算通过。原模型与旧结果保留；本轮源代码、配置、清单哈希、开发集完整网格、图像特征、分数分量和最终logits均留存。']
    delta = []
    for name in sorted(set(normalized_category(s['category']) for s in manifest['samples'])):
        mask = np.array([normalized_category(s['category']) == name for s in manifest['samples']])
        delta.append({'category': name, 'support': int(mask.sum()),
                      'base_correct': int(hits['V0'][mask].sum()), 'fused_correct': int(hits['V2'][mask].sum()),
                      'corrected': int((~hits['V0'][mask] & hits['V2'][mask]).sum()),
                      'broken': int((hits['V0'][mask] & ~hits['V2'][mask]).sum())})
    (analysis / 'table_8cols.tsv').write_text('\n'.join(table)+'\n')
    (analysis / 'report.md').write_text('\n'.join(lines)+'\n')
    write_json(analysis / 'per_class.json', per_class)
    write_json(analysis / 'class_changes.json', delta)
    write_json(analysis / 'summary_verification.json', {'status': 'passed', 'evaluations': len(report['records']),
               'summarizer_sha256': file_hash(Path(__file__)), 'metrics_independently_recounted': True})
    print((analysis / 'report.md').read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    main(parser.parse_args().output)
