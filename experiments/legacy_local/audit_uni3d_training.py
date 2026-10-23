"""Independently verify the final Uni3D supervised results before reporting."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from category_common_v82 import file_hash, normalized_category
from run_storage import write_json
from uni3d_classifier import point_encoder_state


def audit(output):
    config = json.loads((output/'config.json').read_text())
    manifest = json.loads((Path(config['args']['manifests'])/'eval.json').read_text())
    vocabulary = config['vocabulary']
    rows = manifest['samples']
    reference = json.loads((Path(config['args']['point_bank_dir'])/'linear_predictions.json').read_text())['results']
    expected_keys = [(r['scene'], r['bbox_id']) for r in rows]
    assert expected_keys == [(r['scene'], r['bbox_id']) for r in reference]
    assert len(rows) == len(set(expected_keys)) == 3413
    assert sum(r['label'] < 0 for r in rows) == 157
    assert json.loads((output/'status.json').read_text())['stage'] == 'completed'
    table = ['模型\t评估对象数\tTrue正确数\tTrue准确率\tExact Top-1\tTop-5\tClass Mean\tErrors']
    results = {}
    for mode, label in (('linear', '冻结主干＋分类头'), ('finetune', '微调最后4层＋分类头')):
        logits = np.load(output/f'{mode}_logits.npy', allow_pickle=False)
        predictions = json.loads((output/f'{mode}_predictions.json').read_text())['results']
        saved = json.loads((output/f'{mode}_results.json').read_text())
        history = json.loads((output/f'{mode}_history.json').read_text())
        assert logits.shape == (3413, 197) and np.isfinite(logits).all()
        assert [(r['scene'], r['bbox_id']) for r in predictions] == expected_keys
        order = np.argsort(-logits, axis=-1, kind='stable')[:, :5]
        correct = exact = top5 = 0
        by_class = defaultdict(list)
        for row, prediction, ranked in zip(rows, predictions, order):
            categories = [vocabulary[i] for i in ranked]
            assert prediction['status'] == 'completed'
            assert prediction['predicted_object'] == categories[0]
            assert prediction['top5'] == categories
            target = normalized_category(row['category'])
            hit = normalized_category(categories[0]) == target
            correct += hit
            exact += categories[0] == row['category']
            top5 += target in [normalized_category(c) for c in categories]
            by_class[target].append(hit)
        summary = {'count': len(rows), 'correct': correct, 'exact_correct': exact,
                   'top5_correct': top5, 'accuracy': correct/len(rows),
                   'exact_accuracy': exact/len(rows), 'top5_accuracy': top5/len(rows),
                   'class_mean_accuracy': float(np.mean([np.mean(v) for v in by_class.values()])),
                   'errors': 0, 'targets_outside_vocabulary': 157}
        for key, value in summary.items():
            if key not in ('exact_correct', 'top5_correct'):
                assert abs(saved['summary'][key]-value) < 1e-6, (mode, key)
        best_epoch = min(history, key=lambda r: r['dev']['category_loss'])['epoch']
        assert saved['best_internal_dev_epoch'] == best_epoch
        results[mode] = {'summary': summary, 'best_internal_dev_epoch': best_epoch,
                         'epochs_completed': len(history),
                         'artifacts_sha256': {suffix: file_hash(output/f'{mode}_{suffix}')
                                              for suffix in ('best.pt', 'logits.npy', 'predictions.json', 'results.json')}}
        table.append(f'Uni3D-B（ScanNet训练，{label}）\t3413\t{correct}\t{summary["accuracy"]:.2%}'
                     f'\t{summary["exact_accuracy"]:.2%}\t{summary["top5_accuracy"]:.2%}'
                     f'\t{summary["class_mean_accuracy"]:.2%}\t0')
    original = point_encoder_state(torch.load(config['args']['checkpoint'], map_location='cpu', weights_only=True))
    adapted = torch.load(output/'finetune_best.pt', map_location='cpu', weights_only=True)['encoder']
    changed = []
    for name, value in original.items():
        allowed = (any(name.startswith(f'visual.blocks.{i}.') for i in range(8, 12))
                   or name.startswith(('visual.norm.', 'visual.fc_norm.', 'trans2embed.')))
        same = torch.equal(value, adapted['point_encoder.'+name])
        assert allowed or same, f'Frozen tensor changed: {name}'
        if not same:
            changed.append(name)
    assert all(any(n.startswith(f'visual.blocks.{i}.') for n in changed) for i in range(8, 12))
    assert any(n.startswith('trans2embed.') for n in changed)
    audit_result = {'passed': True, 'evaluation_order_matches_pointbert': True,
                    'all_scores_finite_and_rankings_consistent': True,
                    'all_frozen_tensors_unchanged': True, 'changed_tensors': changed,
                    'model_selection_verified_against_internal_dev_history': True,
                    'comparison': results}
    write_json(output/'audit.json', audit_result)
    (output/'table.tsv').write_text('\n'.join(table)+'\n')
    print(json.dumps(audit_result, ensure_ascii=False), flush=True)
    print('\n'.join(table), flush=True)
    return audit_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    audit(parser.parse_args().output)
