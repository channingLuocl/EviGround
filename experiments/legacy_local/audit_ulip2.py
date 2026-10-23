"""Recompute ULIP-2 results from scores and check selection and frozen weights."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from category_common_v82 import category_vocabulary, file_hash, normalized_category
from run_storage import write_json
from ulip2_classifier import CHECKPOINT_SHA256, encoder_state


def recalculate(rows, predictions, scores, vocabulary):
    assert scores.shape == (len(rows), 197) and np.isfinite(scores).all()
    assert [(r['scene'], r['bbox_id']) for r in rows] == [
        (r['scene'], r['bbox_id']) for r in predictions]
    order = np.argsort(-scores, axis=-1, kind='stable')[:, :5]
    correct = exact = top5 = 0
    by_class = defaultdict(list)
    for row, prediction, ranked in zip(rows, predictions, order):
        categories = [vocabulary[i] for i in ranked]
        assert prediction['status'] == 'completed'
        assert prediction['predicted_object'] == categories[0]
        actual_top5 = [v['category'] if isinstance(v, dict) else v for v in prediction['top5']]
        assert actual_top5 == categories
        target = row.get('category', row.get('target'))
        hit = normalized_category(categories[0]) == normalized_category(target)
        correct += hit
        exact += categories[0] == target
        top5 += normalized_category(target) in [normalized_category(c) for c in categories]
        by_class[normalized_category(target)].append(hit)
    return {'count': len(rows), 'correct': correct, 'exact_correct': exact, 'top5_correct': top5,
            'accuracy': correct/len(rows), 'exact_accuracy': exact/len(rows),
            'top5_accuracy': top5/len(rows),
            'class_mean_accuracy': float(np.mean([np.mean(v) for v in by_class.values()])), 'errors': 0}


def assert_summary(actual, saved):
    for key, value in actual.items():
        if key in saved:
            assert abs(saved[key]-value) < 1e-6, (key, saved[key], value)


def zero_shot(output):
    saved = json.loads((output/'results.json').read_text())
    predictions = json.loads((output/'predictions.json').read_text())['results']
    selection = json.loads((output/'selection.json').read_text())['instances']
    reference = Path('/root/autodl-tmp/v8_2_openshape_full_3413')
    assert selection == json.loads((reference/'selection.json').read_text())['instances']
    assert json.loads((output/'inputs.json').read_text()) == json.loads((reference/'inputs.json').read_text())
    assert len(selection) == len({(r['scene'], r['bbox_id']) for r in selection}) == 3413
    assert saved['summary']['complete'] and saved['summary']['targets_outside_vocabulary'] == 157
    scores = np.array([r['scores'] for r in predictions])
    result = recalculate(saved['results'], predictions, scores, list(category_vocabulary()))
    assert_summary(result, saved['summary'])
    np.save(output/'audit_scores.npy', scores, allow_pickle=False)
    audit = {'passed': True, 'summary': result, 'identical_selection_and_input_hashes': True,
             'all_scores_finite_and_rankings_consistent': True,
             'artifacts_sha256': {f: file_hash(output/f) for f in
                                  ('config.json', 'runtime.json', 'results.json', 'predictions.json', 'audit_scores.npy')}}
    write_json(output/'audit.json', audit)
    print(json.dumps(audit), flush=True)
    return result


def training(output):
    config = json.loads((output/'config.json').read_text())
    manifest = json.loads((Path(config['args']['manifests'])/'eval.json').read_text())
    rows, vocabulary = manifest['samples'], config['vocabulary']
    reference = json.loads((Path(config['args']['point_bank_dir'])/'linear_predictions.json').read_text())['results']
    keys = [(r['scene'], r['bbox_id']) for r in rows]
    assert keys == [(r['scene'], r['bbox_id']) for r in reference]
    assert len(keys) == len(set(keys)) == 3413 and sum(r['label'] < 0 for r in rows) == 157
    assert json.loads((output/'status.json').read_text())['stage'] == 'completed'
    results = {}
    for mode in ('linear', 'finetune'):
        logits = np.load(output/f'{mode}_logits.npy', allow_pickle=False)
        predictions = json.loads((output/f'{mode}_predictions.json').read_text())['results']
        saved = json.loads((output/f'{mode}_results.json').read_text())
        summary = recalculate(rows, predictions, logits, vocabulary)
        assert_summary(summary, saved['summary'])
        history = json.loads((output/f'{mode}_history.json').read_text())
        best_epoch = min(history, key=lambda r: r['dev']['category_loss'])['epoch']
        assert saved['best_internal_dev_epoch'] == best_epoch
        results[mode] = {'summary': summary, 'best_internal_dev_epoch': best_epoch,
                         'epochs_completed': len(history),
                         'artifacts_sha256': {f: file_hash(output/f'{mode}_{f}') for f in
                                              ('best.pt', 'results.json', 'logits.npy', 'predictions.json')}}
    assert file_hash(config['args']['checkpoint']) == CHECKPOINT_SHA256
    original = encoder_state(torch.load(config['args']['checkpoint'], map_location='cpu', weights_only=False))
    adapted = torch.load(output/'finetune_best.pt', map_location='cpu', weights_only=True)['encoder']
    assert set(original) == set(adapted)
    changed = []
    for name, value in original.items():
        allowed = (name == 'pc_projection' or name.startswith('point_encoder.norm.')
                   or any(name.startswith(f'point_encoder.blocks.blocks.{i}.') for i in range(14, 18)))
        same = torch.equal(value, adapted[name])
        assert allowed or same, f'Frozen tensor changed: {name}'
        if not same:
            changed.append(name)
    assert 'pc_projection' in changed
    assert all(any(n.startswith(f'point_encoder.blocks.blocks.{i}.') for n in changed) for i in range(14, 18))
    audit = {'passed': True, 'comparison': results, 'changed_tensors': changed,
             'all_frozen_tensors_unchanged': True, 'model_selection_verified': True,
             'evaluation_order_matches_pointbert': True}
    write_json(output/'audit.json', audit)
    print(json.dumps(audit), flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zero-shot', type=Path)
    parser.add_argument('--training', type=Path)
    args = parser.parse_args()
    if args.zero_shot:
        zero_shot(args.zero_shot)
    if args.training:
        training(args.training)
