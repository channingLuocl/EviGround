"""Recount exported rankings and audit internal-dev checkpoint selection."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from eviground.categories.category_common_v82 import file_hash, normalized_category
from eviground.classifiers.category_v10 import run
from eviground.classifiers.category_v10.model import CategoryHead
from eviground.evaluation.run_storage import write_json


def recount(logits, manifest):
    order = np.argsort(-logits, axis=1, kind='stable')
    vocabulary = manifest['vocabulary']
    correct, exact, top5, groups = [], [], [], defaultdict(list)
    for row, indices in zip(manifest['samples'], order):
        name, target = vocabulary[indices[0]], row['category']
        hit = normalized_category(name) == normalized_category(target)
        correct.append(hit)
        exact.append(name == target)
        top5.append(any(normalized_category(vocabulary[i]) == normalized_category(target) for i in indices[:5]))
        groups[normalized_category(target)].append(hit)
    return {'count': len(order), 'correct': sum(correct), 'accuracy': float(np.mean(correct)),
            'exact_accuracy': float(np.mean(exact)), 'top5_accuracy': float(np.mean(top5)),
            'class_mean_accuracy': float(np.mean([np.mean(v) for v in groups.values()])), 'errors': 0}


def audit(output, feature_root):
    torch.set_num_threads(4)
    assert json.loads((output/'status.json').read_text())['stage'] == 'completed'
    manifest = json.loads((feature_root/'manifests/eval.json').read_text())
    records = json.loads((output/'experiment_summary.json').read_text())['comparison']
    assert len(records) == 9
    for record in records:
        folder = output/f'predictions/seed{record["seed"]}/{record["mode"]}'
        for name, digest in json.loads((folder/'FROZEN.json').read_text()).items():
            assert file_hash(folder/name) == digest
        logits = np.load(folder/'logits.npy', allow_pickle=False)
        independent = recount(logits, manifest)
        for key, value in independent.items():
            assert np.isclose(record['summary'][key], value, atol=1e-7), (record, key, value)
        checkpoint_path = output/f'models/seed{record["seed"]}/{record["mode"]}/head.pt'
        assert file_hash(checkpoint_path) == record['head_sha256']
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        history = json.loads(checkpoint_path.with_name('history.json').read_text())
        assert len(history) == 40
        assert checkpoint['best_epoch'] == min(history, key=lambda r: r['dev']['category_loss'])['epoch']
        predictions = json.loads((folder/'predictions.json').read_text())['results']
        order = np.argsort(-logits, axis=1, kind='stable')
        for prediction, row, indices in zip(predictions, manifest['samples'], order):
            assert (prediction['scene'], prediction['bbox_id']) == (row['scene'], row['bbox_id'])
            assert prediction['top5'] == [manifest['vocabulary'][i] for i in indices[:5]]
    # Recount the unchanged encoder/text control using the same cached inputs.
    data, _ = run.read_cache(feature_root/'features/eval/features.npz')
    checkpoint = torch.load(output/'models/seed42/global/head.pt', map_location='cpu', weights_only=True)
    frozen = CategoryHead(checkpoint['state_dict']['category_text'], 'frozen')
    logits = run.evaluate_head(frozen, data, 64, 'cpu').numpy()
    folder = output/'frozen_cache_control'
    folder.mkdir(exist_ok=True)
    np.save(folder/'logits.npy', logits, allow_pickle=False)
    write_json(folder/'results.json', recount(logits, manifest))
    result = {'models_verified': 9, 'all_checks_passed': True,
              'frozen_same_cache_control': recount(logits, manifest),
              'frozen_logits_sha256': file_hash(folder/'logits.npy')}
    write_json(output/'independent_audit.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--feature-root', type=Path, required=True)
    args = parser.parse_args()
    audit(args.output, args.feature_root)
