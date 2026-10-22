"""Audit packed proposal masks, class-blind associations and fixed-denominator scores."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import runpy

import numpy as np

from category_common_v82 import file_hash, normalized_category
from mask3d_matching import associate
from run_storage import write_json


def audit(root):
    output = root/'full'
    manifest = json.loads((root/'../seeground-v10/manifests/eval.json').read_text())
    for name, digest in json.loads((output/'FROZEN.json').read_text()).items():
        assert file_hash(output/name) == digest
    result = json.loads((output/'results.json').read_text())
    protocol = result['protocol']
    assert file_hash(root/'assets/scannet200_val.ckpt') == protocol['checkpoint_sha256']
    constants = runpy.run_path(str(root/'assets/Mask3D/datasets/scannet200/scannet200_constants.py'))
    labels = constants['CLASS_LABELS_200']
    predictions = json.loads((output/'predictions.json').read_text())['results']
    by_scene, targets = defaultdict(list), {}
    for row in manifest['samples']:
        by_scene[row['scene']].append(row)
        targets[row['scene'], row['bbox_id']] = row
    assert len(predictions) == len(targets) == 3413
    assert len(by_scene) == 130
    for scene, rows in by_scene.items():
        record = json.loads((output/f'scenes/{scene}.json').read_text())
        mask_path = output/f'scenes/{scene}.masks.npz'
        assert file_hash(mask_path) == record['masks_sha256']
        with np.load(mask_path, allow_pickle=False) as data:
            masks = np.unpackbits(data['packed_masks'], axis=0,
                                  count=int(data['vertices'])).astype(bool)
            classes, scores = data['classes'], data['scores']
        with np.load(rows[0]['cloud'], allow_pickle=False) as cloud:
            matches = associate(masks, cloud['ids'], [r['bbox_id'] for r in rows])
        assert len({j for j, _ in matches.values()}) == len(matches)
        for i, prediction in enumerate(record['results']):
            assert (prediction['scene'], prediction['bbox_id']) == (scene, rows[i]['bbox_id'])
            assert prediction['matched'] == (i in matches)
            if i in matches:
                j, iou = matches[i]
                label_index = 1 if int(classes[j]) == 0 else int(classes[j])+2
                name = labels[label_index] if label_index < len(labels) else f'unmapped_model_class_{classes[j]}'
                assert prediction['predicted_object'] == name
                assert np.isclose(prediction['mask_iou'], iou, atol=1e-12)
                assert np.isclose(prediction['score'], scores[j], atol=1e-12)
            else:
                assert prediction['predicted_object'] is None
    correct, exact, groups, seen = [], [], defaultdict(list), set()
    for prediction in predictions:
        key = prediction['scene'], prediction['bbox_id']
        assert key not in seen
        seen.add(key)
        target = targets[key]['category']
        name = prediction['predicted_object']
        hit = name is not None and normalized_category(name) == normalized_category(target)
        correct.append(hit)
        exact.append(name == target)
        groups[normalized_category(target)].append(hit)
    matched = sum(p['matched'] for p in predictions)
    values = {'count': len(correct), 'correct': sum(correct), 'accuracy': float(np.mean(correct)),
        'exact_accuracy': float(np.mean(exact)), 'class_mean_accuracy': float(np.mean([np.mean(v) for v in groups.values()])),
        'errors': 0, 'matched': matched, 'coverage': matched/len(correct),
        'matched_accuracy': sum(correct)/matched}
    for key, value in values.items():
        assert np.isclose(value, result['summary'][key], atol=1e-12), key
    assert result['summary']['top5_accuracy'] is None
    write_json(output/'independent_audit.json', {'all_checks_passed': True,
        'scenes': 130, 'packed_masks_and_associations_verified': True, 'summary': values})
    print(json.dumps(values), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    audit(parser.parse_args().root)
