"""Category ranks and class-blind instance association for ScanRefer."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def model_class_names(class_labels):
    """Invert Mask3D ScanNet200's clamp(label_index - 2, min=0).

    Wall (index 0) and floor (index 2) are excluded during training.
    Chair (index 1) is clamped to model class 0. Two unused output heads
    remain because the released checkpoint has 200 foreground outputs.
    """
    if len(class_labels) != 200 or list(class_labels[:3]) != ['wall', 'chair', 'floor']:
        raise ValueError('Unexpected ScanNet200 class order')
    return [class_labels[1], *class_labels[3:]]


def ranked_categories(probabilities, class_names, k=5):
    """Rank real category heads, without alias normalization or GT filtering."""
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 2 or probabilities.shape[1] != 200:
        raise ValueError('Expected the 200 foreground Mask3D class scores')
    if len(class_names) != 198 or not np.isfinite(probabilities).all():
        raise ValueError('Invalid class mapping or class scores')
    indices = np.argsort(-probabilities[:, :198], axis=1, kind='stable')[:, :k]
    labels = [[class_names[int(i)] for i in row] for row in indices]
    return indices, labels


def unique_mask_indices(masks, scores):
    """Keep the highest-confidence prediction for each identical nonempty mask."""
    seen, keep = set(), []
    for j in np.argsort(-np.asarray(scores), kind='stable'):
        if not masks[:, j].any():
            continue
        key = np.packbits(masks[:, j]).tobytes()
        if key not in seen:
            seen.add(key)
            keep.append(int(j))
    return keep


def associate(masks, instance_ids, target_ids, threshold=0.5):
    """One-to-one mask IoU matching: maximize valid count, then total IoU."""
    masks = np.asarray(masks, dtype=bool)
    instance_ids = np.asarray(instance_ids)
    if masks.ndim != 2 or len(instance_ids) != len(masks):
        raise ValueError('Mask and GT vertex counts differ')
    if len(set(target_ids)) != len(target_ids):
        raise ValueError('Duplicate target object')
    ious = np.zeros((len(target_ids), masks.shape[1]), dtype=np.float64)
    sizes = masks.sum(0)
    for i, target_id in enumerate(target_ids):
        target = instance_ids == target_id
        if not target.any():
            raise ValueError(f'Target {target_id} has no vertices')
        intersection = masks[target].sum(0)
        ious[i] = intersection / (target.sum() + sizes - intersection)
    valid = ious >= threshold
    reward = valid * (min(ious.shape) + 1 + ious)
    rows, cols = linear_sum_assignment(reward, maximize=True)
    return {int(i): (int(j), float(ious[i, j]))
            for i, j in zip(rows, cols) if valid[i, j]}


def summarize(rows):
    """Unmatched referenced objects remain in the denominator and count wrong."""
    if not rows:
        raise ValueError('No objects evaluated')
    keys = {(r['scene_id'], r['object_id']) for r in rows}
    if len(keys) != len(rows):
        raise ValueError('Evaluation contains duplicate objects')
    top1 = sum(bool(r['top5']) and r['top5'][0] == r['gt_label'] for r in rows)
    top5 = sum(r['gt_label'] in r['top5'] for r in rows)
    matched = sum(r['matched'] for r in rows)
    return {'objects': len(rows), 'matched': matched, 'unmatched': len(rows) - matched,
            'exact_top1_correct': top1, 'exact_top5_correct': top5,
            'exact_top1_percent': 100 * top1 / len(rows),
            'exact_top5_percent': 100 * top5 / len(rows)}
