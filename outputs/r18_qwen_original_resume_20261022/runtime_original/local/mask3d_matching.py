"""Class-blind, one-to-one mask association for a fixed-object comparison."""
import hashlib

import numpy as np
from scipy.optimize import linear_sum_assignment


def unique_predictions(masks, scores, classes):
    """Collapse identical geometry to the highest-score label before association."""
    keep, seen = [], set()
    for i in np.argsort(-np.asarray(scores), kind='stable'):
        key = hashlib.sha256(np.packbits(masks[:, i]).tobytes()).digest()
        if masks[:, i].any() and key not in seen:
            seen.add(key)
            keep.append(int(i))
    return masks[:, keep], np.asarray(scores)[keep], np.asarray(classes)[keep]


def associate(masks, instance_ids, target_ids, threshold=.5):
    """Maximize valid match count, then IoU; categories never enter assignment."""
    masks = np.asarray(masks, dtype=bool)
    target_ids = np.asarray(target_ids)
    if masks.ndim != 2 or len(instance_ids) != len(masks):
        raise ValueError('Mask/vertex dimensions differ')
    if len(np.unique(target_ids)) != len(target_ids):
        raise ValueError('Duplicate target instance')
    ious = np.zeros((len(target_ids), masks.shape[1]), dtype=np.float64)
    sizes = masks.sum(0)
    for i, oid in enumerate(target_ids):
        target = instance_ids == oid
        if not target.any():
            raise ValueError('Target instance has no vertices')
        intersection = masks[target].sum(0)
        ious[i] = intersection/(target.sum()+sizes-intersection)
    # The bonus makes one extra valid match outweigh every possible IoU change.
    valid = ious >= threshold
    reward = valid*(min(ious.shape, default=0)+1+ious)
    rows, cols = linear_sum_assignment(reward, maximize=True)
    return {int(i): (int(j), float(ious[i, j]))
            for i, j in zip(rows, cols) if valid[i, j]}
