"""Class-blind duplicate proposal suppression, using predicted masks only."""

import numpy as np


BIT_COUNTS = np.asarray([i.bit_count() for i in range(256)], dtype=np.uint8)


def mask_nms(packed_masks, scores, threshold=0.8):
    """Return confidence-ordered representatives and their suppressed mask indices."""
    packed = np.asarray(packed_masks, dtype=np.uint8)
    scores = np.asarray(scores, dtype=float)
    if packed.ndim != 2 or len(packed) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Expected one packed mask and finite confidence per proposal")
    sizes = BIT_COUNTS[packed].sum(axis=1, dtype=np.int64)
    if np.any(sizes == 0) or not 0 < threshold <= 1:
        raise ValueError("Expected nonempty masks and a valid NMS threshold")
    remaining = np.argsort(-scores, kind="stable")
    groups = {}
    while len(remaining):
        representative, others = int(remaining[0]), remaining[1:]
        intersection = BIT_COUNTS[packed[others] & packed[representative]].sum(axis=1, dtype=np.int64)
        overlaps = intersection / (sizes[others] + sizes[representative] - intersection)
        groups[representative] = [representative, *map(int, others[overlaps >= threshold])]
        remaining = others[overlaps < threshold]
    return list(groups), groups
