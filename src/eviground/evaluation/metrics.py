"""Geometry metrics for 3D grounding evaluation."""

from __future__ import annotations
from typing import Sequence


def bbox_iou_3d(first: Sequence[float], second: Sequence[float]) -> float:
    """Compute axis-aligned 3D IoU for [cx, cy, cz, sx, sy, sz] boxes."""
    if len(first) != 6 or len(second) != 6:
        raise ValueError("3D boxes must contain six values")
    a_min = [first[i] - first[i + 3] / 2 for i in range(3)]
    a_max = [first[i] + first[i + 3] / 2 for i in range(3)]
    b_min = [second[i] - second[i + 3] / 2 for i in range(3)]
    b_max = [second[i] + second[i + 3] / 2 for i in range(3)]
    intersection = 1.0
    for axis in range(3):
        intersection *= max(
            0.0, min(a_max[axis], b_max[axis]) - max(a_min[axis], b_min[axis])
        )
    volume_a = first[3] * first[4] * first[5]
    volume_b = second[3] * second[4] * second[5]
    union = volume_a + volume_b - intersection
    return intersection / union if union > 0 else 0.0
