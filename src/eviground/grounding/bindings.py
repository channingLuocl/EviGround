"""Query-conditioned category bindings without evaluation labels or learned cutoffs."""

import re
import numpy as np
from eviground.categories.retrieval import (
    match_retrieval_categories,
    retrieval_category,
)


def target_head_hint(query, categories):
    """Recognize only a leading category noun, not subjects in later clauses."""
    match = re.match(
        "^\\s*(?:please\\s+)?(?:(?:select|choose|pick|find|locate)\\s+)?(?:the|a|an|this|that)\\s+(.+?)\\s+(?:in|on|with|under|above|below|beneath|at|that|which)\\b",
        query,
        re.I,
    )
    if not match or re.search(
        ",\\s*(?:the|a|an|this|that)\\s+", query[match.end() :], re.I
    ):
        return None
    name = retrieval_category(match.group(1))
    if not name or name not in {retrieval_category(c) for c in categories}:
        return None
    return {"category": name, "source_quote": match.group(1)}


def matching_hypotheses(names, item):
    return [
        h
        for h in item.get("category_hypotheses", [])
        if match_retrieval_categories(names, [{**item, "category_hypotheses": [h]}])
    ]


def generic_target_pool(names):
    return bool(names) and all(
        (retrieval_category(name) in ("object", "thing", "item") for name in names)
    )


def camera_eye(lower, upper, direction):
    """Fit every bounding-box corner within the renderer's 65-degree FOV."""
    at = (lower + upper) / 2
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    right = np.cross(-direction, [0.0, 0.0, 1.0])
    if np.linalg.norm(right) < 1e-06:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, -direction)
    corners = (
        np.array(
            [
                [x, y, z]
                for x in (lower[0], upper[0])
                for y in (lower[1], upper[1])
                for z in (lower[2], upper[2])
            ]
        )
        - at
    )
    half_fov = np.tan(np.deg2rad(65 / 2))
    radial = np.maximum(np.abs(corners @ right), np.abs(corners @ up))
    distance = max(float(np.max(corners @ direction + 1.12 * radial / half_fov)), 0.3)
    return (at + distance * direction, at)
