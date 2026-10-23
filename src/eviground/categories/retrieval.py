"""Query matching and measured geometry; no ground-truth labels are used."""

import re
from eviground.categories.hypotheses import category_names
import numpy as np

ALIASES = {
    "television": "tv",
    "televisions": "tv",
    "screen": "tv",
    "screens": "tv",
    "monitor": "tv",
    "monitors": "tv",
    "trashcan": "trash can",
    "trashcans": "trash can",
    "trash cans": "trash can",
    "garbage bin": "trash can",
    "garbage can": "trash can",
    "fridge": "refrigerator",
    "refrigerators": "refrigerator",
    "front doors": "door",
    "front door": "door",
    "double doors": "door",
    "double door": "door",
    "doors": "door",
    "windows": "window",
    "chairs": "chair",
    "tables": "table",
    "cabinets": "cabinet",
    "curtains": "curtain",
    "carts": "cart",
    "photocopiers": "copier",
    "photocopier": "copier",
    "copiers": "copier",
    "blackboards": "blackboard",
    "countertop": "counter",
    "countertops": "counter",
}


def _viewpoint_origin(viewpoint):
    """Extract a numeric observer origin without inventing one from a category."""
    if viewpoint is None:
        return None
    value = viewpoint
    if isinstance(value, dict):
        if "origin" in value and (
            not any((key in value for key in ("xyz", "point", "position")))
        ):
            value = value.get("origin")
        if isinstance(value, dict):
            for key in ("xyz", "point", "position", "origin_xyz"):
                if key in value:
                    value = value[key]
                    break
    try:
        point = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if point.size < 3 or not np.isfinite(point[:3]).all():
        return None
    return point[:3]


def distance_pair_metrics(target, anchor, viewpoint=None):
    """Compute separately named pair distances for one bound target/anchor.

    ``surface_gap`` is the AABB gap proxy.  A finite fitted wall can provide a
    ``plane`` distance; otherwise that metric remains unknown instead of using
    the wall center as a surrogate.  A viewpoint distance is only present when
    a numeric observer origin is supplied.
    """
    a = np.asarray(target["bbox_3d"], dtype=float)
    b = np.asarray(anchor["bbox_3d"], dtype=float)
    gap_xyz = np.maximum(np.abs(a[:3] - b[:3]) - (a[3:] + b[3:]) / 2.0, 0.0)
    gap_xy = np.maximum(np.abs(a[:2] - b[:2]) - (a[3:5] + b[3:5]) / 2.0, 0.0)
    target_center = a[:3]
    anchor_center = b[:3]
    metrics = {
        "center": float(np.linalg.norm(target_center - anchor_center)),
        "aabb_gap": float(np.linalg.norm(gap_xyz)),
        "surface_gap": float(np.linalg.norm(gap_xyz)),
        "horizontal_center": float(
            np.linalg.norm(target_center[:2] - anchor_center[:2])
        ),
        "horizontal_gap": float(np.linalg.norm(gap_xy)),
        "viewpoint": None,
        "plane": None,
    }
    observer = _viewpoint_origin(viewpoint)
    if observer is not None:
        metrics["viewpoint"] = float(np.linalg.norm(target_center - observer))
    wall = anchor.get("wall_geometry")
    if isinstance(wall, dict) and wall.get("valid"):
        try:
            origin = np.asarray(wall["origin_xy_m"], dtype=float)
            normal = np.asarray(wall["normal_xy"], dtype=float)
            tangent = np.asarray(wall["tangent_xy"], dtype=float)
            extent = np.asarray(wall["tangent_extent_m"], dtype=float)
            tangent_coordinate = float((target_center[:2] - origin) @ tangent)
            if extent[0] <= tangent_coordinate <= extent[1]:
                metrics["plane"] = float(abs((target_center[:2] - origin) @ normal))
        except (KeyError, TypeError, ValueError):
            metrics["plane"] = None
    return metrics


def canonical_category(name):
    text = " ".join(re.findall("[a-z]+", name.lower()))
    aliases = {
        **ALIASES,
        **{
            "chalk board": "blackboard",
            "chalk boards": "blackboard",
            "chalkboard": "blackboard",
            "chalkboards": "blackboard",
        },
    }
    for source in sorted(aliases, key=len, reverse=True):
        text = re.sub("\\b" + re.escape(source) + "\\b", aliases[source], text)
    return text


def match_categories(names, objects):
    if isinstance(names, str):
        names = [names]
    aliases = {
        "sofa": "couch",
        "sofas": "couch",
        "loveseat": "couch",
        "bookcase": "bookshelf",
        "bookcases": "bookshelf",
        "cupboard": "cabinet",
        "cupboards": "cabinet",
        "filing cabinet": "file cabinet",
        "painting": "picture",
        "paintings": "picture",
        "artwork": "picture",
        "picture frame": "picture",
        "picture frames": "picture",
    }

    def words(name):
        text = canonical_category(name)
        return aliases.get(text, text).split()

    categories = [words(n) for n in names if n.strip()]
    matched = []
    for obj in objects:
        labels = [words(name) for name in category_names(obj)]
        if any(
            (
                category
                and words
                and (category[-1] == words[-1])
                and all((w in words for w in category))
                for category in categories
                for words in labels
            )
        ):
            matched.append(obj)
    return matched


def retrieval_category(name):
    """Remove common appearance modifiers only for category retrieval."""
    text = canonical_category(name)
    if text in {"wall", "floor", "ground", "ceiling"}:
        return text
    modifiers = {
        "large",
        "small",
        "big",
        "little",
        "black",
        "white",
        "red",
        "blue",
        "green",
        "gray",
        "grey",
        "brown",
        "wooden",
        "wall",
        "mounted",
    }
    return " ".join((word for word in text.split() if word not in modifiers))


def match_retrieval_categories(names, objects):
    if isinstance(names, str):
        names = [names]
    return match_categories([retrieval_category(name) for name in names], objects)


def superlative_rejection(query, relation):
    kind = relation.get("relation")
    phrase = relation.get("phrase", "")
    pattern = (
        "\\b(nearest|closest)\\b" if kind == "nearest" else "\\b(farthest|furthest)\\b"
    )
    if (
        not isinstance(phrase, str)
        or not phrase.strip()
        or phrase.casefold() not in query.casefold()
        or (not re.search(pattern, phrase, re.I))
    ):
        return "no matching superlative quote in original query"
    if re.search("\\b(no|not|never|without|except)\\b|n['’]t\\b", query, re.I):
        return "negation/exclusion requires further parsing"
    return None


def validate_distance_relations(query, parsed):
    """Suppress unsupported distance operators while retaining raw API calls separately."""
    relations = parsed.get("Relations", [])
    if not isinstance(relations, list) or any(
        (not isinstance(r, dict) for r in relations)
    ):
        raise ValueError("Relations must be a list of objects")
    valid, rejected = ([], [])
    for relation in relations:
        reason = (
            superlative_rejection(query, relation)
            if relation.get("relation") in ("nearest", "farthest")
            else None
        )
        if reason:
            rejected.append({"relation": relation, "reason": reason})
            valid.append({**relation, "relation": "other"})
        else:
            valid.append(relation)
    return {**parsed, "Relations": valid, "rejected_distance_relations": rejected}
