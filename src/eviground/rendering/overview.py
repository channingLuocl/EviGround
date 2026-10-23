"""Initial classifier-covered scene views and final evidence validation."""

import numpy as np
from pathlib import Path
from eviground.rendering.renderer import render
from eviground.categories.retrieval import match_retrieval_categories


def scene_views(parsed, objects, candidates, xyz, rgb, ids, out):
    covered_ids = [o["bbox_id"] for o in objects if "category_hypotheses" in o]
    keep = np.isin(ids, covered_ids)
    shell_ids = []
    points, colors, instance_ids = (xyz[keep], rgb[keep], ids[keep])
    references = match_retrieval_categories(parsed.get("Anchor", []), objects)
    focus = list({o["bbox_id"]: o for o in candidates + references}.values())
    relevant_ids = {o["bbox_id"] for o in candidates + references}
    bounds = np.array([o["bbox_3d"] for o in focus])
    views = []
    angles = ((45, 50), (225, 40), (0, 89))
    for index, (azimuth, elevation) in enumerate(angles):
        region = np.ones(len(points), dtype=bool)
        lo, hi = (points.min(0), points.max(0))
        if index == 1 and len(bounds):
            lo = (bounds[:, :3] - bounds[:, 3:] / 2).min(0)
            hi = (bounds[:, :3] + bounds[:, 3:] / 2).max(0)
            margin = max(float(np.max(hi - lo)) * 0.2, 0.3)
            region = np.all((points >= lo - margin) & (points <= hi + margin), axis=1)
        at = (lo + hi) / 2
        span = max(float(np.linalg.norm(hi - lo)), 0.5)
        a, e = np.deg2rad([azimuth, elevation])
        eye = at + 1.35 * span * np.array(
            [np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)]
        )
        from eviground.grounding.bindings import camera_eye

        eye, at = camera_eye(lo, hi, eye - at)
        labels = np.where(
            np.isin(instance_ids[region], list(relevant_ids)), instance_ids[region], -1
        )
        path = Path(out) / f"scene_{index + 1}.png"
        stats = render(
            points[region],
            colors[region],
            path,
            eye=eye,
            at=at,
            ids=labels,
            size=1000,
            point_radius=1,
        )
        views.append(
            {
                "image": str(path),
                **stats,
                "kind": "room_cutaway" if index == 0 else "target_reference_context",
                "hidden_shell_ids": shell_ids,
            }
        )
        views[-1].update(
            kind=(
                "predicted_instance_overview",
                "target_reference_context",
                "top_down_geometry_context",
            )[index],
            hidden_supplemental_instance_ids=[
                o["bbox_id"] for o in objects if "category_hypotheses" not in o
            ],
        )
    return views


def validate_decision(data, query, allowed_ids, known_ids):
    oid = data.get("predicted_id")
    if oid is not None and (type(oid) is not int or oid not in allowed_ids):
        raise ValueError("Scene decision must select an allowed target ID")
    if not isinstance(data.get("reason"), str) or not data["reason"].strip():
        raise ValueError("Scene decision requires a reason")
    evidence = data.get("evidence", [])
    if not isinstance(evidence, list):
        raise ValueError("Scene decision evidence must be an array")
    for row in evidence:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("source_quote"), str)
            or (not row["source_quote"].strip())
            or (row["source_quote"].casefold() not in query.casefold())
            or (not isinstance(row.get("object_ids"), list))
            or (not row["object_ids"])
            or any(
                (type(i) is not int or i not in known_ids for i in row["object_ids"])
            )
            or (not isinstance(row.get("observation"), str))
            or (not row["observation"].strip())
        ):
            raise ValueError(
                "Scene evidence requires an exact quote, known IDs and an observation"
            )
    if oid is not None and (not any((oid in row["object_ids"] for row in evidence))):
        raise ValueError("Scene choice requires evidence about the selected object")
    return data
