"""Frozen predictions become category hypotheses, never annotations."""

import hashlib
import json
from pathlib import Path
import numpy as np

VERSION = "eviground-category-hypotheses-v1"
SCORE_TYPE = "weighted_standardized_3d_rgb_fusion_not_probability"
CATEGORY_INSTRUCTIONS = " Category labels are uncertain, mutually exclusive hypotheses per object ID. category is the frozen top-1 display label only; use category_hypotheses for eligible bindings. Read category_score_type: scores are standardized 3D/RGB fusion scores; they are not probabilities. Rank 1 is the classifier primary hypothesis; lower ranks are alternatives, not equally confident labels. Use rank and gap_from_top1 as classifier evidence, not proof or a calibrated confidence threshold. A lower-ranked category can win when the complete description and visual/spatial evidence support it. Match target and reference categories to retained hypotheses, then verify with visual/spatial evidence. Do not assign incompatible categories to the same object in one interpretation. A retained category is not proof; a rejected category is not proof of physical absence. Return uncertainty when essential category evidence is missing."


def category_names(obj):
    if "category_hypotheses" in obj:
        return [h["category"] for h in obj["category_hypotheses"]]
    return [obj["target"]]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gate_categories(top5, top_k=5, score_type=SCORE_TYPE):
    if score_type != SCORE_TYPE or top_k not in (1, 5):
        raise ValueError("Expected fusion scores and top_k=1|5")
    if (
        len(top5) != 5
        or len({r["category"] for r in top5}) != 5
        or any(
            (
                not isinstance(r["category"], str) or not r["category"].strip()
                for r in top5
            )
        )
    ):
        raise ValueError("Expected five distinct category predictions")
    scores = np.asarray([r["score"] for r in top5], dtype=float)
    if not np.isfinite(scores).all() or np.any(scores[:-1] < scores[1:]):
        raise ValueError("Category scores must be finite and sorted descending")
    if any(("rank" in row and row["rank"] != rank for rank, row in enumerate(top5, 1))):
        raise ValueError("Category ranks disagree with score order")
    accepted, audit = ([], [])
    for rank, row in enumerate(top5, 1):
        gap = float(scores[0] - row["score"])
        reasons = []
        if rank > top_k:
            reasons.append("outside_top_k")
        item = {
            "category": row["category"],
            "score": float(row["score"]),
            "rank": rank,
            "gap_from_top1": gap,
        }
        if not reasons:
            accepted.append(item)
        audit.append({**item, "retained": not reasons, "reasons": reasons})
    return (accepted, audit)


def load_category_objects(path):
    data = json.loads(Path(path).read_text())
    if data.get("version") != VERSION:
        raise ValueError("Expected a prepared EviGround category table")
    score_type = data["gate"].get("score_type", SCORE_TYPE)
    if score_type != SCORE_TYPE:
        raise ValueError("Category table version disagrees with score type")
    result = []
    for item in data["objects"]:
        box = np.asarray(item["bbox_3d"], dtype=float)
        if (
            type(item["bbox_id"]) is not int
            or box.shape != (6,)
            or (not np.isfinite(box).all())
            or np.any(box[3:] <= 0)
        ):
            raise ValueError("Invalid predicted instance geometry")
        retained, audit = gate_categories(item["top5"], **data["gate"])
        if (
            retained != item["category_hypotheses"]
            or audit != item["category_gate_audit"]
        ):
            raise ValueError("Category table disagrees with its recorded gate")
        result.append(
            {
                key: item[key]
                for key in (
                    "bbox_id",
                    "target",
                    "bbox_3d",
                    "category_hypotheses",
                    "category_status",
                )
            }
        )
        result[-1]["category_score_type"] = score_type
        expected_status = "single_hypothesis" if len(retained) == 1 else "uncertain"
        if item["category_status"] != expected_status:
            raise ValueError("Category status disagrees with retained hypotheses")
        if item["target"] != item["top5"][0]["category"]:
            raise ValueError("Display label differs from frozen top-1")
    if not result or len({r["bbox_id"] for r in result}) != len(result):
        raise ValueError("Empty or duplicate category table")
    return result


def prepare_tables(source, scene_root, output, scenes, top_k=5):
    from eviground.categories.export import prepare_tables as prepare_export

    return prepare_export(Path(source), Path(scene_root), Path(output), scenes, top_k)
