"""Frozen predictions become category hypotheses, never annotations."""

import hashlib
import json
from pathlib import Path
import numpy as np

VERSION = "eviground-category-hypotheses-v1"
SCORE_TYPE = "weighted_standardized_3d_rgb_fusion_not_probability"
MASK3D_SCORE_TYPE = "mask3d_foreground_softmax_probability"
SEEGROUND_SCORE_TYPE = "frozen_pointnext_clip_similarity_times_10_not_probability"
NATIVE_SCORE_TYPES = {
    "seeground_mask3d_scannet200": MASK3D_SCORE_TYPE,
    "seeground_pointnext_clip_reproduction": SEEGROUND_SCORE_TYPE,
}
CATEGORY_INSTRUCTIONS = " Category labels are uncertain, mutually exclusive hypotheses per object ID. category is the frozen top-1 display label only; use category_hypotheses for eligible bindings. Read category_source and category_score_type for the original score meaning; scores from different frontends are not comparable or calibrated evidence of physical truth. Rank 1 is the classifier primary hypothesis; lower ranks are alternatives, not equally confident labels. Use rank and gap_from_top1 as classifier evidence, not proof or a calibrated confidence threshold. A lower-ranked category can win when the complete description and visual/spatial evidence support it. Match target and reference categories to retained hypotheses, then verify with visual/spatial evidence. Do not assign incompatible categories to the same object in one interpretation. A retained category is not proof; a rejected category is not proof of physical absence. Return uncertainty when essential category evidence is missing."


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
    if score_type not in (SCORE_TYPE, *NATIVE_SCORE_TYPES.values()) or top_k not in (1, 5):
        raise ValueError("Expected a supported original score type and top_k=1|5")
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
    if score_type == MASK3D_SCORE_TYPE and (np.any(scores < 0) or np.any(scores > 1)
                                           or scores.sum() > 1 + 1e-6):
        raise ValueError("Invalid original Mask3D foreground probabilities")
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
    if score_type not in (SCORE_TYPE, *NATIVE_SCORE_TYPES.values()):
        raise ValueError("Category table version disagrees with score type")
    if score_type != SCORE_TYPE and NATIVE_SCORE_TYPES.get(data.get("category_source")) != score_type:
        raise ValueError("Native scores need their explicit corresponding prediction source")
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
        if "category_source" in data:
            result[-1]["category_source"] = data["category_source"]
        if "proposal_detection_score" in item:
            score = item["proposal_detection_score"]
            if isinstance(score, bool) or not isinstance(score, (float, int)) or not 0 <= score <= 1:
                raise ValueError("Invalid proposal detection score")
            result[-1]["proposal_detection_score"] = float(score)
        expected_status = "single_hypothesis" if len(retained) == 1 else "uncertain"
        if item["category_status"] != expected_status:
            raise ValueError("Category status disagrees with retained hypotheses")
        if item["target"] != item["top5"][0]["category"]:
            raise ValueError("Display label differs from frozen top-1")
    if not result or len({r["bbox_id"] for r in result}) != len(result):
        raise ValueError("Empty or duplicate category table")
    return result


def prepare_tables(source, scene_root, output, scenes, top_k=5, proposal_policy=None):
    from eviground.categories.export import prepare_tables as prepare_export

    return prepare_export(Path(source), Path(scene_root), Path(output), scenes, top_k,
                          proposal_policy=proposal_policy)
