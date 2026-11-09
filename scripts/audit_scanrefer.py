"""Offline ScanRefer diagnostics. GT is used for evaluation only, never inference."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


def pairwise_iou(first, second):
    first = np.asarray(first, dtype=float).reshape(-1, 6)
    second = np.asarray(second, dtype=float).reshape(-1, 6)
    low = np.maximum(first[:, None, :3] - first[:, None, 3:] / 2,
                     second[None, :, :3] - second[None, :, 3:] / 2)
    high = np.minimum(first[:, None, :3] + first[:, None, 3:] / 2,
                      second[None, :, :3] + second[None, :, 3:] / 2)
    intersection = np.maximum(high - low, 0).prod(-1)
    union = first[:, 3:].prod(-1)[:, None] + second[:, 3:].prod(-1)[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def audit(root, run):
    manifest = json.loads((root / "scanrefer_val_manifest.json").read_text())
    rows = manifest["rows"]
    scenes, proposal_oracle, records = {}, {}, []
    for scene in sorted({r["scene"] for r in rows}):
        directory = root / "scanrefer_mask3d_prepared_scenes" / scene
        proposals = json.loads((directory / "scene_instances.json").read_text())
        gt = json.loads((directory / "objects_gt.json").read_text())
        overlaps = pairwise_iou([o["bbox_3d"] for o in gt],
                                [o["bbox_3d"] for o in proposals])
        scenes[scene] = (proposals, {o["bbox_id"]: i for i, o in enumerate(gt)}, overlaps)
        for i, obj in enumerate(gt):
            proposal_oracle[scene, obj["bbox_id"]] = float(overlaps[i].max(initial=0))
    for row in rows:
        case = run / f"q{row['csv_row']:05d}"
        evaluation = case / "evaluation.json"
        if not evaluation.exists():
            continue
        value = json.loads(evaluation.read_text())
        assert all(value[k] == row[k] for k in ("scene", "query", "target_id"))
        proposals, gt_indices, overlaps = scenes[row["scene"]]
        target_overlaps = overlaps[gt_indices[row["target_id"]]]
        by_id = {o["bbox_id"]: i for i, o in enumerate(proposals)}
        prediction = json.loads((case / "prediction.json").read_text()) if (case / "prediction.json").exists() else {}
        selected = prediction.get("predicted_id")
        candidates = prediction.get("category_target_ids", [])
        candidate_oracle = max((float(target_overlaps[by_id[i]]) for i in candidates), default=0)
        iou = float(target_overlaps[by_id[selected]]) if selected is not None and not value.get("error") else 0
        records.append({**row, "error": value.get("error"), "predicted_id": selected,
                        "reported_iou": value.get("iou_3d"), "recomputed_iou": iou,
                        "legacy_id_equal": value.get("correct", False),
                        "proposal_oracle_iou": float(target_overlaps.max(initial=0)),
                        "category_oracle_iou": candidate_oracle, "candidate_count": len(candidates),
                        "best_proposal_id": proposals[int(target_overlaps.argmax())]["bbox_id"],
                        "category_ids_with_iou25": [i for i in candidates if target_overlaps[by_id[i]] >= .25]})

    def summarize(items):
        count = len(items)
        return {"queries": count, "acc25": sum(r["recomputed_iou"] >= .25 for r in items) / count if count else None,
                "acc50": sum(r["recomputed_iou"] >= .5 for r in items) / count if count else None,
                "proposal_oracle25": sum(r["proposal_oracle_iou"] >= .25 for r in items) / count if count else None,
                "category_oracle25": sum(r["category_oracle_iou"] >= .25 for r in items) / count if count else None}

    finished = [r for r in records if not r["error"]]
    coverage = [proposal_oracle[r["scene"], r["target_id"]] for r in rows]
    return {"scope": "Offline diagnosis only; oracle scores are upper bounds, not model accuracy",
            "selected_queries": len(rows), "saved_evaluations": len(records),
            "finished_without_error": summarize(finished), "including_interrupted_errors": summarize(records),
            "full_proposal_oracle": {"queries": len(rows), "acc25": float(np.mean(np.asarray(coverage) >= .25)),
                                     "acc50": float(np.mean(np.asarray(coverage) >= .5))},
            "observed_scene_counts": dict(Counter(r["scene"] for r in finished)),
            "iou_mismatches": sum(not np.isclose(r["reported_iou"], r["recomputed_iou"]) for r in finished),
            "failures_at25": {"missing_proposal": sum(r["proposal_oracle_iou"] < .25 for r in finished),
                              "lost_in_category_retrieval": sum(r["proposal_oracle_iou"] >= .25 and r["category_oracle_iou"] < .25 for r in finished),
                              "wrong_react_choice": sum(r["category_oracle_iou"] >= .25 and r["recomputed_iou"] < .25 for r in finished)},
            "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.data_root, args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
