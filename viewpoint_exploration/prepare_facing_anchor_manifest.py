#!/usr/bin/env python3
"""Build an unseen, automatically grounded facing-anchor Nr3D manifest.

Only exact anchor-label mentions that resolve to one scene instance are kept.
Ground-truth target IDs are retained for evaluation and never used to select an
anchor, direction, or sample.
"""

import argparse
import csv
import json
import random
import re
from pathlib import Path

from explore_horizontal_viewpoints import normalized_label


VIEW_CUE = re.compile(r"\b(?:facing|looking at|looking towards|look at)\b")
RELATION = re.compile(r"\b(left|right)\b")


def read_objects(path):
    objects = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(objects, list):
        raise ValueError("物体表必须是列表")
    return [{"id": item["bbox_id"], "label": normalized_label(item["target"])} for item in objects]


def infer_anchor(utterance, objects):
    """Return an exact, unique anchor ID mentioned after a facing cue."""
    text = normalized_label(utterance)
    cue = VIEW_CUE.search(text)
    if cue is None:
        return None
    tail = text[cue.end():]
    labels = {}
    for obj in objects:
        labels.setdefault(obj["label"], []).append(obj["id"])
    matches = []
    for label, ids in labels.items():
        if len(ids) != 1:
            continue
        if re.search(r"\b(?:the )?" + re.escape(label) + r"\b", tail):
            matches.append((label, ids[0]))
    return matches[0][1] if len(matches) == 1 else None


def eligible_rows(csv_path, objects_dir):
    rows = []
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("correct_guess", "").lower() != "true" or row.get("mentions_target_class", "").lower() != "true":
                continue
            relation_words = RELATION.findall(normalized_label(row["utterance"]))
            if len(set(relation_words)) != 1:
                continue
            scene_id = row["scan_id"]
            object_path = objects_dir / f"{scene_id}.json"
            if not object_path.is_file():
                continue
            objects = read_objects(object_path)
            try:
                target_id = int(row["target_id"])
            except ValueError:
                continue
            target_category = normalized_label(row["instance_type"])
            candidate_ids = [obj["id"] for obj in objects if obj["label"] == target_category]
            if len(candidate_ids) < 2 or target_id not in candidate_ids:
                continue
            anchor_id = infer_anchor(row["utterance"], objects)
            if anchor_id is None or anchor_id == target_id:
                continue
            rows.append({
                "query_id": row["assignmentid"], "scene_id": scene_id, "objects_path": f"{scene_id}.json",
                "utterance": row["utterance"], "target_category": target_category, "anchor_id": anchor_id,
                "relation": relation_words[0], "target_id": target_id, "viewpoint_constraint": "facing_anchor",
            })
    return rows


def excluded_query_ids(path):
    if path is None:
        return set()
    return {
        json.loads(line)["query_id"]
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def main():
    parser = argparse.ArgumentParser(description="自动构建严格的 Nr3D facing-anchor 清单")
    parser.add_argument("--nr3d-csv", type=Path, required=True)
    parser.add_argument("--objects-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--exclude-manifest", type=Path, help="排除已查看样本的 JSONL 清单")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("输出文件已存在；请使用新路径")
    if args.limit < 1:
        raise SystemExit("limit 必须大于 0")
    excluded = excluded_query_ids(args.exclude_manifest)
    rows = [row for row in eligible_rows(args.nr3d_csv, args.objects_dir) if row["query_id"] not in excluded]
    random.Random(args.seed).shuffle(rows)
    selected = rows[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    print(json.dumps({"eligible_queries": len(rows), "selected_queries": len(selected), "excluded_queries": len(excluded), "seed": args.seed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
