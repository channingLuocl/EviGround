#!/usr/bin/env python3
"""Create a human-review sheet for the first Nr3D left/right experiment."""

import argparse
import csv
import json
import random
import re
from pathlib import Path

from explore_horizontal_viewpoints import load_objects, normalized_label


RELATION_PATTERN = re.compile(r"\b(left|right)\b", re.IGNORECASE)
REVIEW_FIELDS = [
    "query_id",
    "scene_id",
    "utterance",
    "target_category",
    "target_id",
    "relation_hint",
    "candidate_ids",
    "objects_path",
    "anchor_id",
    "relation",
    "allowed_headings_deg",
    "review_status",
    "notes",
]


def is_true(value):
    return str(value).strip().lower() == "true"


def relation_hint(utterance):
    relations = set(RELATION_PATTERN.findall(utterance))
    return relations.pop() if len(relations) == 1 else None


def eligible_rows(nr3d_csv, objects_dir):
    eligible = []
    cache = {}
    with Path(nr3d_csv).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if not (is_true(row["correct_guess"]) and is_true(row["mentions_target_class"])):
                continue
            hint = relation_hint(row["utterance"])
            if hint is None:
                continue
            scene_path = objects_dir / f"{row['scan_id']}.json"
            if not scene_path.is_file():
                continue
            if scene_path not in cache:
                cache[scene_path] = load_objects(scene_path)
            target_category = normalized_label(row["instance_type"])
            candidates = [obj["id"] for obj in cache[scene_path] if obj["label"] == target_category]
            target_id = int(row["target_id"])
            if len(candidates) < 2 or target_id not in candidates:
                continue
            eligible.append(
                {
                    "query_id": row["assignmentid"],
                    "scene_id": row["scan_id"],
                    "utterance": row["utterance"],
                    "target_category": target_category,
                    "target_id": target_id,
                    "relation_hint": hint,
                    "candidate_ids": json.dumps(candidates),
                    "objects_path": f"{row['scan_id']}.json",
                    "anchor_id": "",
                    "relation": "",
                    "allowed_headings_deg": "",
                    "review_status": "pending",
                    "notes": "",
                }
            )
    return eligible


def parse_args():
    parser = argparse.ArgumentParser(description="从 Nr3D CSV 生成 left/right 查询的人工审核表")
    parser.add_argument("--nr3d-csv", type=Path, required=True)
    parser.add_argument("--objects-dir", type=Path, required=True, help="Nr3D gt 或 pred 对象表目录")
    parser.add_argument("--limit", type=int, default=200, help="固定随机抽样数量")
    parser.add_argument("--seed", type=int, default=20260920, help="抽样随机种子")
    parser.add_argument("--output", type=Path, required=True, help="必须不存在的审核 CSV 输出路径")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.limit < 1:
        raise SystemExit("limit 必须至少为 1")
    if args.output.exists():
        raise SystemExit("输出文件已存在；请指定新路径，避免覆盖既有审核记录")
    rows = eligible_rows(args.nr3d_csv, args.objects_dir)
    if not rows:
        raise SystemExit("没有可审核查询；请检查 CSV 与对象表是否匹配")
    rows.sort(key=lambda row: row["query_id"])
    random.Random(args.seed).shuffle(rows)
    selected = rows[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(selected)
    print(json.dumps({"eligible_queries": len(rows), "selected_queries": len(selected), "seed": args.seed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
