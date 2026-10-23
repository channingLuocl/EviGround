#!/usr/bin/env python3
"""Convert approved rows in a manually reviewed CSV into the experiment JSONL."""

import argparse
import csv
import json
from pathlib import Path


def headings(value, line_number):
    if not value.strip():
        return None
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError(f"审核表第 {line_number} 行的 allowed_headings_deg 必须是逗号分隔的整数") from error
    if not result or any(not 0 <= item < 360 for item in result):
        raise ValueError(f"审核表第 {line_number} 行的 allowed_headings_deg 必须在 0 到 359 之间")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="将人工审核通过的 Nr3D 查询转换为实验 JSONL 清单")
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="必须不存在的 JSONL 输出路径")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists():
        raise SystemExit("输出文件已存在；请指定新路径，避免覆盖既有实验清单")
    manifest = []
    with args.review_csv.open(encoding="utf-8", newline="") as stream:
        for line_number, row in enumerate(csv.DictReader(stream), 2):
            if row.get("review_status", "").strip().lower() != "approved":
                continue
            if row.get("relation", "").strip() not in {"left", "right"}:
                raise ValueError(f"审核表第 {line_number} 行已批准，但 relation 不是 left/right")
            try:
                anchor_id = int(row["anchor_id"])
                target_id = int(row["target_id"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"审核表第 {line_number} 行已批准，但 anchor_id 或 target_id 无效") from error
            record = {
                "query_id": row["query_id"],
                "scene_id": row["scene_id"],
                "objects_path": row["objects_path"],
                "utterance": row["utterance"],
                "target_category": row["target_category"],
                "anchor_id": anchor_id,
                "relation": row["relation"].strip(),
                "target_id": target_id,
            }
            allowed = headings(row.get("allowed_headings_deg", ""), line_number)
            if allowed is not None:
                record["allowed_headings_deg"] = allowed
            manifest.append(record)
    if not manifest:
        raise SystemExit("没有 review_status=approved 的查询")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in manifest), encoding="utf-8")
    print(json.dumps({"approved_queries": len(manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
