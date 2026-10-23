#!/usr/bin/env python3
"""Run the minimal viewpoint-stability experiment over a curated query manifest.

The manifest contains human-checked target category, anchor instance, and
left/right relation.  It deliberately does not parse natural language or use a
ground-truth target to choose a viewpoint.  A target_id, if supplied, is used
only for the diagnostic fields in the output.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from explore_horizontal_viewpoints import explore, load_objects


REQUIRED_FIELDS = {"query_id", "scene_id", "target_category", "anchor_id", "relation"}


def read_manifest(path):
    records = []
    seen_ids = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"清单第 {line_number} 行不是合法 JSON：{error.msg}") from error
        if not isinstance(record, dict) or not REQUIRED_FIELDS <= record.keys():
            missing = sorted(REQUIRED_FIELDS - set(record) if isinstance(record, dict) else REQUIRED_FIELDS)
            raise ValueError(f"清单第 {line_number} 行缺少字段：{', '.join(missing)}")
        if not isinstance(record["query_id"], str) or not record["query_id"]:
            raise ValueError(f"清单第 {line_number} 行的 query_id 必须是非空字符串")
        if record["query_id"] in seen_ids:
            raise ValueError(f"query_id 重复：{record['query_id']}")
        if not isinstance(record["scene_id"], str) or not record["scene_id"]:
            raise ValueError(f"清单第 {line_number} 行的 scene_id 必须是非空字符串")
        if not isinstance(record["target_category"], str) or not record["target_category"]:
            raise ValueError(f"清单第 {line_number} 行的 target_category 必须是非空字符串")
        if not isinstance(record["anchor_id"], int):
            raise ValueError(f"清单第 {line_number} 行的 anchor_id 必须是整数")
        if record["relation"] not in {"left", "right"}:
            raise ValueError(f"清单第 {line_number} 行的 relation 必须是 left 或 right")
        if "target_id" in record and not isinstance(record["target_id"], int):
            raise ValueError(f"清单第 {line_number} 行的 target_id 必须是整数")
        if "allowed_headings_deg" in record:
            headings = record["allowed_headings_deg"]
            if not isinstance(headings, list) or not headings or any(
                not isinstance(heading, int) or not 0 <= heading < 360 for heading in headings
            ):
                raise ValueError(f"清单第 {line_number} 行的 allowed_headings_deg 必须是非空的 [0, 359] 整数列表")
        seen_ids.add(record["query_id"])
        records.append(record)
    if not records:
        raise ValueError("查询清单为空")
    return records


def object_path(record, objects_dir, objects_suffix):
    relative_path = record.get("objects_path", f"{record['scene_id']}{objects_suffix}")
    path = objects_dir / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"查询 {record['query_id']} 找不到物体表：{path}")
    return path


def run_record(record, objects_dir, objects_suffix, default_headings):
    headings = record.get("allowed_headings_deg", default_headings)
    base = {
        "query_id": record["query_id"],
        "scene_id": record["scene_id"],
        "utterance": record.get("utterance"),
        "input": {
            "target_category": record["target_category"],
            "anchor_id": record["anchor_id"],
            "relation": record["relation"],
            "headings_deg": headings,
        },
    }
    try:
        result = explore(
            load_objects(object_path(record, objects_dir, objects_suffix)),
            record["target_category"],
            record["anchor_id"],
            record["relation"],
            headings,
        )
    except ValueError as error:
        base.update({"status": "ineligible", "reason": str(error)})
        return base

    base.update(result)
    base["status"] = "stable" if result["is_stable"] else "conflict"
    if "target_id" in record:
        target_id = record["target_id"]
        wins = next((item["wins"] for item in result["candidate_support"] if item["candidate_id"] == target_id), 0)
        base["evaluation_only"] = {
            "target_id": target_id,
            "target_is_candidate": target_id in result["candidate_ids"],
            "target_wins": wins,
            "target_support_fraction": round(wins / len(headings), 6),
            "stable_winner_matches_target": result["is_stable"] and result["per_heading"][0]["winner_id"] == target_id,
        }
    return base


def summarize(results):
    status_counts = Counter(result["status"] for result in results)
    eligible = [result for result in results if result["status"] in {"stable", "conflict"}]
    diagnostics = [result["evaluation_only"] for result in eligible if "evaluation_only" in result]
    summary = {
        "total_queries": len(results),
        "status_counts": dict(sorted(status_counts.items())),
        "eligible_queries": len(eligible),
        "conflict_fraction_among_eligible": round(status_counts["conflict"] / len(eligible), 6) if eligible else None,
    }
    if diagnostics:
        summary["evaluation_only"] = {
            "queries_with_target_id": len(diagnostics),
            "stable_winner_matches_target_fraction": round(
                sum(item["stable_winner_matches_target"] for item in diagnostics) / len(diagnostics), 6
            ),
            "target_is_candidate_fraction": round(
                sum(item["target_is_candidate"] for item in diagnostics) / len(diagnostics), 6
            ),
        }
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="批量枚举水平参照方向，测量 left/right 查询的候选稳定性")
    parser.add_argument("--manifest", type=Path, required=True, help="人工核验的 JSONL 查询清单")
    parser.add_argument("--objects-dir", type=Path, required=True, help="物体表根目录")
    parser.add_argument("--objects-suffix", default=".json", help="未给 objects_path 时，scene_id 后接的文件后缀")
    parser.add_argument("--step-deg", type=int, default=10, help="默认水平角度采样间隔，必须整除 360")
    parser.add_argument("--output-dir", type=Path, required=True, help="必须不存在的实验输出目录")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 1 <= args.step_deg <= 180 or 360 % args.step_deg:
        raise SystemExit("step-deg 必须在 1 到 180 之间且整除 360")
    if args.output_dir.exists():
        raise SystemExit("输出目录已存在；请指定新路径，避免覆盖既有实验")
    records = read_manifest(args.manifest)
    headings = list(range(0, 360, args.step_deg))
    results = [run_record(record, args.objects_dir, args.objects_suffix, headings) for record in records]
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "per_query.jsonl").write_text(
        "".join(json.dumps(result, ensure_ascii=False) + "\n" for result in results), encoding="utf-8"
    )
    summary = summarize(results)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
