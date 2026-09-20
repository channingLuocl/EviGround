#!/usr/bin/env python3
"""Enumerate horizontal headings and expose viewpoint-sensitive candidates.

This is a geometry-only diagnostic for the first research question.  It never
uses ground-truth target IDs and does not infer a speaker's true pose.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def normalized_label(label):
    return " ".join(str(label).lower().replace("_", " ").split())


def load_objects(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError("objects JSON 必须是物体列表")
    objects = []
    ids = set()
    for item in data:
        if not isinstance(item, dict) or not {"bbox_id", "target", "bbox_3d"} <= item.keys():
            raise ValueError("每个物体必须包含 bbox_id、target、bbox_3d")
        object_id = item["bbox_id"]
        center = item["bbox_3d"][:3]
        if not isinstance(object_id, int) or object_id in ids:
            raise ValueError("bbox_id 必须是唯一整数")
        if not isinstance(center, list) or len(center) != 3 or not all(isinstance(x, (int, float)) for x in center):
            raise ValueError(f"物体 {object_id} 的 bbox_3d 前三项必须是坐标")
        ids.add(object_id)
        objects.append({"id": object_id, "label": normalized_label(item["target"]), "center": center})
    return objects


def right_vector(heading_deg):
    """Camera forward is (cos(theta), sin(theta)); right is its clockwise normal."""
    theta = math.radians(heading_deg)
    return (math.sin(theta), -math.cos(theta))


def projection(candidate_center, anchor_center, heading_deg):
    right_x, right_y = right_vector(heading_deg)
    return (candidate_center[0] - anchor_center[0]) * right_x + (candidate_center[1] - anchor_center[1]) * right_y


def choose_candidate(candidates, anchor, relation, heading_deg):
    scored = [
        {"id": candidate["id"], "projection": projection(candidate["center"], anchor["center"], heading_deg)}
        for candidate in candidates
    ]
    if relation == "left":
        return min(scored, key=lambda item: (item["projection"], item["id"]))
    return max(scored, key=lambda item: (item["projection"], -item["id"]))


def explore(objects, target_category, anchor_id, relation, headings):
    headings = list(headings)
    if not headings:
        raise ValueError("至少需要一个水平观察方向")
    if any(not isinstance(heading, int) or not 0 <= heading < 360 for heading in headings):
        raise ValueError("观察方向必须是 0 到 359 的整数角度")
    anchor = next((obj for obj in objects if obj["id"] == anchor_id), None)
    if anchor is None:
        raise ValueError(f"找不到锚点 ID {anchor_id}")
    target = normalized_label(target_category)
    candidates = [obj for obj in objects if obj["label"] == target and obj["id"] != anchor_id]
    if len(candidates) < 2:
        raise ValueError("需要至少两个非锚点的同类候选，才能检查视角敏感性")
    per_heading = []
    for heading in headings:
        winner = choose_candidate(candidates, anchor, relation, heading)
        per_heading.append({"heading_deg": heading, "winner_id": winner["id"], "winner_projection": round(winner["projection"], 6)})
    counts = Counter(item["winner_id"] for item in per_heading)
    support = [
        {"candidate_id": candidate["id"], "wins": counts[candidate["id"]], "fraction": round(counts[candidate["id"]] / len(per_heading), 6)}
        for candidate in sorted(candidates, key=lambda item: item["id"])
    ]
    return {
        "assumptions": {
            "geometry": "object centers projected onto the observer right axis in the world XY plane",
            "heading": "0 degrees faces +X; 90 degrees faces +Y",
            "excluded": ["camera position", "visibility", "occlusion", "object orientation", "full language parsing"],
        },
        "query_template": {"target_category": target, "anchor_id": anchor_id, "anchor_label": anchor["label"], "relation": relation},
        "candidate_ids": [obj["id"] for obj in sorted(candidates, key=lambda item: item["id"])],
        "per_heading": per_heading,
        "candidate_support": support,
        "distinct_winners": len(counts),
        "is_stable": len(counts) == 1,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="枚举水平观察方向，检查 left/right 关系的候选稳定性")
    parser.add_argument("--objects", type=Path, required=True, help="物体表 JSON")
    parser.add_argument("--target-category", required=True, help="候选目标类别，例如 window")
    parser.add_argument("--anchor-id", type=int, required=True, help="已确定的锚点实例 ID")
    parser.add_argument("--relation", choices=["left", "right"], required=True)
    parser.add_argument("--step-deg", type=int, default=10, help="水平角度采样间隔，必须整除 360")
    parser.add_argument("--output", type=Path, required=True, help="新建的结果 JSON 路径")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 1 <= args.step_deg <= 180 or 360 % args.step_deg:
        raise SystemExit("step-deg 必须在 1 到 180 之间且整除 360")
    if args.output.exists():
        raise SystemExit("输出文件已存在；请指定新路径，避免覆盖既有实验")
    result = explore(load_objects(args.objects), args.target_category, args.anchor_id, args.relation, range(0, 360, args.step_deg))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ["query_template", "distinct_winners", "is_stable", "candidate_support"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
