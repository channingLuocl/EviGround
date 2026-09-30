#!/usr/bin/env python3
"""Infer canonical facing sides from mesh-supported free space.

This diagnostic deliberately chooses anchor-relative *directions*, rather than
arbitrary observer positions.  A direction is retained only when the strip of
floor immediately outside that anchor side contains enough unblocked cells.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from explore_horizontal_viewpoints import choose_candidate, load_objects, normalized_label
from run_facing_anchor_pose_experiment import aligned_vertices, floor_cells, is_blocked, load_obstacles, mesh_vertices


SIDES = (("+x", 1, 0), ("-x", -1, 0), ("+y", 0, 1), ("-y", 0, -1))
WALL_ANCHOR_WORDS = {"door", "window", "whiteboard", "blackboard", "mirror"}


def read_manifest(path):
    records = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        required = {"query_id", "scene_id", "target_category", "anchor_id", "relation", "viewpoint_constraint"}
        if not isinstance(record, dict) or not required <= record.keys():
            raise ValueError(f"清单第 {number} 行缺少必填字段")
        if record["viewpoint_constraint"] != "facing_anchor" or record["relation"] not in {"left", "right"}:
            raise ValueError(f"清单第 {number} 行不是有效的 facing_anchor left/right 查询")
        records.append(record)
    if not records:
        raise ValueError("清单为空")
    return records


def anchor_kind(label):
    words = set(normalized_label(label).split())
    return "wall" if words & WALL_ANCHOR_WORDS else "furniture"


def side_free_cells(anchor, cells, obstacles, cell_size, body_radius, strip_depth, strip_half_width):
    """Count mesh-supported, collision-free floor cells in each anchor-side strip."""
    center_x, center_y = anchor["center"][:2]
    size_x, size_y = anchor["size"][:2]
    scores = []
    for name, normal_x, normal_y in SIDES:
        normal_extent = size_x / 2 if normal_x else size_y / 2
        tangent_extent = size_y / 2 if normal_x else size_x / 2
        retained = []
        for cell_x, cell_y in cells:
            x, y = cell_x * cell_size, cell_y * cell_size
            dx, dy = x - center_x, y - center_y
            forward = dx * normal_x + dy * normal_y
            lateral = abs(-dx * normal_y + dy * normal_x)
            if normal_extent + body_radius <= forward <= normal_extent + strip_depth and lateral <= tangent_extent + strip_half_width:
                if not is_blocked(x, y, obstacles, body_radius):
                    retained.append((x, y))
        scores.append({"side": name, "normal_xy": [normal_x, normal_y], "free_cell_count": len(retained), "cells": retained})
    return scores


def choose_sides(scores, tie_ratio, min_free_cells):
    best = max(score["free_cell_count"] for score in scores)
    if best < min_free_cells:
        raise ValueError(f"锚点四侧均没有至少 {min_free_cells} 个 mesh 支持空闲地面格点")
    return [score for score in scores if score["free_cell_count"] >= best * tie_ratio]


def load_scene(record, objects_dir, mesh_dir, cell_size, floor_band):
    object_path = objects_dir / record.get("objects_path", f"{record['scene_id']}.json")
    scene_dir = mesh_dir / "scans" / record["scene_id"]
    mesh_path = scene_dir / f"{record['scene_id']}_vh_clean_2.ply"
    metadata_path = scene_dir / f"{record['scene_id']}.txt"
    objects = load_objects(object_path)
    raw = json.loads(object_path.read_text())
    raw_by_id = {item["bbox_id"]: item for item in raw}
    anchor = next((obj for obj in objects if obj["id"] == record["anchor_id"]), None)
    if anchor is None or anchor["id"] not in raw_by_id:
        raise ValueError("找不到锚点")
    bbox = raw_by_id[anchor["id"]]["bbox_3d"]
    if len(bbox) < 6:
        raise ValueError("锚点缺少六维 bbox_3d")
    anchor["size"] = [abs(bbox[3]), abs(bbox[4])]
    candidates = [obj for obj in objects if obj["id"] != anchor["id"] and obj["label"] == normalized_label(record["target_category"])]
    if len(candidates) < 2:
        raise ValueError("少于两个同类非锚点候选")
    vertices = aligned_vertices(mesh_vertices(mesh_path), metadata_path)
    cells, floor_z = floor_cells(vertices, cell_size, floor_band)
    return anchor, candidates, cells, floor_z, load_obstacles(object_path)


def run_record(record, objects_dir, mesh_dir, cell_size, floor_band, body_radius, strip_depth, strip_half_width, wall_strip_depth, wall_strip_half_width, tie_ratio, min_free_cells):
    anchor, candidates, cells, floor_z, obstacles = load_scene(record, objects_dir, mesh_dir, cell_size, floor_band)
    kind = anchor_kind(anchor["label"])
    depth, half_width = (wall_strip_depth, wall_strip_half_width) if kind == "wall" else (strip_depth, strip_half_width)
    scores = side_free_cells(anchor, cells, obstacles, cell_size, body_radius, depth, half_width)
    selected = choose_sides(scores, tie_ratio, min_free_cells)
    directions = []
    for side in selected:
        normal_x, normal_y = side["normal_xy"]
        heading = math.degrees(math.atan2(-normal_y, -normal_x)) % 360
        winner = choose_candidate(candidates, anchor, record["relation"], heading)
        directions.append({"side": side["side"], "observer_normal_xy": side["normal_xy"], "heading_deg": round(heading, 3), "winner_id": winner["id"], "free_cell_count": side["free_cell_count"]})
    winners = Counter(item["winner_id"] for item in directions)
    result = {
        "query_id": record["query_id"], "scene_id": record["scene_id"], "utterance": record.get("utterance"),
        "status": "stable" if len(winners) == 1 else "conflict",
        "assumptions": {"canonical_view": "one camera heading per retained anchor side", "anchor_kind": kind, "side_inference": "mesh-supported, bbox-unblocked floor cells in a local anchor-side strip; wall anchors use a wider room-side strip", "excluded": ["true speaker pose", "visibility", "object orientation", "explicit wall-plane segmentation", "room connectivity"]},
        "input": {"target_category": record["target_category"], "anchor_id": record["anchor_id"], "relation": record["relation"]},
        "floor_z": round(floor_z, 3), "side_scores": [{key: value for key, value in score.items() if key != "cells"} for score in scores], "retained_directions": directions,
    }
    if "target_id" in record:
        result["evaluation_only"] = {"target_id": record["target_id"], "winner_matches_target": all(item["winner_id"] == record["target_id"] for item in directions)}
    return result


def main():
    parser = argparse.ArgumentParser(description="用 anchor 周围空闲地面推断规范 facing 方向")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--objects-dir", type=Path, required=True)
    parser.add_argument("--mesh-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cell-size", type=float, default=0.25)
    parser.add_argument("--floor-band", type=float, default=0.08)
    parser.add_argument("--body-radius", type=float, default=0.30)
    parser.add_argument("--strip-depth", type=float, default=1.50)
    parser.add_argument("--strip-half-width", type=float, default=0.50)
    parser.add_argument("--wall-strip-depth", type=float, default=3.0)
    parser.add_argument("--wall-strip-half-width", type=float, default=2.5)
    parser.add_argument("--tie-ratio", type=float, default=0.80, help="保留得分至少为最高分该比例的方向")
    parser.add_argument("--min-free-cells", type=int, default=3, help="最高分方向至少需要的空闲地面格点数")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("输出目录已存在；请指定新路径")
    if min(args.cell_size, args.floor_band, args.body_radius, args.strip_depth, args.strip_half_width, args.wall_strip_depth, args.wall_strip_half_width) <= 0 or not 0 < args.tie_ratio <= 1 or args.min_free_cells < 1:
        raise SystemExit("几何参数必须为正，tie-ratio 必须在 (0, 1] 内")
    results = []
    for record in read_manifest(args.manifest):
        try:
            results.append(run_record(record, args.objects_dir, args.mesh_dir, args.cell_size, args.floor_band, args.body_radius, args.strip_depth, args.strip_half_width, args.wall_strip_depth, args.wall_strip_half_width, args.tie_ratio, args.min_free_cells))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            results.append({"query_id": record["query_id"], "scene_id": record["scene_id"], "utterance": record.get("utterance"), "status": "ineligible", "reason": str(error)})
    counts = Counter(row["status"] for row in results)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "per_query.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8")
    summary = {"total_queries": len(results), "status_counts": dict(sorted(counts.items())), "eligible_queries": counts["stable"] + counts["conflict"], "conflict_fraction_among_eligible": round(counts["conflict"] / (counts["stable"] + counts["conflict"]), 6) if counts["stable"] + counts["conflict"] else None}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
