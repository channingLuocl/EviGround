#!/usr/bin/env python3
"""Classify mesh-supported floor cells by a semantic wall enclosure proxy.

This is a geometry diagnostic, not a room-layout parser.  It uses ScanNet's
vertex segmentation and aggregation files to identify floor and wall points,
closes small gaps in the 2-D wall raster (doorways), then flood-fills from the
outside of the scene bounds.  Cells not reachable from outside are reported as
interior candidates.  The result is independent of the query target.
"""

import argparse
import json
import math
from collections import deque
from pathlib import Path

from run_facing_anchor_pose_experiment import aligned_vertices, mesh_vertices


def load_semantic_vertex_labels(segs_path, aggregation_path, vertex_count):
    segs = json.loads(Path(segs_path).read_text(encoding="utf-8"))["segIndices"]
    groups = json.loads(Path(aggregation_path).read_text(encoding="utf-8"))["segGroups"]
    if len(segs) != vertex_count:
        raise ValueError(f"segIndices 数量 {len(segs)} 与顶点数量 {vertex_count} 不一致")
    labels = {}
    for group in groups:
        label = str(group.get("label", "")).lower().strip()
        for segment_id in group.get("segments", []):
            labels[segment_id] = label
    return [labels.get(segment_id, "") for segment_id in segs]


def quantized_xy(x, y, cell_size):
    return round(x / cell_size), round(y / cell_size)


def dilate(cells, radius):
    if radius < 1:
        return set(cells)
    result = set()
    for x, y in cells:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    result.add((x + dx, y + dy))
    return result


def classify_cells(vertices, labels, cell_size, floor_band, wall_height, wall_gap):
    floor_points = [v for v, label in zip(vertices, labels) if label == "floor"]
    wall_points = [v for v, label in zip(vertices, labels) if label in {"wall", "door", "window"}]
    if not floor_points:
        raise ValueError("语义分割中没有 floor 顶点")
    floor_z = min(v[2] for v in floor_points)
    floor_cells = {
        quantized_xy(x, y, cell_size)
        for x, y, z in floor_points
        if z <= floor_z + floor_band
    }
    wall_cells = {
        quantized_xy(x, y, cell_size)
        for x, y, z in wall_points
        if floor_z + 0.25 <= z <= floor_z + wall_height
    }
    wall_cells = dilate(wall_cells, math.ceil(wall_gap / cell_size))
    if not floor_cells:
        raise ValueError("floor 顶点在 floor-band 内没有形成网格")

    all_cells = set(floor_cells) | wall_cells
    min_x = min(x for x, _ in all_cells) - 2
    max_x = max(x for x, _ in all_cells) + 2
    min_y = min(y for _, y in all_cells) - 2
    max_y = max(y for _, y in all_cells) + 2
    blocked = wall_cells
    queue = deque()
    outside = set()
    for x in range(min_x, max_x + 1):
        queue.extend(((x, min_y), (x, max_y)))
    for y in range(min_y, max_y + 1):
        queue.extend(((min_x, y), (max_x, y)))
    while queue:
        cell = queue.popleft()
        x, y = cell
        if cell in outside or cell in blocked or not (min_x <= x <= max_x and min_y <= y <= max_y):
            continue
        outside.add(cell)
        queue.extend(((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)))
    interior = floor_cells - outside
    return {
        "floor_z": floor_z,
        "floor_cells": floor_cells,
        "wall_cells": wall_cells,
        "interior_cells": interior,
        "outside_floor_cells": floor_cells - interior,
        "grid_bounds": [min_x, max_x, min_y, max_y],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell-size", type=float, default=0.125)
    parser.add_argument("--floor-band", type=float, default=0.08)
    parser.add_argument("--wall-height", type=float, default=2.4)
    parser.add_argument("--wall-gap", type=float, default=0.45)
    args = parser.parse_args()
    if min(args.cell_size, args.floor_band, args.wall_height, args.wall_gap) <= 0:
        raise SystemExit("几何参数必须大于 0")
    scene_id = args.scene_dir.name
    mesh_path = args.scene_dir / f"{scene_id}_vh_clean_2.ply"
    metadata_path = args.scene_dir / f"{scene_id}.txt"
    segs_path = args.scene_dir / f"{scene_id}_vh_clean_2.0.010000.segs.json"
    aggregation_path = args.scene_dir / f"{scene_id}.aggregation.json"
    raw_vertices = mesh_vertices(mesh_path)
    vertices = aligned_vertices(raw_vertices, metadata_path)
    labels = load_semantic_vertex_labels(segs_path, aggregation_path, len(vertices))
    result = classify_cells(vertices, labels, args.cell_size, args.floor_band, args.wall_height, args.wall_gap)
    payload = {
        "scene_id": scene_id,
        "parameters": {"cell_size": args.cell_size, "floor_band": args.floor_band, "wall_height": args.wall_height, "wall_gap": args.wall_gap},
        "counts": {key: len(result[key]) for key in ("floor_cells", "wall_cells", "interior_cells", "outside_floor_cells")},
        "floor_z": result["floor_z"],
        "grid_bounds": result["grid_bounds"],
        "interior_cells": sorted(result["interior_cells"]),
        "outside_floor_cells": sorted(result["outside_floor_cells"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"scene_id": scene_id, "counts": payload["counts"], "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
