#!/usr/bin/env python3
"""Check left/right stability over mesh-supported poses facing one anchor.

This is a conservative second diagnostic.  It approximates standing locations
from low mesh vertices and removes positions inside expanded GT object boxes.
It does not claim to recover the speaker's real pose or model visibility.
"""

import argparse
import json
import math
import struct
from collections import Counter
from pathlib import Path

from explore_horizontal_viewpoints import choose_candidate, load_objects, normalized_label


def read_manifest(path):
    records = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        required = {"query_id", "scene_id", "target_category", "anchor_id", "relation", "viewpoint_constraint"}
        if not isinstance(record, dict) or not required <= record.keys():
            raise ValueError(f"清单第 {number} 行缺少必填字段")
        if record["viewpoint_constraint"] != "facing_anchor":
            raise ValueError(f"清单第 {number} 行不是 facing_anchor 查询")
        if record["relation"] not in {"left", "right"}:
            raise ValueError(f"清单第 {number} 行 relation 必须为 left 或 right")
        records.append(record)
    if not records:
        raise ValueError("清单为空")
    return records


def mesh_vertices(path):
    """Read x/y/z vertices from ASCII or binary-little-endian PLY."""
    with Path(path).open("rb") as handle:
        header = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"PLY 头不完整：{path}")
            decoded = line.decode("ascii").strip()
            header.append(decoded)
            if decoded == "end_header":
                break
        if header[0] != "ply":
            raise ValueError(f"不是 PLY 文件：{path}")
        format_line = next((line for line in header if line.startswith("format ")), None)
        if format_line not in {"format ascii 1.0", "format binary_little_endian 1.0"}:
            raise ValueError(f"仅支持 ASCII 或小端二进制 PLY：{path}")
        vertex_index = header.index(next(line for line in header if line.startswith("element vertex ")))
        count = int(header[vertex_index].split()[-1])
        properties = []
        for line in header[vertex_index + 1:]:
            if line.startswith("element "):
                break
            if line.startswith("property list"):
                raise ValueError("vertex list property 不受支持")
            if line.startswith("property "):
                _, kind, name = line.split()
                properties.append((kind, name))
        names = [name for _, name in properties]
        if not {"x", "y", "z"} <= set(names):
            raise ValueError(f"PLY 缺少 x/y/z 顶点坐标：{path}")
        if format_line == "format ascii 1.0":
            rows = [handle.readline().split() for _ in range(count)]
            return [(float(row[names.index("x")]), float(row[names.index("y")]), float(row[names.index("z")])) for row in rows]
        codes = {"char": "b", "uchar": "B", "short": "h", "ushort": "H", "int": "i", "uint": "I", "float": "f", "double": "d"}
        try:
            unpacker = struct.Struct("<" + "".join(codes[kind] for kind, _ in properties))
        except KeyError as error:
            raise ValueError(f"不支持的 PLY 属性类型：{error.args[0]}") from error
        x_i, y_i, z_i = names.index("x"), names.index("y"), names.index("z")
        return [tuple(unpacker.unpack(handle.read(unpacker.size))[i] for i in (x_i, y_i, z_i)) for _ in range(count)]


def load_obstacles(path):
    raw = json.loads(Path(path).read_text())
    obstacles = []
    for item in raw:
        bbox = item.get("bbox_3d", [])
        if len(bbox) >= 6 and all(isinstance(value, (int, float)) for value in bbox[:6]):
            obstacles.append((bbox[0], bbox[1], abs(bbox[3]) / 2, abs(bbox[4]) / 2))
    return obstacles


def floor_cells(vertices, cell_size, floor_band):
    floor_z = min(vertex[2] for vertex in vertices)
    return {
        (round(x / cell_size), round(y / cell_size))
        for x, y, z in vertices
        if z <= floor_z + floor_band
    }, floor_z


def is_blocked(x, y, obstacles, radius):
    return any(abs(x - cx) <= half_x + radius and abs(y - cy) <= half_y + radius for cx, cy, half_x, half_y in obstacles)


def pose_result(record, objects_dir, mesh_dir, cell_size, floor_band, body_radius):
    object_path = objects_dir / record.get("objects_path", f"{record['scene_id']}.json")
    mesh_path = mesh_dir / "scans" / record["scene_id"] / f"{record['scene_id']}_vh_clean_2.ply"
    objects = load_objects(object_path)
    anchor = next((obj for obj in objects if obj["id"] == record["anchor_id"]), None)
    if anchor is None:
        raise ValueError("找不到锚点")
    candidates = [obj for obj in objects if obj["id"] != anchor["id"] and obj["label"] == normalized_label(record["target_category"])]
    if len(candidates) < 2:
        raise ValueError("少于两个同类非锚点候选")
    cells, floor_z = floor_cells(mesh_vertices(mesh_path), cell_size, floor_band)
    obstacles = load_obstacles(object_path)
    poses = []
    for cell_x, cell_y in sorted(cells):
        x, y = cell_x * cell_size, cell_y * cell_size
        if is_blocked(x, y, obstacles, body_radius):
            continue
        dx, dy = anchor["center"][0] - x, anchor["center"][1] - y
        if math.hypot(dx, dy) < body_radius:
            continue
        heading = math.degrees(math.atan2(dy, dx)) % 360
        winner = choose_candidate(candidates, anchor, record["relation"], heading)
        poses.append({"x": round(x, 3), "y": round(y, 3), "z": round(floor_z, 3), "heading_deg": round(heading, 3), "winner_id": winner["id"]})
    if not poses:
        raise ValueError("没有 mesh 支持且不与物体框碰撞的候选观察位置")
    counts = Counter(pose["winner_id"] for pose in poses)
    support = [{"candidate_id": candidate["id"], "poses": counts[candidate["id"]], "fraction": round(counts[candidate["id"]] / len(poses), 6)} for candidate in candidates]
    return {
        "query_id": record["query_id"], "scene_id": record["scene_id"], "utterance": record.get("utterance"),
        "status": "stable" if len(counts) == 1 else "conflict",
        "assumptions": {"viewpoint_constraint": "observer faces the anchor from every sampled position", "standing_locations": "low mesh vertices quantized to a grid, excluding expanded GT object boxes", "excluded": ["true speaker pose", "visibility", "occlusion", "room connectivity", "camera height"]},
        "input": {"target_category": record["target_category"], "anchor_id": record["anchor_id"], "relation": record["relation"]},
        "floor_z": round(floor_z, 3), "feasible_pose_count": len(poses), "candidate_support": support, "poses": poses,
    }


def main():
    parser = argparse.ArgumentParser(description="在 mesh 支持的可站立近似位置中检验 facing-anchor 查询")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--objects-dir", type=Path, required=True)
    parser.add_argument("--mesh-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cell-size", type=float, default=0.25)
    parser.add_argument("--floor-band", type=float, default=0.08)
    parser.add_argument("--body-radius", type=float, default=0.30)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("输出目录已存在；请指定新路径")
    if min(args.cell_size, args.floor_band, args.body_radius) <= 0:
        raise SystemExit("cell-size、floor-band 与 body-radius 必须大于 0")
    results = []
    for record in read_manifest(args.manifest):
        try:
            results.append(pose_result(record, args.objects_dir, args.mesh_dir, args.cell_size, args.floor_band, args.body_radius))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            results.append({"query_id": record["query_id"], "scene_id": record["scene_id"], "status": "ineligible", "reason": str(error)})
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "per_query.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8")
    counts = Counter(row["status"] for row in results)
    summary = {"total_queries": len(results), "status_counts": dict(sorted(counts.items())), "eligible_queries": counts["stable"] + counts["conflict"], "conflict_fraction_among_eligible": round(counts["conflict"] / (counts["stable"] + counts["conflict"]), 6) if counts["stable"] + counts["conflict"] else None}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
