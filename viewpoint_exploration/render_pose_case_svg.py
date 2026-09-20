#!/usr/bin/env python3
"""Render one facing-anchor diagnostic as a self-contained top-down SVG."""

import argparse
import html
import json
from pathlib import Path


PALETTE = ["#d73027", "#4575b4", "#1a9850", "#984ea3", "#ff7f00"]


def read_query(path, query_id):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["query_id"] == query_id:
            if row["status"] not in {"stable", "conflict"}:
                raise ValueError("该查询没有可视化的可行位姿")
            return row
    raise ValueError(f"结果中找不到 query_id={query_id}")


def scale(value, low, high, pixels):
    return 50 + (value - low) / (high - low) * (pixels - 100)


def main():
    parser = argparse.ArgumentParser(description="输出 facing-anchor 案例的俯视 SVG")
    parser.add_argument("--per-query", type=Path, required=True)
    parser.add_argument("--objects", type=Path, required=True)
    parser.add_argument("--query-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("输出文件已存在；请使用新路径")

    row = read_query(args.per_query, args.query_id)
    objects = json.loads(args.objects.read_text(encoding="utf-8"))
    by_id = {obj["bbox_id"]: obj for obj in objects}
    anchor_id = row["input"]["anchor_id"]
    candidate_ids = [item["candidate_id"] for item in row["candidate_support"]]
    highlighted = {anchor_id, *candidate_ids}
    boxes = []
    for obj in objects:
        x, y, _, width, length, _ = obj["bbox_3d"]
        boxes.append((x - width / 2, x + width / 2, y - length / 2, y + length / 2))
    points = [(pose["x"], pose["y"]) for pose in row["poses"]]
    x_low = min([box[0] for box in boxes] + [point[0] for point in points]) - 0.4
    x_high = max([box[1] for box in boxes] + [point[0] for point in points]) + 0.4
    y_low = min([box[2] for box in boxes] + [point[1] for point in points]) - 0.4
    y_high = max([box[3] for box in boxes] + [point[1] for point in points]) + 0.4
    side = 800
    winners = sorted({pose["winner_id"] for pose in row["poses"]})
    colors = {winner: PALETTE[index] for index, winner in enumerate(winners)}
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{side}" height="{side + 100}" viewBox="0 0 {side} {side + 100}">', '<rect width="100%" height="100%" fill="white"/>']
    svg.append(f'<text x="50" y="28" font-family="sans-serif" font-size="16">{html.escape(row.get("utterance") or args.query_id)}</text>')
    for obj, box in zip(objects, boxes):
        x0, x1, y0, y1 = box
        left, right = scale(x0, x_low, x_high, side), scale(x1, x_low, x_high, side)
        top, bottom = side - scale(y1, y_low, y_high, side), side - scale(y0, y_low, y_high, side)
        if obj["bbox_id"] == anchor_id:
            fill, stroke, opacity = "#9ecae1", "#08519c", "0.85"
        elif obj["bbox_id"] in candidate_ids:
            fill, stroke, opacity = "#fdd0a2", "#d94801", "0.85"
        else:
            fill, stroke, opacity = "#d9d9d9", "#999999", "0.35"
        svg.append(f'<rect x="{left:.1f}" y="{top:.1f}" width="{right-left:.1f}" height="{bottom-top:.1f}" fill="{fill}" fill-opacity="{opacity}" stroke="{stroke}"/>')
        if obj["bbox_id"] in highlighted:
            label = f'#{obj["bbox_id"]} {obj["target"]}'
            svg.append(f'<text x="{left+3:.1f}" y="{top+15:.1f}" font-family="sans-serif" font-size="13">{html.escape(label)}</text>')
    for pose in row["poses"]:
        x, y = scale(pose["x"], x_low, x_high, side), side - scale(pose["y"], y_low, y_high, side)
        color = colors[pose["winner_id"]]
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" stroke="white" stroke-width="2"/>')
    legend_y = side + 35
    svg.append('<text x="50" y="820" font-family="sans-serif" font-size="13">蓝色框：锚点；橙色框：同类候选；圆点：mesh 支持且避开物体框的观察位置。</text>')
    for index, winner in enumerate(winners):
        x = 50 + index * 210
        label = f'赢家 #{winner} ({by_id[winner]["target"]})'
        svg.append(f'<circle cx="{x}" cy="{legend_y}" r="7" fill="{colors[winner]}"/><text x="{x+13}" y="{legend_y+5}" font-family="sans-serif" font-size="14">{html.escape(label)}</text>')
    svg.append('</svg>')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(json.dumps({"query_id": args.query_id, "output": str(args.output), "poses": len(row["poses"]), "winner_ids": winners}, ensure_ascii=False))


if __name__ == "__main__":
    main()
