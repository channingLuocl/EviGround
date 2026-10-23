#!/usr/bin/env python3
"""Compare a virtual scan center with nearby floor-supported observer positions."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

from explore_horizontal_viewpoints import choose_candidate, load_objects, normalized_label
from run_canonical_anchor_view_experiment import read_manifest
from run_facing_anchor_pose_experiment import aligned_vertices, floor_cells, is_blocked, load_obstacles, mesh_vertices


def scan_center(vertices):
    if not vertices:
        raise ValueError("空网格")
    return tuple((min(v[i] for v in vertices) + max(v[i] for v in vertices)) / 2 for i in (0, 1))


def observer_positions(mode, center, cells, obstacles, radius, cell_size, body_radius):
    if mode == "center":
        return [center]
    return sorted(
        (x * cell_size, y * cell_size)
        for x, y in cells
        if math.hypot(x * cell_size - center[0], y * cell_size - center[1]) <= radius
        and not is_blocked(x * cell_size, y * cell_size, obstacles, body_radius)
    )


def evaluate(record, objects_dir, mesh_dir, args):
    path = objects_dir / record.get("objects_path", f"{record['scene_id']}.json")
    objects = load_objects(path)
    anchor = next((o for o in objects if o['id'] == record['anchor_id']), None)
    if anchor is None:
        raise ValueError("找不到锚点")
    candidates = [o for o in objects if o['id'] != anchor['id'] and o['label'] == normalized_label(record['target_category'])]
    if len(candidates) < 2:
        raise ValueError("少于两个同类非锚点候选")
    scene = mesh_dir / 'scans' / record['scene_id']
    vertices = aligned_vertices(mesh_vertices(scene / f"{record['scene_id']}_vh_clean_2.ply"), scene / f"{record['scene_id']}.txt")
    center = scan_center(vertices)
    cells, floor_z = floor_cells(vertices, args.cell_size, args.floor_band)
    obstacles = load_obstacles(path)
    results = {}
    for mode in ('center', 'central_free'):
        positions = observer_positions(mode, center, cells, obstacles, args.central_radius, args.cell_size, args.body_radius)
        poses = []
        for x, y in positions:
            dx, dy = anchor['center'][0] - x, anchor['center'][1] - y
            if math.hypot(dx, dy) < 1e-6:
                continue
            heading = math.degrees(math.atan2(dy, dx)) % 360
            winner = choose_candidate(candidates, anchor, record['relation'], heading)
            poses.append({'x': x, 'y': y, 'heading_deg': heading, 'winner_id': winner['id']})
        winners = {p['winner_id'] for p in poses}
        status = 'ineligible' if not poses else ('stable' if len(winners) == 1 else 'conflict')
        row = {
            'query_id': record['query_id'], 'scene_id': record['scene_id'],
            'utterance': record.get('utterance'), 'mode': mode, 'status': status,
            'scan_center_xy': center, 'floor_z': floor_z, 'floor_cell_count': len(cells),
            'center_bbox_blocked': is_blocked(*center, obstacles, args.body_radius),
            'pose_count': len(poses), 'poses': poses,
        }
        if not poses:
            row['reason'] = '中央区域没有通过地面/障碍检查且朝向可定义的位置' if mode == 'central_free' else '中心与锚点水平位置重合，朝向未定义'
        if 'target_id' in record:
            row['evaluation_only'] = {'target_id': record['target_id'], 'winner_matches_target': status == 'stable' and next(iter(winners)) == record['target_id']}
        results[mode] = row
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--objects-dir', type=Path, required=True)
    parser.add_argument('--mesh-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--central-radius', type=float, default=1.0)
    parser.add_argument('--cell-size', type=float, default=0.25)
    parser.add_argument('--floor-band', type=float, default=0.08)
    parser.add_argument('--body-radius', type=float, default=0.30)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('输出目录已存在，请使用新路径')
    if any(not math.isfinite(v) or v <= 0 for v in (args.central_radius, args.cell_size, args.floor_band, args.body_radius)):
        parser.error('几何参数必须为有限正数')
    results = {'center': [], 'central_free': []}
    records = read_manifest(args.manifest)
    for record in records:
        try:
            pair = evaluate(record, args.objects_dir, args.mesh_dir, args)
        except (OSError, ValueError) as error:
            pair = {mode: {'query_id': record['query_id'], 'scene_id': record['scene_id'], 'utterance': record.get('utterance'), 'mode': mode, 'status': 'error', 'reason': str(error)} for mode in results}
        for mode in results:
            results[mode].append(pair[mode])
    args.output_dir.mkdir(parents=True)
    summary = {'parameters': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               'limitations': ['扫描水平AABB中心，不保证位于房间内部', '中央位置沿用最低点地面估计和二维bbox障碍代理，不验证可达性或可见性', '沿用原有候选横向排序，仅比较观察方向'], 'methods': {}}
    for mode, rows in results.items():
        counts = Counter(r['status'] for r in rows)
        evaluated = [r for r in rows if 'evaluation_only' in r]
        stable_evaluated = [r for r in evaluated if r['status'] == 'stable']
        correct = sum(r['evaluation_only']['winner_matches_target'] for r in evaluated)
        summary['methods'][mode] = {'total': len(rows), 'status_counts': dict(counts), 'correct': correct,
            'coverage': counts['stable'] / len(rows),
            'stable_accuracy': correct / len(stable_evaluated) if stable_evaluated else None,
            'correct_fraction_all': correct / len(rows) if all('target_id' in r for r in records) else None}
        (args.output_dir / f'{mode}_per_query.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    (args.output_dir / 'summary.json').write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
