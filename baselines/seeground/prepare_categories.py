"""Pack ScanRefer clouds for inference, keeping GT in separate evaluation files."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(root, output):
    queries_path = root / 'ScanRefer_filtered_val.json'
    queries = json.loads(queries_path.read_text())
    targets = {}
    for row in queries:
        key = (row['scene_id'], int(row['object_id']))
        label = row['object_name'].replace('_', ' ')
        if key in targets and targets[key] != label:
            raise ValueError(f'Inconsistent query labels for {key}')
        targets[key] = label
    scenes = sorted({scene for scene, _ in targets})
    for directory in ('inputs', 'gt'):
        (output / directory).mkdir(parents=True, exist_ok=True)
    records, gt_records = [], []
    for number, scene in enumerate(scenes, 1):
        source = root / 'raw_scans' / scene
        ply = source / f'{scene}_vh_clean_2.ply'
        segs = source / f'{scene}_vh_clean_2.0.010000.segs.json'
        agg = source / f'{scene}.aggregation.json'
        destination = output / 'inputs' / f'{scene}.npz'
        ground_truth = output / 'gt' / f'{scene}.npz'
        segments = np.asarray(json.loads(segs.read_text())['segIndices'], dtype=np.int32)
        groups = json.loads(agg.read_text())['segGroups']
        if not destination.exists():
            vertices = PlyData.read(ply)['vertex']
            xyz = np.column_stack([vertices[k] for k in ('x', 'y', 'z')]).astype(np.float32)
            rgb = np.column_stack([vertices[k] for k in ('red', 'green', 'blue')]).astype(np.uint8)
            if len(segments) != len(xyz):
                raise ValueError(f'Segment/vertex mismatch in {scene}')
            temporary = destination.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, xyz=xyz, rgb=rgb, segments=segments)
            temporary.replace(destination)
        segment_to_object = np.full(int(segments.max()) + 1, -1, dtype=np.int32)
        labels = {}
        for group in groups:
            object_id = int(group['objectId'])
            segment_to_object[group['segments']] = object_id
            labels[object_id] = group['label']
        ids = segment_to_object[segments]
        np.savez_compressed(ground_truth, instance_ids=ids)
        for (target_scene, object_id), query_label in sorted(targets.items()):
            if target_scene == scene:
                if not np.any(ids == object_id):
                    raise ValueError(f'Empty target {scene}/{object_id}')
                gt_records.append({'scene_id': scene, 'object_id': object_id,
                                   'gt_label': labels[object_id], 'query_label': query_label})
        records.append({'scene_id': scene, 'vertices': len(segments),
                        'input_sha256': sha256(destination),
                        'gt_sha256': sha256(ground_truth),
                        'raw_sha256': {p.name: sha256(p) for p in (ply, segs, agg)}})
        print(json.dumps({'stage': 'packing', 'completed': number, 'scenes': len(scenes)}), flush=True)
    (output / 'targets.json').write_text(json.dumps(gt_records, ensure_ascii=False, indent=2))
    (output / 'manifest.json').write_text(json.dumps({
        'scenes': records, 'queries': len(queries), 'objects': len(targets),
        'queries_sha256': sha256(queries_path),
        'gt_label': 'Exact original ScanNet aggregation label; query label also retained',
        'coordinates': 'raw ScanNet vertices, no alignment or centering; original vertex order',
        'inputs': 'xyz, RGB, superpoints only; no GT class or instance IDs'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root, args.output)
