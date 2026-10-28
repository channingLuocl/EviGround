"""Stage existing GT clouds and build absent scenes without inference category labels."""
import hashlib
import json
from pathlib import Path
import numpy as np
from plyfile import PlyData

ROOT = Path(__file__).resolve().parent
CACHE = Path('/Volumes/KESU/论文/paper/reproductions/SeeGround/local_outputs')
RAW = Path('/Volumes/KESU/datasets/ScanRefer/raw_scans')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def boxes(xyz, ids):
    result = []
    for oid in sorted(set(map(int, np.unique(ids))) - {-1}):
        points = xyz[ids == oid]
        low, high = points.min(0), points.max(0)
        result.append({'bbox_id': oid, 'target': 'object',
                       'bbox_3d': np.r_[(low + high) / 2, high - low].tolist()})
    return result


def build(scene, directory):
    raw = RAW / scene
    ply = raw / f'{scene}_vh_clean_2.ply'
    vertex = PlyData.read(ply)['vertex']
    xyz = np.column_stack([vertex[k] for k in ('x', 'y', 'z')]).astype(float)
    rgb = np.column_stack([vertex[k] for k in ('red', 'green', 'blue')]).astype(np.uint8)
    segmentation = raw / f'{scene}_vh_clean_2.0.010000.segs.json'
    seg = np.asarray(json.loads(segmentation.read_text())['segIndices'])
    if len(seg) != len(xyz) or not np.isfinite(xyz).all():
        raise ValueError(f'Invalid raw coordinates/segments: {scene}')
    alignment = raw / f'{scene}.txt'
    line = next(x for x in alignment.read_text().splitlines() if x.startswith('axisAlignment'))
    # Match frozen prepare_scene's float32 matrix applied to float64 coordinates.
    matrix = np.asarray([float(v) for v in line.split('=', 1)[1].split()], dtype=np.float32).reshape(4, 4)
    xyz = (np.column_stack([xyz, np.ones(len(xyz))]) @ matrix.T)[:, :3]
    aggregation = raw / f'{scene}.aggregation.json'
    groups = json.loads(aggregation.read_text())['segGroups']
    ids = np.full(len(xyz), -1, dtype=np.int64)
    seen = set()
    for group in groups:
        oid = int(group['objectId'])
        mask = np.isin(seg, group['segments'])
        if oid in seen or (ids[mask] >= 0).any():
            raise ValueError(f'Duplicate/overlapping raw GT instance: {scene}/{oid}')
        seen.add(oid)
        ids[mask] = oid
    np.savez_compressed(directory / 'scene.npz', xyz=xyz, rgb=rgb, ids=ids)
    table = boxes(xyz, ids)
    save(directory / 'scene_instances.json', table)
    save(directory / 'objects_gt.json', table)  # Evaluation needs only ID/geometry; no GT names retained.
    record = {'scene': scene, 'method': 'frozen prepare_scene geometry; raw GT segmentation and axis alignment',
              'labels': 'object (all GT category names removed)', 'vertices': len(xyz), 'instances': len(table),
              'raw_sha256': {p.name: digest(p) for p in (ply, segmentation, alignment, aggregation)},
              'output_sha256': {n: digest(directory / n) for n in ('scene.npz', 'scene_instances.json', 'objects_gt.json')}}
    save(directory / 'GEOMETRY_PROVENANCE.json', record)
    return record


def main():
    rows = json.loads((ROOT / 'inputs/full_manifest.json').read_text())['rows']
    export = json.loads((ROOT / 'inputs/r8_predictions/predictions.json').read_text())
    source = json.loads((ROOT / 'inputs/r8_predictions/config.json').read_text())
    sources = {r['scene']: r['cloud_sha256'] for r in source['sources']}
    predicted = {(r['scene'], r['bbox_id']) for r in export['results']}
    scenes = sorted({r['scene'] for r in rows})
    records = []
    for scene in scenes:
        directory = ROOT / 'inputs/scenes' / scene
        directory.mkdir(exist_ok=True)
        if not (directory / 'scene.npz').exists():
            cached = CACHE / scene / 'scene.npz'
            if scene in sources:
                if digest(cached) != sources[scene]:
                    raise ValueError(f'Cached cloud hash differs from original r8 input: {scene}')
                (directory / 'scene.npz').symlink_to(cached)
            else:
                build(scene, directory)
        provenance = directory / 'GEOMETRY_PROVENANCE.json'
        if provenance.exists():
            records.append(json.loads(provenance.read_text()))
        if not all((directory / n).exists() for n in ('scene_instances.json', 'objects_gt.json')):
            with np.load(directory / 'scene.npz', allow_pickle=False) as cloud:
                table = boxes(cloud['xyz'], cloud['ids'])
            for name in ('scene_instances.json', 'objects_gt.json'):
                if not (directory / name).exists():
                    save(directory / name, table)
        objects = json.loads((directory / 'objects_gt.json').read_text())
        known = {obj['bbox_id'] for obj in objects}
        missing = {r['target_id'] for r in rows if r['scene'] == scene} - known
        if missing:
            raise ValueError(f'Missing referred GT geometry: {scene}/{missing}')
        print(json.dumps({'scene': scene, 'geometry_ready': True, 'original_r8_categories': scene in sources}), flush=True)
    missing_rows = [r for r in rows if (r['scene'], r['target_id']) not in predicted]
    save(ROOT / 'provenance/GENERATED_GT_GEOMETRY.json', records)
    save(ROOT / 'provenance/INPUT_COVERAGE.json', {
        'queries': len(rows), 'scenes': len(scenes), 'geometry_complete': True,
        'geometry_instances_created': sum(r['instances'] for r in records),
        'category_covered_queries': len(rows) - len(missing_rows),
        'missing_category_queries': len(missing_rows),
        'missing_category_scenes': sorted({r['scene'] for r in missing_rows}),
        'manifest_sha256': digest(ROOT / 'inputs/full_manifest.json'),
        'no_gt_categories_used': True,
        'classification_status': 'pending original r8 inference for missing scenes'})


if __name__ == '__main__':
    main()
