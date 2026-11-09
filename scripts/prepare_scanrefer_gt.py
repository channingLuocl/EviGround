"""Prepare a fixed ScanRefer development subset with GT instances and predicted Top-5."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


DATA = Path('/Volumes/KESU/datasets/ScanRefer')
SOURCE = DATA / 'experiment_sources/r23_scores_recovery_v1'
GEOMETRY = Path('/Volumes/KESU/论文/paper/reproductions/SeeGround/local_outputs')
PREDICTIONS = Path('/Users/luocl/Desktop/seeground/outputs/20261020_r8_object_predictions')
DEV = DATA / 'evaluation_splits/r19_v1/dev96_manifest.json'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_once(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Existing input differs; use a fresh output directory: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=32)
    args = parser.parse_args()
    sys.path.insert(0, str(SOURCE / 'src'))
    from eviground.categories.export import load_export

    frozen = read(SOURCE / 'SOURCE_SHA256.json')
    if any(digest(SOURCE / name) != expected for name, expected in frozen.items()):
        raise ValueError('Frozen R23 source changed')
    predictions, clouds, export_hashes = load_export(PREDICTIONS)
    development = read(DEV)
    eligible = [row for row in development['rows']
                if row['scene'] in clouds
                and (GEOMETRY / row['scene'] / 'scene.npz').is_file()
                and (GEOMETRY / row['scene'] / 'scene_instances.json').is_file()]
    if not 1 <= args.count <= len(eligible):
        parser.error(f'--count must be between 1 and {len(eligible)}')
    # Selection uses query identity only, never targets, IoU or historical correctness.
    ordered = sorted(eligible, key=lambda row: hashlib.sha256(
        f"scanrefer-gt-predtop5-dev-v1|{row['csv_row']}".encode()).hexdigest())
    rows = ordered[:args.count]
    prepared = DATA / f'gt_instance_diagnostics/dev{args.count}_v1'
    scenes = sorted({row['scene'] for row in rows})
    identities = {}
    for scene in scenes:
        original = GEOMETRY / scene
        destination = prepared / 'scenes' / scene
        destination.mkdir(parents=True, exist_ok=True)
        cloud = original / 'scene.npz'
        if digest(cloud) != clouds[scene]:
            raise ValueError(f'GT cloud differs from r8 classification input: {scene}')
        raw = read(original / 'scene_instances.json')
        official = DATA / 'evaluation_gt' / scene / 'objects_gt.json'
        gt_boxes = {obj['bbox_id']: obj['bbox_3d'] for obj in read(official)}
        raw_boxes = {obj['bbox_id']: obj['bbox_3d'] for obj in raw}
        for obj in raw:
            if not np.allclose(obj['bbox_3d'], gt_boxes[obj['bbox_id']], atol=2e-5, rtol=0):
                raise ValueError(f'GT coordinate/instance mismatch: {scene}/{obj["bbox_id"]}')
        for prediction in (obj for obj in predictions if obj['scene'] == scene):
            if not np.allclose(prediction['bbox_3d'], raw_boxes[prediction['bbox_id']], atol=1e-7, rtol=0):
                raise ValueError(f'r8 instance geometry mismatch: {scene}/{prediction["bbox_id"]}')
        # Strip annotation category names from the inference geometry table.
        neutral = [{'bbox_id': obj['bbox_id'], 'bbox_3d': obj['bbox_3d'], 'target': 'object'}
                   for obj in raw]
        save_once(destination / 'scene_instances.json', neutral)
        for name, target in [('scene.npz', cloud), ('objects_gt.json', official)]:
            link = destination / name
            if link.is_symlink():
                if link.resolve() != target.resolve():
                    raise ValueError(f'Existing link differs: {link}')
            elif link.exists():
                raise ValueError(f'Expected a symlink: {link}')
            else:
                link.symlink_to(target)
        identities[scene] = {
            'cloud_sha256': clouds[scene],
            'original_instances_sha256': digest(original / 'scene_instances.json'),
            'inference_instances_sha256': digest(destination / 'scene_instances.json'),
            'evaluation_gt_sha256': digest(official),
        }
    manifest = {**development, 'rows': rows,
                'scope': 'Fixed development subset: GT instances with r8 predicted Top-5; not full ScanRefer benchmark',
                'selection_policy': 'SHA256 scanrefer-gt-predtop5-dev-v1|csv_row; no outcome filtering',
                'source_development_sha256': digest(DEV)}
    save_once(prepared / 'manifest.json', manifest)
    status = {
        'queries': len(rows), 'scenes': len(scenes),
        'unique': sum(row['unique'] for row in rows),
        'multiple': sum(not row['unique'] for row in rows),
        'eligible_development_queries': len(eligible),
        'original_development_queries': len(development['rows']),
        'excluded_without_cached_gt_inputs': len(development['rows']) - len(eligible),
        'gt_categories_supplied_to_model': False,
        'query_target_id_supplied_to_model': False,
        'prediction_export': str(PREDICTIONS), 'export_hashes': export_hashes,
        'frozen_source_sha256': frozen, 'scene_inputs': identities,
        'manifest': str(prepared / 'manifest.json'),
        'scene_root': str(prepared / 'scenes'), 'model_calls_sent': 0,
    }
    save_once(prepared / 'PREPARATION.json', status)
    print(json.dumps({key: status[key] for key in (
        'queries', 'scenes', 'unique', 'multiple', 'eligible_development_queries',
        'excluded_without_cached_gt_inputs', 'manifest', 'scene_root', 'model_calls_sent')},
        ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
