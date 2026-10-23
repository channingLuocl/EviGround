"""Scene-disjoint category manifests and annotation-free frozen feature extraction."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from seeground.categories.category_common_v82 import category_vocabulary, file_hash, load_cloud, normalized_category
from seeground.evaluation.run_storage import ensure_run_writable, write_json


VERSION = 'pointnext-category-v10-r1'
SUFFIXES = ('.txt', '.aggregation.json', '_vh_clean_2.0.010000.segs.json', '_vh_clean_2.ply')


def scene_names(path):
    path = Path(path)
    if path.suffix == '.json':
        names = {row['scene'] for row in json.loads(path.read_text())['rows']}
    else:
        names = set(path.read_text().split())
    if not names or any(not re.fullmatch(r'scene\d{4}_\d{2}', name) for name in names):
        raise ValueError('Expected ScanNet scene IDs, one per line or JSON {rows:[{scene:...}]}')
    return sorted(names)


def room(scene):
    return scene.split('_')[0]


def assert_disjoint(first, second):
    if {room(s) for s in first} & {room(s) for s in second}:
        raise ValueError('Scene/physical-room overlap between training, development or evaluation')


def split_training(scenes, dev_fraction=.1, seed=42):
    if not 0 < dev_fraction < 1:
        raise ValueError('dev_fraction must be between zero and one')
    scenes = sorted(scenes)
    groups = sorted({room(s) for s in scenes},
                    key=lambda s: hashlib.sha256(f'{seed}:{s}'.encode()).hexdigest())
    if len(groups) < 2:
        raise ValueError('At least two training rooms are required')
    count = min(len(groups)-1, max(1, round(len(groups)*dev_fraction)))
    dev = set(groups[:count])
    return ([s for s in scenes if room(s) not in dev], [s for s in scenes if room(s) in dev])


def make_manifest(scene_root, scenes, split):
    vocabulary = {normalized_category(c): i for i, c in enumerate(category_vocabulary())}
    samples, sources, excluded, structural = [], [], [], []
    for scene in sorted(scenes):
        directory = Path(scene_root)/scene
        cloud, annotation = directory/'scene.npz', directory/'objects_gt.json'
        _, _, ids = load_cloud(cloud)
        available, seen = set(map(int, np.unique(ids))), set()
        for row in json.loads(annotation.read_text()):
            oid = row['bbox_id']
            if type(oid) is not int or oid < 0 or oid in seen or oid not in available:
                raise ValueError(f'Invalid/duplicate annotated instance: {scene}/{oid}')
            seen.add(oid)
            label = vocabulary.get(normalized_category(row['target']), -1)
            sample = {'scene': scene, 'bbox_id': oid, 'cloud': str(cloud.resolve()),
                      'category': row['target'], 'label': label}
            if normalized_category(row['target']) in ('wall', 'floor', 'ceiling'):
                structural.append(sample)
            elif label < 0 and split == 'train':
                excluded.append(sample)
            else:
                samples.append(sample)
        sources.append({'scene': scene, 'cloud_sha256': file_hash(cloud),
                        'annotation_sha256': file_hash(annotation)})
    if not samples:
        raise ValueError(f'No usable instances in {split}')
    return {'version': VERSION, 'split': split, 'vocabulary': list(category_vocabulary()),
            'scenes': sorted(scenes), 'sources': sources, 'samples': samples,
            'excluded_training_instances': excluded,
            'excluded_structural_instances': structural,
            'supervision': 'ScanNet instance category labels; no part annotations',
            'evaluation_status': 'previously_inspected_validation' if split == 'eval' else split}


def prepare_manifests(args):
    training, validation = scene_names(args.train_scenes), scene_names(args.validation_scenes)
    evaluation = scene_names(args.eval_scenes)
    assert_disjoint(training, validation)
    if not set(evaluation) <= set(validation):
        raise ValueError('Evaluation selection must belong to the supplied official validation list')
    train, dev = split_training(training, args.dev_fraction, args.seed)
    missing = [{'scene': s, 'file': name} for s in sorted(set(training+evaluation))
               for name in ('scene.npz', 'objects_gt.json') if not (args.scene_root/s/name).is_file()]
    if missing:
        raise ValueError(f'Missing {len(missing)} prepared files across '
                         f'{len({r["scene"] for r in missing})} scenes; first: {missing[0]}. '
                         'Run convert on official raw training scans first.')
    ensure_run_writable(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError('Choose an empty manifest output directory')
    for name, scenes in (('train', train), ('dev', dev), ('eval', evaluation)):
        write_json(args.output/f'{name}.json', make_manifest(args.scene_root, scenes, name))
    write_json(args.output/'split.json', {'seed': args.seed, 'dev_fraction': args.dev_fraction,
               'train_scenes_sha256': file_hash(args.train_scenes),
               'validation_scenes_sha256': file_hash(args.validation_scenes),
               'eval_scenes_sha256': file_hash(args.eval_scenes),
               'train': train, 'dev': dev, 'eval': evaluation})


def convert_scene(raw_root, scene, output):
    """Official mesh + instance aggregation -> existing numeric scene format; no author OLT."""
    from plyfile import PlyData
    files = [Path(raw_root)/scene/(scene+s) for s in SUFFIXES]
    if not all(p.is_file() for p in files):
        raise ValueError(f'Missing raw ScanNet files for {scene}: {[str(p) for p in files if not p.is_file()]}')
    ensure_run_writable(output)
    if output.exists():
        raise ValueError(f'Converted scene already exists; choose a fresh output root: {output}')
    vertices = PlyData.read(files[3])['vertex']
    xyz = np.stack([vertices[k] for k in ('x', 'y', 'z')], 1)
    rgb = np.stack([vertices[k] for k in ('red', 'green', 'blue')], 1)
    line = next(line for line in files[0].read_text().splitlines() if line.startswith('axisAlignment'))
    matrix = np.array(list(map(float, line.split('=', 1)[1].split()))).reshape(4, 4)
    xyz = (np.column_stack([xyz, np.ones(len(xyz))]) @ matrix.T)[:, :3].astype(np.float32)
    segments = np.asarray(json.loads(files[2].read_text())['segIndices'])
    if len(segments) != len(xyz) or not np.isfinite(xyz).all():
        raise ValueError('Invalid mesh or segment alignment')
    ids, objects, seen = np.full(len(xyz), -1, dtype=np.int32), [], set()
    for group in json.loads(files[1].read_text())['segGroups']:
        oid = int(group['objectId'])
        mask = np.isin(segments, group['segments'])
        if oid < 0 or oid in seen or (ids[mask] >= 0).any() or not mask.any():
            raise ValueError(f'Invalid instance aggregation: {scene}/{oid}')
        seen.add(oid)
        ids[mask] = oid
        points = xyz[mask]
        objects.append({'bbox_id': oid, 'target': group['label'],
                        'bbox_3d': np.r_[(points.min(0)+points.max(0))/2,
                                             np.ptp(points, axis=0)].tolist()})
    output.mkdir(parents=True)
    np.savez_compressed(output/'scene.npz', xyz=xyz, rgb=rgb, ids=ids)
    write_json(output/'objects_gt.json', objects)
    write_json(output/'provenance.json', {'scene': scene, 'source': 'official ScanNet raw annotations',
               'input_sha256': {p.name: file_hash(p) for p in files},
               'category_mapping': 'raw label with fixed lexical normalization; no subtype merging'})
