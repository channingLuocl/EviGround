"""Numeric inputs and explicitly annotated evidence; no automatic absence labels."""
import json
from pathlib import Path
import re

import numpy as np

from category_common_v82 import category_vocabulary, file_hash, normalized_category


VERSION = 'object-category-v8.3-r1'
STATES = ('present', 'absent', 'unknown')
PATCHES = 8


def load_parts(path):
    value = json.loads(Path(path).read_text())
    names = value['parts']
    if (value.get('format') != 'v8.3-parts-v1' or not names
            or any(not isinstance(n, str) or not n.strip() for n in names)
            or len(set(names)) != len(names) or value.get('prompt', '').count('{}') != 1):
        raise ValueError('Invalid public part vocabulary/prompt')
    return value


def state_targets(annotations, parts):
    """-1 is UNLABELLED (loss masked); 2 is an explicit unknown annotation."""
    if set(annotations) - set(parts) or any(s not in STATES for s in annotations.values()):
        raise ValueError('Unrecognized part/state annotation')
    return np.array([STATES.index(annotations[p]) if p in annotations else -1
                     for p in parts], dtype=np.int64)


def load_manifest(path, parts):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if (value.get('format') != 'v8.3-training-manifest-v1'
            or value.get('split') not in ('train', 'dev')
            or not value.get('annotation_protocol') or not value.get('samples')):
        raise ValueError('Need a train/dev manifest and explicit annotation_protocol; see README_V8_3.md')
    vocabulary = {normalized_category(c): i for i, c in enumerate(category_vocabulary())}
    samples, seen = [], set()
    for row in value['samples']:
        scene, oid = row['scene'], row['bbox_id']
        if (not isinstance(scene, str) or not scene or Path(scene).name != scene
                or scene in ('.', '..') or type(oid) is not int or oid < 0):
            raise ValueError('Invalid scene/instance identity')
        cloud = (path.parent / row['cloud']).resolve()
        label = normalized_category(row['category'])
        if label not in vocabulary:
            raise ValueError(f'Training category outside fixed vocabulary: {row["category"]}')
        if (scene, oid) in seen:
            raise ValueError('Duplicate scene-instance in manifest')
        seen.add((scene, oid))
        masks = (path.parent / row['part_masks']).resolve() if row.get('part_masks') else None
        samples.append({'scene': scene, 'bbox_id': oid, 'cloud': cloud,
                        'label': vocabulary[label], 'states': state_targets(row.get('part_states', {}), parts),
                        'part_masks': masks})
    return value, samples


def load_part_masks(path, count, parts, states):
    with np.load(path, allow_pickle=False) as data:
        names, masks = data['parts'].tolist(), data['masks']
    if names != parts or masks.shape != (count, len(parts)) or masks.dtype != np.bool_:
        raise ValueError('part_masks must contain parts in schema order and bool masks[N_instance,P]')
    if any(masks[:, i].any() and states[i] != 0 for i in range(len(parts))):
        raise ValueError('Observed part masks require an explicit present annotation')
    if any(states[i] == 0 and not masks[:, i].any() for i in range(len(parts))):
        raise ValueError('Every present part needs observed support for crop supervision')
    return masks


def crop_view(xyz, states, masks, seed):
    """Observed-point removal, NOT physical part removal or absence supervision."""
    rng = np.random.RandomState(seed)
    direction = rng.normal(size=3)
    order = np.argsort(xyz @ direction, kind='stable')
    keep = np.sort(order[:max(3, int(len(xyz) * rng.uniform(.5, .8)))])
    targets = np.where(states >= 0, 2, -1).astype(np.int64)
    for i in np.flatnonzero(states == 0):
        required = min(3, int(masks[:, i].sum()))
        if masks[keep, i].sum() >= required:
            targets[i] = 0
    return keep, targets


def local_patches(xyz, seed):
    """Deterministic FPS anchors + nearest patches, no semantic labels needed."""
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 3 or not np.isfinite(xyz).all():
        raise ValueError('Need at least three finite XYZ points')
    center = xyz.mean(0)
    radius = np.linalg.norm(xyz - center, axis=1).max()
    if radius <= 1e-12:
        raise ValueError('Instance geometry has zero radius')
    normalized = (xyz - center) / radius
    rng = np.random.RandomState(seed)
    pool = rng.choice(len(xyz), min(4096, len(xyz)), replace=False)
    candidates = normalized[pool]
    anchor = int(np.argmax(np.linalg.norm(candidates, axis=1)))
    distance = np.full(len(pool), np.inf)
    patches, layout = [], []
    for _ in range(PATCHES):
        position = candidates[anchor]
        distances = np.sum((normalized - position) ** 2, axis=1)
        indices = np.argsort(distances, kind='stable')[:max(3, min(len(xyz), max(64, len(xyz)//4)))]
        points = normalized[indices]
        if np.ptp(points, axis=0).max() <= 1e-12:
            indices, points = np.arange(len(xyz)), normalized
        patches.append(indices)
        layout.append(np.r_[points.mean(0), np.ptp(points, axis=0), len(indices)/len(xyz)])
        distance = np.minimum(distance, np.sum((candidates - position) ** 2, axis=1))
        anchor = int(np.argmax(distance))
    return patches, np.asarray(layout, dtype=np.float32)


def manifest_sources(samples):
    hashes = {p: file_hash(p) for r in samples for p in (r['cloud'], r['part_masks']) if p}
    return [{'scene': r['scene'], 'bbox_id': r['bbox_id'], 'cloud_sha256': hashes[r['cloud']],
             'part_masks_sha256': hashes[r['part_masks']] if r['part_masks'] else None}
            for r in samples]


def check_disjoint(first, second):
    def group(scene):
        # ScanNet rescans of the same physical room must stay in the same split.
        return scene.split('_')[0] if re.fullmatch(r'scene\d{4}_\d{2}', scene) else scene
    scenes = {group(s['scene']) for s in first} & {group(s['scene']) for s in second}
    clouds = {s['cloud_sha256'] for s in first} & {s['cloud_sha256'] for s in second}
    if scenes or clouds:
        raise ValueError('Train/dev/evaluation overlap (scene or cloud contents); use disjoint scenes')
