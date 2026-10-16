"""Frozen v8.3 predictions become gated category hypotheses, never annotations."""
import hashlib
import json
from pathlib import Path

import numpy as np

VERSION = 'predicted-category-grounding-v9-r1'
MIN_SCORE = .20
MAX_GAP = .03
CATEGORY_INSTRUCTIONS = (
    ' Category labels are uncertain, mutually exclusive hypotheses per object ID. '
    'category is the frozen top-1 display label only; use category_hypotheses for eligible bindings. '
    'Scores are similarities with positive local support, not probabilities. '
    'Match target and reference categories to retained hypotheses, then verify with visual/spatial evidence. '
    'Do not assign incompatible categories to the same object in one interpretation. '
    'A retained category is not proof; a rejected category is not proof of physical absence. '
    'Return uncertainty when essential category evidence is missing.')


def category_names(obj):
    if 'category_hypotheses' in obj:
        return [h['category'] for h in obj['category_hypotheses']]
    return [obj['target']]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def gate_categories(top5, top_k=5, min_score=MIN_SCORE, max_gap=MAX_GAP):
    if (top_k not in (1, 5) or not -1 <= min_score <= 1 or
            not 0 <= max_gap <= 2 or not np.isfinite([min_score, max_gap]).all()):
        raise ValueError('Expected top_k=1|5, finite similarity threshold and nonnegative score gap')
    if (len(top5) != 5 or len({r['category'] for r in top5}) != 5 or
            any(not isinstance(r['category'], str) or not r['category'].strip() for r in top5)):
        raise ValueError('Expected five distinct category predictions')
    scores = np.asarray([r['score'] for r in top5], dtype=float)
    if not np.isfinite(scores).all() or np.any(scores[:-1] < scores[1:]):
        raise ValueError('Category scores must be finite and sorted descending')
    accepted, audit = [], []
    for rank, row in enumerate(top5, 1):
        gap = float(scores[0] - row['score'])
        reasons = []
        if rank > top_k:
            reasons.append('outside_top_k')
        if row['score'] < min_score:
            reasons.append('below_min_score')
        if gap > max_gap + 1e-12:
            reasons.append('gap_exceeds_max')
        item = {'category': row['category'], 'score': float(row['score']),
                'rank': rank, 'gap_from_top1': gap}
        if not reasons:
            accepted.append(item)
        audit.append({**item, 'retained': not reasons, 'reasons': reasons})
    return accepted, audit


def category_fields(obj):
    """Keep hypotheses in model tables; absence of evidence is not a category verdict."""
    if 'category_hypotheses' not in obj:
        return {}
    return {'category_hypotheses': obj['category_hypotheses'],
            'category_status': obj['category_status'],
            'category_score_type': 'cosine_plus_positive_support_not_probability'}


def load_category_objects(path):
    data = json.loads(Path(path).read_text())
    if data.get('version') != VERSION:
        raise ValueError('Expected a prepared v9 category table')
    result = []
    for item in data['objects']:
        box = np.asarray(item['bbox_3d'], dtype=float)
        if (type(item['bbox_id']) is not int or box.shape != (6,) or
                not np.isfinite(box).all() or np.any(box[3:] <= 0)):
            raise ValueError('Invalid predicted instance geometry')
        retained, audit = gate_categories(item['top5'], **data['gate'])
        if retained != item['category_hypotheses'] or audit != item['category_gate_audit']:
            raise ValueError('Category table disagrees with its recorded gate')
        result.append({key: item[key] for key in
                       ('bbox_id', 'target', 'bbox_3d', 'category_hypotheses', 'category_status')})
        expected_status = 'single_hypothesis' if len(retained) == 1 else 'uncertain'
        if item['category_status'] != expected_status:
            raise ValueError('Category status disagrees with retained hypotheses')
        if item['target'] != item['top5'][0]['category']:
            raise ValueError('Display label differs from frozen top-1')
    if not result or len({r['bbox_id'] for r in result}) != len(result):
        raise ValueError('Empty or duplicate category table')
    return result


def prepare_tables(source, scene_root, output, scenes, top_k=5,
                   min_score=MIN_SCORE, max_gap=MAX_GAP):
    """Read predictions and input hashes only; no author/GT category files."""
    source, scene_root, output = map(Path, (source, scene_root, output))
    marker = json.loads((source/'FROZEN.json').read_text())
    for name in ('predictions.json', 'config.json'):
        if sha256(source/name) != marker['files'][name]:
            raise ValueError(f'Frozen classifier file changed: {name}')
    config = json.loads((source/'config.json').read_text())
    data = json.loads((source/'predictions.json').read_text())
    if (not data['summary']['complete'] or data['summary']['errors'] or
            data['summary']['version'] != 'object-category-v8.3-zero-shot-r1'):
        raise ValueError('Need a complete, successful frozen v8.3 prediction run')
    jobs = {(r['scene'], r['bbox_id']) for r in config['jobs']}
    rows = {(r['scene'], r['bbox_id']): r for r in data['results']}
    if len(rows) != len(data['results']) or set(rows) != jobs:
        raise ValueError('Predictions do not cover the frozen selection exactly')
    sources = {r['scene']: r['cloud_sha256'] for r in config['sources']}
    gate = {'top_k': top_k, 'min_score': min_score, 'max_gap': max_gap}
    files, counts = {}, []
    for scene in sorted(scenes):
        if Path(scene).name != scene or scene in ('.', '..') or scene not in sources:
            raise ValueError(f'Unknown scene: {scene}')
        cloud_path = scene_root/scene/'scene.npz'
        if sha256(cloud_path) != sources[scene]:
            raise ValueError(f'Point cloud differs from classifier input: {scene}')
        with np.load(cloud_path, allow_pickle=False) as cloud:
            known = set(map(int, np.unique(cloud['ids'])))
        objects = []
        for (name, oid), row in sorted(rows.items()):
            if name != scene:
                continue
            if oid not in known or row['status'] != 'completed':
                raise ValueError(f'Missing/failed classified instance: {scene}/{oid}')
            retained, audit = gate_categories(row['top5'], **gate)
            objects.append({'bbox_id': oid, 'target': row['predicted_object'],
                'bbox_3d': row['bbox_3d'], 'top5': row['top5'],
                'category_hypotheses': retained, 'category_gate_audit': audit,
                'category_status': 'uncertain' if len(retained) != 1 else 'single_hypothesis'})
            counts.append(len(retained))
        value = {'version': VERSION, 'scene': scene, 'gate': gate,
            'source_predictions_sha256': marker['files']['predictions.json'],
            'cloud_sha256': sources[scene], 'objects': objects}
        path = output/scene/f'objects_top{top_k}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError('Category inputs changed; choose a new output directory')
        if not path.exists():
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
        load_category_objects(path)
        files[scene] = sha256(path)
    return {'version': VERSION, 'gate': gate, 'files': files,
            'source_predictions_sha256': marker['files']['predictions.json'],
            'retained_count_histogram': {str(i): counts.count(i) for i in range(6)},
            'mean_retained': float(np.mean(counts)), 'instances': len(counts)}
