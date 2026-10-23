"""Adapt the verified r8 export without reading its evaluation_only payloads."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from seeground.categories.categories_v9 import (R8_VERSION, R8_SCORE_TYPE, gate_categories,
                           load_category_objects, sha256)

EXPORT_VERSION = 'pointnext-sampling-vote-r8-export-v1'


def load_export(source):
    source = Path(source)
    if source.is_dir():
        def read(name):
            return (source/name).read_bytes()
        archive = None
    else:
        archive = ZipFile(source)
        markers = [n for n in archive.namelist()
                   if n.endswith('/EXPORT_VERIFIED.json') or n == 'EXPORT_VERIFIED.json']
        if len(markers) != 1:
            archive.close()
            raise ValueError('Expected one r8 export root in ZIP')
        prefix = markers[0][:-len('EXPORT_VERIFIED.json')]
        def read(name):
            return archive.read(prefix+name)
    try:
        marker = json.loads(read('EXPORT_VERIFIED.json'))
        if marker.get('status') != 'passed':
            raise ValueError('Need a verified r8 export')
        payloads = {}
        for name in ('predictions.json', 'config.json'):
            raw = read(name)
            if hashlib.sha256(raw).hexdigest() != marker['files'][name]:
                raise ValueError(f'Verified r8 file changed: {name}')
            payloads[name] = json.loads(raw)
    finally:
        if archive is not None:
            archive.close()
    data, config = payloads['predictions.json'], payloads['config.json']
    if (data.get('version') != EXPORT_VERSION or config.get('version') != EXPORT_VERSION
            or data['summary'].get('score_type') != R8_SCORE_TYPE
            or config.get('score_type') != R8_SCORE_TYPE
            or not data['summary'].get('complete') or data['summary'].get('errors') != 0):
        raise ValueError('Need a complete r8 export with standardized fusion scores')
    rows = data['results']
    sources = {r['scene']: r['cloud_sha256'] for r in config['sources']}
    keys = {(r['scene'], r['bbox_id']) for r in rows}
    if (not rows or len(keys) != len(rows) or len(sources) != len(config['sources'])
            or len(rows) != data['summary']['count']
            or len(sources) != data['summary']['scenes']
            or {r['scene'] for r in rows} != set(sources)):
        raise ValueError('r8 object/scene coverage is inconsistent')
    for row in rows:
        if (row.get('status') != 'completed' or row.get('score_type') != R8_SCORE_TYPE
                or type(row['bbox_id']) is not int
                or row['predicted_object'] != row['top5'][0]['category']):
            raise ValueError('Invalid r8 prediction row')
        gate_categories(row['top5'], score_type=R8_SCORE_TYPE)
    return rows, sources, marker['files']


def prepare_r8_tables(source, scene_root, output, scenes, top_k, min_score, max_gap):
    if min_score is not None or max_gap is not None:
        raise ValueError('r8 uses rank-only Top-1/Top-5; omit cosine score/gap thresholds')
    rows, sources, hashes = load_export(source)
    gate = {'top_k': top_k, 'score_type': R8_SCORE_TYPE}
    files, counts = {}, []
    for scene in sorted(scenes):
        if Path(scene).name != scene or scene in ('.', '..') or scene not in sources:
            raise ValueError(f'Unknown r8 scene: {scene}')
        cloud_path = scene_root/scene/'scene.npz'
        if sha256(cloud_path) != sources[scene]:
            raise ValueError(f'Point cloud differs from r8 input: {scene}')
        with np.load(cloud_path, allow_pickle=False) as cloud:
            known = set(map(int, np.unique(cloud['ids'])))
        objects = []
        for row in sorted((r for r in rows if r['scene'] == scene), key=lambda r: r['bbox_id']):
            if row['bbox_id'] not in known:
                raise ValueError(f'Unknown classified instance: {scene}/{row["bbox_id"]}')
            retained, audit = gate_categories(row['top5'], **gate)
            objects.append({'bbox_id': row['bbox_id'], 'target': row['predicted_object'],
                'bbox_3d': row['bbox_3d'], 'top5': row['top5'],
                'category_hypotheses': retained, 'category_gate_audit': audit,
                'category_status': 'single_hypothesis' if len(retained) == 1 else 'uncertain'})
            counts.append(len(retained))
        value = {'version': R8_VERSION, 'scene': scene, 'gate': gate,
            'source_predictions_sha256': hashes['predictions.json'],
            'source_config_sha256': hashes['config.json'],
            'cloud_sha256': sources[scene], 'objects': objects}
        path = output/scene/f'objects_top{top_k}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError('Category inputs changed; choose a new output directory')
        if not path.exists():
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
        load_category_objects(path)
        files[scene] = sha256(path)
    return {'version': R8_VERSION, 'source_version': EXPORT_VERSION,
        'gate': gate, 'files': files, 'source_predictions_sha256': hashes['predictions.json'],
        'source_config_sha256': hashes['config.json'],
        'retained_count_histogram': {str(i): counts.count(i) for i in range(6)},
        'mean_retained': float(np.mean(counts)), 'instances': len(counts)}
