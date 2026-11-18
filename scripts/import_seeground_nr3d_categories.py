"""Adapt archived native SeeGround scores to the unchanged Nr3D instance geometry."""

import argparse
import json
from pathlib import Path

import numpy as np

from eviground.categories.export import NATIVE_EXPORT_VERSION
from eviground.categories.hypotheses import SEEGROUND_SCORE_TYPE, gate_categories, sha256
from eviground.evaluation.run_storage import write_json


def import_predictions(source, scene_root, output):
    source, scene_root, output = Path(source), Path(scene_root), Path(output)
    if output.exists():
        raise ValueError('Choose a fresh output directory for the native Nr3D export')
    predictions = json.loads((source / 'predictions_top5.json').read_text())
    vocabulary = json.loads((source / 'vocabulary.json').read_text())
    matrix_path = source / 'raw_scores/native_logits_3413x197.npy'
    frozen = json.loads((source / 'provenance/RESULTS_FROZEN.json').read_text())
    if (predictions.get('score_type') != SEEGROUND_SCORE_TYPE
            or sha256(matrix_path) != predictions['source_scores_sha256']
            or sha256(matrix_path) != frozen['files']['eval/V0.npy']):
        raise ValueError('Need the archived native similarity scores, without fusion or calibration')
    matrix = np.load(matrix_path, allow_pickle=False)
    objects = predictions['objects']
    if (len(vocabulary) != 197 or len(set(vocabulary)) != 197
            or matrix.shape != (len(objects), len(vocabulary)) or not np.isfinite(matrix).all()):
        raise ValueError('Native score rows or vocabulary are not aligned')
    keys = [(o['scene_id'], o['object_id']) for o in objects]
    if len(set(keys)) != len(keys) or not keys:
        raise ValueError('Duplicate or empty SeeGround instance keys')
    ranking = np.argsort(-matrix, axis=1, kind='stable')[:, :5]
    rows, sources = [], []
    for j, obj in enumerate(objects):
        expected = [dict(category=vocabulary[int(i)], rank=k, score=float(matrix[j, i]))
                    for k, i in enumerate(ranking[j], 1)]
        if obj['top5'] != expected or obj['top1'] != expected[0]['category']:
            raise ValueError(f'Original score/rank alignment changed at row {j}')
        gate_categories(expected, score_type=SEEGROUND_SCORE_TYPE)
        rows.append(dict(scene=obj['scene_id'], bbox_id=obj['object_id'], bbox_3d=obj['bbox_3d'],
            status='completed', predicted_object=obj['top1'], top5=expected, score_type=SEEGROUND_SCORE_TYPE))
    for scene in sorted({o['scene_id'] for o in objects}):
        if Path(scene).name != scene or scene in ('.', '..'):
            raise ValueError('Invalid scene name')
        selected = [o for o in objects if o['scene_id'] == scene]
        scene_source = source / 'scenes' / (scene + '.json')
        per_scene = json.loads(scene_source.read_text())
        if (per_scene['scene_id'] != scene or per_scene['score_type'] != SEEGROUND_SCORE_TYPE
                or per_scene['objects'] != selected):
            raise ValueError(f'Per-scene and aggregate native predictions differ: {scene}')
        author_path = source / 'author_top1_lookup_tables' / (scene + '.json')
        author = json.loads(author_path.read_text())
        original = {o['bbox_id']: o['bbox_3d'] for o in author}
        classified = {o['object_id']: o['bbox_3d'] for o in selected}
        directory = scene_root / scene
        candidate_path = directory / 'objects.json'
        current_candidates = {o['bbox_id']: o['bbox_3d'] for o in json.loads(candidate_path.read_text())}
        full_path = directory / 'scene_instances.json'
        full = {o['bbox_id']: o['bbox_3d'] for o in json.loads(full_path.read_text())}
        with np.load(directory / 'scene.npz', allow_pickle=False) as cloud:
            known = set(map(int, np.unique(cloud['ids'])))
        if (len(original) != len(author) or original != classified or current_candidates != original
                or not set(classified) <= known
                or any(i not in full or not np.allclose(full[i], box, atol=1e-5, rtol=0)
                       for i, box in classified.items())):
            raise ValueError(f'Original candidate IDs or geometry differ: {scene}')
        sources.append(dict(scene=scene, cloud_sha256=sha256(directory / 'scene.npz'),
            instances_sha256=sha256(full_path), candidate_ids=sorted(classified),
            candidate_geometry_sha256=sha256(candidate_path),
            author_olt_sha256=sha256(author_path), prediction_scene_sha256=sha256(scene_source)))
        print(json.dumps(dict(scene=scene, completed=len(sources), instances=len(selected))), flush=True)
    protocol = dict(source='seeground_pointnext_clip_reproduction', score_type=SEEGROUND_SCORE_TYPE,
        source_prediction_sha256=sha256(source / 'predictions_top5.json'),
        source_scores_sha256=sha256(matrix_path), vocabulary_sha256=sha256(source / 'vocabulary.json'),
        tie_break='stable vocabulary index', geometry='original GT instance points and author OLT boxes',
        qualification='Archived baseline reproduction, not assumed identical to author Top-1 labels',
        evaluation_only_read=False, query_answers_read=False, model_rerun=False)
    output.mkdir(parents=True)
    write_json(output / 'frontend_protocol.json', protocol)
    config = dict(version=NATIVE_EXPORT_VERSION, category_source=protocol['source'],
        score_type=SEEGROUND_SCORE_TYPE, sources=sources,
        frontend_protocol_sha256=sha256(output / 'frontend_protocol.json'))
    write_json(output / 'config.json', config)
    write_json(output / 'predictions.json', dict(version=NATIVE_EXPORT_VERSION,
        summary=dict(complete=True, errors=0, count=len(rows), scenes=len(sources),
                     score_type=SEEGROUND_SCORE_TYPE), results=rows))
    write_json(output / 'EXPORT_VERIFIED.json', dict(status='passed', files={
        name: sha256(output / name) for name in ('config.json', 'predictions.json')}))
    return dict(scenes=len(sources), instances=len(rows), original_candidates_preserved=True,
                native_scores_preserved=True, **protocol)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--scene-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = import_predictions(args.source, args.scene_root, args.output)
    write_json(args.output / 'IMPORT_COMPLETE.json', summary)
