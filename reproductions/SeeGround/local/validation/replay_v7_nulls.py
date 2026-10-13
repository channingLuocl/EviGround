"""Offline fallback audit using saved parser/visual evidence, not a v7 API evaluation.

Run with PYTHONPATH=local SEEGROUND_V7=1 .venv/bin/python
local/validation/replay_v7_nulls.py --baseline ... --output ...
Only old nulls receive the new fallback; old non-null answers are held fixed.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from diagnostic import diagnose, validate_structure
from real_scene import select_objects, structural_objects
from v4_policy import v7_enabled
from v7_evidence import supported_choice
from wall_geometry import attach_wall_geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('local/validation/v5_development_68.json'))
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--scene-root', type=Path, default=Path('local_outputs'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not v7_enabled():
        parser.error('SEEGROUND_V7=1 is required')
    manifest = json.loads(args.manifest.read_text())
    cache, results, unavailable = {}, [], []
    for row in manifest['rows']:
        sample = args.baseline / f"q{row['csv_row']:05d}"
        if not (sample / 'run.json').exists() or not (sample / 'prediction.json').exists():
            unavailable.append(row['csv_row'])
            continue
        run = json.loads((sample / 'run.json').read_text())
        prediction = json.loads((sample / 'prediction.json').read_text())
        scene = row['scene']
        if scene not in cache:
            path = args.scene_root / scene
            objects = json.loads((path / 'objects_gt.json').read_text())
            with np.load(path / 'scene.npz', allow_pickle=False) as cloud:
                xyz, ids = cloud['xyz'], cloud['ids']
                geometry = attach_wall_geometry(objects + structural_objects(path, objects), xyz, ids)
                cache[scene] = objects, geometry, (xyz.min(0) + xyz.max(0)) / 2
        objects, geometry, center = cache[scene]
        saved = run['parsed_query']
        parsed = validate_structure(row['query'], {key: deepcopy(saved.get(key, {} if key == 'Observer' else []))
            for key in ('Target', 'Anchor', 'Attributes', 'Relations', 'Observer', 'Unhandled')})
        candidates = select_objects(parsed['Target'], objects)
        checks = prediction.get('active_trace', {}).get('final_checks', [])
        diagnosis = diagnose(parsed, geometry, candidates, center, checks,
                             trusted_categories=True, apply_size_constraints=False, query=row['query'])
        inference = supported_choice(row['query'], parsed, geometry, candidates, diagnosis, checks)
        proposed = inference['selected_id'] if prediction.get('predicted_id') is None else prediction['predicted_id']
        # Evaluation fields are accessed only after inference has finished.
        results.append({'csv_row': row['csv_row'], 'scene': scene, 'query': row['query'],
            'baseline_id': prediction.get('predicted_id'), 'fallback_id': inference['selected_id'],
            'offline_proposed_id': proposed, 'correct': proposed == row['target_id'],
            'baseline_correct': prediction.get('predicted_id') == row['target_id'], 'inference': inference})
    summary = {'method': 'saved parser and visual evidence; fallback on prior nulls only; not a v7 API batch',
        'samples': len(results), 'unavailable_csv_rows': unavailable,
        'baseline_correct': sum(r['baseline_correct'] for r in results),
        'offline_proposed_correct': sum(r['correct'] for r in results),
        'null_to_correct': sum(r['baseline_id'] is None and r['correct'] for r in results),
        'null_to_wrong': sum(r['baseline_id'] is None and r['offline_proposed_id'] is not None and not r['correct'] for r in results),
        'remaining_null': sum(r['offline_proposed_id'] is None for r in results)}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'results.json').write_text(json.dumps({'summary': summary, 'rows': results}, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
