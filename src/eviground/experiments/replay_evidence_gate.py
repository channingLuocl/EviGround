"""Replay cached parses on real geometry without parser or visual API calls."""
import argparse
import json
from pathlib import Path

import numpy as np

from eviground.grounding.active_diagnostic import compute_observation
from eviground.grounding.diagnostic import diagnose, validate_structure
from eviground.grounding.evidence import match_retrieval_categories, validate_distance_relations
from eviground.query.query_rewrite import rewrite_query


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, default=Path('local_outputs/scene0011_00'))
    parser.add_argument('--output', type=Path,
                        default=Path('local_outputs/evidence_module_gate/replay.json'))
    args = parser.parse_args()
    objects = json.loads((args.scene / 'objects_gt.json').read_text())
    with np.load(args.scene / 'scene.npz', allow_pickle=False) as cloud:
        xyz, ids = cloud['xyz'], cloud['ids']
    room_center = (xyz.min(0) + xyz.max(0)) / 2
    sources = [Path(f'local_outputs/offline_object_table_smoke/q{number}_gt/run.json')
               for number in ('001', '003', '004', '006')]
    sources.append(Path('local_outputs/offline_diagnostic_smoke/run.json'))
    cases = []
    for source in sources:
        saved = json.loads(source.read_text())
        cases.append((source.stem + ':' + source.parent.name, saved['query'],
                      saved['parsed_query'], str(source), saved.get('query_source')))
    for query, attrs in (('Choose the largest cabinet.', ['largest']),
                         ('Choose the red largest cabinet.', ['red', 'largest']),
                         ('Choose the skinny cabinet.', ['skinny'])):
        cases.append(('constructed:' + query, query, {
            'Target': ['cabinet'], 'Anchor': [], 'Relations': [], 'Observer': {},
            'Attributes': [{'category': 'cabinet', 'phrase': attr} for attr in attrs],
            'Unhandled': [],
        }, None, 'constructed module request on real geometry; not an NR3D sample'))
    records = []
    for case_id, query, raw, source, source_kind in cases:
        parsed = validate_structure(query, validate_distance_relations(query, raw))
        parsed['query_rewrite'] = rewrite_query(query, parsed)
        candidates = match_retrieval_categories(parsed['Target'], objects)
        diagnosis = diagnose(parsed, objects, candidates, room_center,
                             trusted_categories=True, apply_size_constraints=False)
        diagnosis['size_comparison_pool_ids'] = [obj['bbox_id'] for obj in candidates]
        observations = {dimension: compute_observation(
            dimension, query, parsed, diagnosis, candidates, objects, xyz, ids)
            for dimension in ('position', 'distance', 'size', 'viewpoint', 'appearance')}
        records.append({'case_id': case_id, 'query': query, 'source': source,
                        'source_kind': source_kind, 'parsed_query': parsed,
                        'diagnosis': diagnosis, 'observations': observations})
    report = {
        'scene': args.scene.name, 'network_requests': 0,
        'category_source': 'annotated categories; geometry from prepared scene',
        'query_target_id_used': False,
        'limitations': [
            'Saved parser fixtures are reused; this does not validate a live parser.',
            'No visual verdict is supplied; color, material and shape remain unknown.',
            'Constructed size requests are module checks, not benchmark samples.',
            'This is not the per-module live acceptance batch or a benchmark accuracy result.',
        ],
        'cases': records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_value) + '\n')
    print(json.dumps({'output': str(args.output), 'cases': len(records),
                      'network_requests': 0,
                      'status_counts': {status: sum(r['diagnosis']['status'] == status for r in records)
                                        for status in ('resolved', 'missing', 'ambiguous', 'conflict')}},
                     ensure_ascii=False))


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'unsupported JSON value: {type(value).__name__}')


if __name__ == '__main__':
    main()
