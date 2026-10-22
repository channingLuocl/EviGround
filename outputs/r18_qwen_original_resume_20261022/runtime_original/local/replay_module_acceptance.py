"""Replay saved module parses on real geometry without fresh model evidence."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from active_diagnostic import apply_size_observation, compute_observation
from diagnostic import (diagnose, recover_between_relations, recover_distance_relations,
                        refresh_condition_records, validate_structure, viewpoint_evidence)
from evidence import match_retrieval_categories, validate_distance_relations
from query_rewrite import rewrite_query
from qwen_api import QwenAPIError, parse_json_object
from real_scene import repair_missing_attributes, structural_objects
from run import ROOT, load_objects
from wall_geometry import attach_wall_geometry


def replay_saved_parse(query, saved):
    """Revalidate raw parser output and replay only available attribute repairs."""
    calls = saved.get('calls', [])
    starts = [index for index, call in enumerate(calls) if call.get('stage') == 'parse_query']
    if not starts:
        raise ValueError('Saved raw parse_query response is unavailable')
    start = starts[-1]
    parse_call = calls[start]
    try:
        raw = parse_json_object(parse_call['response'])
    except QwenAPIError:
        raw = None
    syntax_repair = None
    if raw is None:
        repairs = [call for call in calls[start + 1:]
                   if call.get('stage') == 'parse_query_json_repair']
        if parse_call.get('finish_reason') != 'stop' or not repairs or repairs[-1].get('finish_reason') != 'stop':
            raise ValueError('Saved parser syntax repair is unavailable or incomplete')
        syntax_repair = repairs[-1]
        raw = parse_json_object(syntax_repair['response'])
    for key in ('Target', 'Anchor'):
        value = raw.get(key)
        if not isinstance(value, (list, str)) or (isinstance(value, list)
                and not all(isinstance(name, str) for name in value)):
            raise ValueError('Saved parser ' + key + ' must contain category strings')
    parsed = validate_distance_relations(query, deepcopy(raw))
    parsed = recover_distance_relations(query, parsed)
    parsed = recover_between_relations(query, parsed)
    parsed = validate_structure(query, parsed)
    effective_call = syntax_repair or parse_call
    replay_calls = []
    attribute_repairs = [call for call in calls[start + 1:]
                         if call.get('stage') == 'parse_query_attribute_repair']
    repair_needed = any(item.startswith(('appearance condition not represented:',
                                        'appearance condition role not represented:'))
                        for item in parsed.get('diagnostic_missing', []))
    if effective_call.get('finish_reason') == 'stop' and repair_needed and attribute_repairs:
        repair = attribute_repairs[-1]
        response = {'_content': repair['response'], '_finish_reason': repair.get('finish_reason'),
                    '_baseline': {'latency_s': 0}, 'usage': None}
        with patch('real_scene.chat', return_value=response):
            parsed = repair_missing_attributes(query, parsed, replay_calls)
    parsed['query_rewrite'] = rewrite_query(query, parsed)
    if effective_call.get('finish_reason') == 'length':
        parsed['diagnostic_missing'].append('query parser response truncated')
        refresh_condition_records(query, parsed)
    return parsed, {
        'source': 'saved raw parser response; current validation and rewrite',
        'raw_parser_output': raw,
        'syntax_repair_reused': syntax_repair is not None,
        'saved_attribute_repair_available': bool(attribute_repairs),
        'saved_attribute_repair_replayed': bool(replay_calls),
        'attribute_repair_unavailable': repair_needed and not attribute_repairs,
        'attribute_repair_calls': replay_calls,
        'network_requests': 0,
    }


def replay_inference(query, saved, scene_dir, include_size=False):
    parsed, parser_replay = replay_saved_parse(query, saved)
    objects = load_objects(scene_dir / 'objects_gt.json')
    with np.load(scene_dir / 'scene.npz', allow_pickle=False) as cloud:
        xyz, ids = cloud['xyz'], cloud['ids']
    geometry_objects = attach_wall_geometry(
        objects + structural_objects(scene_dir, objects), xyz, ids)
    matched = match_retrieval_categories(parsed['Target'], objects)
    candidates = matched or objects
    diagnosis = diagnose(parsed, geometry_objects, candidates,
                         (xyz.min(0) + xyz.max(0)) / 2,
                         trusted_categories=True, apply_size_constraints=False, query=query)
    initial_ids = sorted(obj['bbox_id'] for obj in candidates)
    diagnosis['size_comparison_pool_ids'] = initial_ids
    result = {
        'parser_replay': parser_replay,
        'parsed_query': parsed,
        'category_retrieval_ids': sorted(obj['bbox_id'] for obj in matched),
        'initial_candidate_ids': initial_ids,
        'initial_diagnosis': diagnosis,
        'viewpoint_evidence': viewpoint_evidence(parsed, diagnosis),
        'saved_visual_evidence_reused': False,
    }
    if include_size:
        result['size_observation'] = compute_observation(
            'size', query, parsed, diagnosis, candidates, geometry_objects, xyz, ids)
        result['size_diagnosis'] = apply_size_observation(
            deepcopy(diagnosis), result['size_observation'])
        result['size_observation_applied'] = True
        result['joint_measurements'] = (result['size_observation']['comparison_metrics']
                                       if result['size_observation'].get('joint_size_diagnosis') is not None else None)
    return result


def replay_case(row, saved_root, scene_root, include_size=False):
    directory = saved_root / f"q{row['csv_row']:05d}"
    saved = json.loads((directory / 'run.json').read_text())
    if saved['query'] != row['query']:
        raise ValueError('Saved query does not match the selection: ' + str(directory))
    prior_path = directory / 'module_evaluation.json'
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else {}
    trace = saved.get('active_diagnostic', {})
    result = {
        'csv_row': row['csv_row'], 'module': row.get('module'),
        'modules': row.get('modules', [row.get('module')]),
        'scene': row['scene'], 'query': row['query'], 'source': str(directory / 'run.json'),
        'previous': {
            'decision_status': prior.get('decision_status', saved.get('decision_status')),
            'accepted': prior.get('accepted'), 'correct': prior.get('correct'),
            'predicted_id': prior.get('predicted_id'),
            'error': prior.get('error', saved.get('error')),
            'initial_diagnosis': trace.get('initial_diagnosis', {}),
            'parsed_query': saved.get('parsed_query', {}),
        },
    }
    result['current'] = replay_inference(row['query'], saved, scene_root / row['scene'], include_size)
    # Evaluation annotations are accessed only after inference has returned.
    result['evaluation_only'] = {
        'target_id': row['target_id'],
        'target_in_initial_candidates': row['target_id'] in result['current']['initial_candidate_ids'],
        'target_in_current_candidates': row['target_id'] in
            result['current'].get('size_diagnosis', result['current']['initial_diagnosis'])['candidate_ids'],
    }
    return result


def replay_batch(saved_root, scene_root, include_size=False):
    manifest = json.loads((saved_root / 'manifest.json').read_text())
    rows = manifest.get('selection', manifest)['rows']
    cases = []
    for row in rows:
        try:
            cases.append(replay_case(row, saved_root, scene_root, include_size))
        except (ValueError, OSError, KeyError) as exc:
            cases.append({'csv_row': row['csv_row'], 'module': row.get('module'),
                          'scene': row['scene'], 'query': row['query'], 'replay_error': str(exc)})
    return {
        'saved_root': str(saved_root), 'scene_root': str(scene_root),
        'network_requests': 0, 'query_target_id_used_for_inference': False,
        'saved_visual_evidence_reused': False,
        'limitations': [
            'Current results are initial diagnoses and optional computed size diagnoses, not live acceptance results.',
            'Raw saved parser outputs are reused; changed parser prompts are not validated.',
            'Only saved attribute repairs can be replayed; unavailable repairs remain missing.',
            'Saved visual checks and planner choices are not reused after role validation changes.',
            'Optional size observations use the active shared gate; joint height/relationship hypotheses are preserved.',
            'No planner finish decision or fresh visual verification is replayed; a resolved computed diagnosis is not live acceptance.',
        ],
        'summary': {
            'total': len(cases), 'replay_errors': sum('replay_error' in case for case in cases),
            'status_counts': dict(Counter(case['current']['initial_diagnosis']['status']
                                         for case in cases if 'current' in case)),
            'computed_size_status_counts': dict(Counter(case['current']['size_diagnosis']['status']
                for case in cases if 'size_diagnosis' in case.get('current', {}))),
        },
        'cases': cases,
    }


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError('Unsupported JSON value: ' + type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--saved-root', type=Path, default=ROOT / 'local_outputs' /
                        'evidence_module_acceptance_25_20261007_v1')
    parser.add_argument('--scene-root', type=Path, default=ROOT / 'local_outputs')
    parser.add_argument('--output', type=Path, default=ROOT / 'local_outputs' /
                        'evidence_module_acceptance_25_20261007_v1_offline_fix/replay.json')
    parser.add_argument('--include-size', action='store_true',
                        help='Compute frozen-pool size evidence and apply the shared active gate without model calls')
    args = parser.parse_args()
    report = replay_batch(args.saved_root, args.scene_root, args.include_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=json_value) + '\n')
    print(json.dumps({'output': str(args.output), 'network_requests': 0, **report['summary']},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
