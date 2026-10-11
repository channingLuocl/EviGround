"""Fresh real NR3D module samples using the natural active evidence planner."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
from pathlib import Path

from appearance_evidence import required_visual_attributes
from batch_nr3d import summarize
from diagnostic import viewpoint_evidence
from qwen_api import _settings, api_provider, api_config_summary
from react_decision import validate_react_config
from run import ROOT
from run_size_validation import run_case, verify_source, write_json

MODULES = ('position', 'size', 'distance', 'viewpoint', 'appearance')


def case_modules(row):
    values = row.get('modules', [row.get('module')])
    if not isinstance(values, list) or not values or any(value not in MODULES for value in values):
        raise ValueError('Every sample needs module or modules from the five evidence dimensions')
    return list(dict.fromkeys(values))


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def preflight(manifest, scene_root, review_limit):
    from query_syntax import syntax_settings
    rows = manifest['rows']
    if not rows or len({row['csv_row'] for row in rows}) != len(rows):
        raise ValueError('Sample CSV rows must be nonempty and unique')
    for row in rows:
        case_modules(row)
        if not isinstance(row.get('query'), str) or not row['query'].strip():
            raise ValueError('Sample query must be nonempty')
    base, model, _ = _settings()
    backend = validate_react_config()
    source_check = verify_source(manifest, ROOT / 'local_outputs/downloads/nr3d_official.csv')
    hashes = {}
    for scene in sorted({row['scene'] for row in rows}):
        directory = scene_root / scene
        names = ('scene.npz', 'objects_gt.json', 'provenance.json')
        absent = [str(directory / name) for name in names if not (directory / name).is_file()]
        if absent:
            raise ValueError('Prepare scenes before inference. Missing files: ' + ', '.join(absent))
        known = {obj['bbox_id'] for obj in json.loads((directory / 'objects_gt.json').read_text())}
        for row in rows:
            if row['scene'] == scene and (row['target_id'] not in known or
                    not set(row.get('stimulus_candidate_ids', [])).issubset(known)):
                raise ValueError('Evaluation annotation IDs absent from scene ' + scene)
        hashes[scene] = {name: digest(directory / name) for name in names}
        structural = directory / 'scene_instances.json'
        hashes[scene]['scene_instances.json'] = (digest(structural)
                                                  if structural.is_file() else None)
    provider = api_provider()
    models = {'decision_backend': backend, 'visual_backend': provider,
              provider + '_model': model, provider + '_base_url': base,
              'api_config': api_config_summary()}
    if backend == 'jev':
        from decision_api import JEV_URL, jev_model
        models.update(jev_model=jev_model(), jev_url=JEV_URL)
    return {'selection': manifest, 'review_limit': review_limit,
            'query_syntax': syntax_settings(),
            'mode': 'active_diagnostic', 'object_table': 'gt',
            'query_target_id_supplied_to_model': False,
            'gt_categories_supplied_to_model': True,
            'scene_root': str(scene_root.resolve()), 'models': models,
            'code_sha256': {path.name: digest(path) for path in sorted((ROOT / 'local').glob('*.py'))
                            if not path.name.startswith('test_')},
            'input_sha256': hashes, 'csv_verification': source_check}


def bind_output(output, settings):
    path = output / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != settings:
        raise ValueError('Output has different code/settings/selection/input hashes; choose a fresh output')
    output.mkdir(parents=True, exist_ok=True)
    write_json(path, settings)


def parsed_dimensions(parsed):
    dimensions = set()
    for attribute in parsed.get('Attributes', []):
        if required_visual_attributes([attribute.get('phrase', '')]):
            dimensions.add('appearance')
    for condition in parsed.get('condition_records', []):
        kind = condition.get('condition_type', '')
        if kind in ('size', 'size_scope'):
            dimensions.add('size')
        elif kind.startswith('viewpoint:'):
            dimensions.add('viewpoint')
    for relation in parsed.get('Relations', []):
        kind = relation.get('relation')
        dimensions.add('distance' if kind in ('nearest', 'farthest', 'near', 'far') else 'position')
        if kind in ('left', 'right', 'front', 'back'):
            dimensions.add('viewpoint')
    if parsed.get('Viewpoint'):
        dimensions.add('viewpoint')
    return sorted(dimensions)


def execution_audit(parsed, trace, diagnosis):
    initial = trace.get('initial_diagnosis', {})
    geometries = [('initial_diagnosis', initial.get('geometry', {})),
                  ('final_diagnosis', diagnosis.get('geometry', {}))]
    result = {}
    for module in MODULES:
        requests, operations, precomputed = [], [], []
        for index, step in enumerate(trace.get('steps', []), 1):
            action, observation = step.get('action', {}), step.get('observation', {})
            if action.get('dimension') == module:
                requests.append(index)
            if observation.get('dimension') == module and not observation.get('no_evidence_acquired'):
                operations.append({'step': index, 'operation_status': observation.get('operation_status'),
                    'coverage': observation.get('coverage', {}),
                    'candidate_evidence': observation.get('candidate_evidence', []),
                    'attribute_coverage': observation.get('attribute_coverage', []),
                    'missing': observation.get('missing', [])})
        for source, geometry in geometries:
            if module == 'position':
                records = (geometry.get('relation_pairs', []) +
                           geometry.get('between_triplets', []) +
                           geometry.get('direction_evidence', []))
                if records:
                    precomputed.append({'source': source, 'predicate_records': len(records),
                        'checked_ids': sorted({row['target_id'] for row in records
                                               if isinstance(row.get('target_id'), int)}),
                        'unknown_ids': sorted({row['target_id'] for row in records
                                               if isinstance(row.get('target_id'), int)
                                               and row.get('passes') is None})})
            elif module == 'distance':
                rankings, ignored = geometry.get('distance_rankings', []), geometry.get('distance_ignored', [])
                proximity = geometry.get('proximity_pairs', [])
                if rankings or ignored or proximity:
                    precomputed.append({'source': source, 'rankings': rankings,
                                        'ignored': ignored, 'proximity_pairs': proximity})
            elif module == 'viewpoint':
                frames = viewpoint_evidence(parsed, initial if source == 'initial_diagnosis' else diagnosis)
                if frames['status'] != 'not_applicable':
                    precomputed.append({'source': source, **frames})
        result[module] = {'parsed': module in parsed_dimensions(parsed),
            'tool_requested': bool(requests), 'requested_steps': requests,
            'tool_executed': bool(operations), 'operations': operations,
            'precomputed_evidence': precomputed,
            'exercised': bool(operations or precomputed)}
    return result


def final_audit_missing(trace, diagnosis):
    """Return all explicit blockers, including unresolved final ambiguity."""
    final_state = trace.get('final_evidence_state') or {}
    reasons = []
    for source in (final_state.get('missing', []), diagnosis.get('missing', [])):
        for reason in source or []:
            if reason not in reasons:
                reasons.append(reason)
    status = final_state.get('status') or diagnosis.get('status')
    candidates = (final_state.get('remaining_candidate_ids') or
                  diagnosis.get('candidate_ids') or [])
    if status in ('ambiguous', 'missing') and len(candidates) > 1:
        reason = 'multiple candidates remain under unresolved evidence: ' + str(candidates)
        if reason not in reasons:
            reasons.append(reason)
    elif status == 'ambiguous' and candidates:
        reasons.append('remaining candidate lacks joint support across legal hypotheses: ' + str(candidates))
    return reasons


def audit_case(row, scene_root, output, review_limit, retry_errors=False):
    result = run_case(row, scene_root, output, review_limit, retry_errors)
    directory = output / f"q{row['csv_row']:05d}"
    run_path = directory / 'run.json'
    run = json.loads(run_path.read_text()) if run_path.exists() else {}
    prediction_path = directory / 'prediction.json'
    prediction = json.loads(prediction_path.read_text()) if prediction_path.exists() else {}
    parsed = run.get('parsed_query', {})
    trace = prediction.get('active_trace') or run.get('active_diagnostic', {})
    diagnosis = prediction.get('diagnosis') or trace.get('diagnosis_after_step') or trace.get('initial_diagnosis', {})
    calls = run.get('calls', [])
    result.update(module_audit=execution_audit(parsed, trace, diagnosis),
        parsed_dimensions=parsed_dimensions(parsed),
        condition_records=parsed.get('condition_records', []),
        condition_records_are_parse_audit=True,
        parser_missing=parsed.get('diagnostic_missing', []),
        final_missing=final_audit_missing(trace, diagnosis),
        pending_size_conditions=diagnosis.get('pending_size_conditions', []),
        visual_needs=diagnosis.get('visual_needs', []),
        recorded_api_calls=len(calls), api_stages=dict(Counter(call.get('stage', 'unknown') for call in calls)),
        candidate_retrieval={'initial_ids': trace.get('category_candidates', []),
                             'target_in_initial_candidates': row['target_id'] in trace.get('category_candidates', [])},
        run_path=str(run_path))
    write_json(directory / 'module_evaluation.json', result)
    return result


def summarize_cases(results):
    return {**summarize(results), 'errors': sum('error' in row for row in results),
        'accepted_wrong': sum(row.get('accepted') is True and not row.get('correct') for row in results),
        'recorded_api_calls': sum(row.get('recorded_api_calls', 0) for row in results),
        'api_stages': dict(sum((Counter(row.get('api_stages', {})) for row in results), Counter()))}


def report(output, results, expected_rows):
    groups = {}
    for module in MODULES:
        rows = [row for row in results if module in case_modules(row)]
        groups[module] = {**summarize_cases(rows),
            'expected_total': sum(module in case_modules(row) for row in expected_rows),
            'parsed_cases': sum(row.get('module_audit', {}).get(module, {}).get('parsed', False) for row in rows),
            'tool_executed_cases': sum(row.get('module_audit', {}).get(module, {}).get('tool_executed', False) for row in rows),
            'exercised_cases': sum(row.get('module_audit', {}).get(module, {}).get('exercised', False) for row in rows)}
    summary = {**summarize_cases(results), 'expected_total': len(expected_rows)}
    write_json(output / 'results.json', {'summary': summary, 'modules': groups,
        'results': sorted(results, key=lambda row: row['csv_row'])})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'local/validation/nr3d_module_acceptance_25.json')
    parser.add_argument('--scene-root', type=Path, default=ROOT / 'local_outputs')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--review-limit', type=int, default=6)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--retry-errors', action='store_true')
    parser.add_argument('--module', action='append', choices=MODULES,
                        help='Run only this module; repeat for several modules (default: all five)')
    args = parser.parse_args()
    if args.workers < 1 or args.review_limit < 1:
        parser.error('workers and review-limit must be positive')
    manifest = json.loads(args.manifest.read_text())
    selected_modules = list(dict.fromkeys(args.module or MODULES))
    settings = preflight(manifest, args.scene_root, args.review_limit)
    selected_rows = [row for row in manifest['rows']
                     if set(case_modules(row)) & set(selected_modules)]
    output = (args.output or ROOT / 'local_outputs' /
              ('evidence_module_acceptance_' + datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    bind_output(output, settings)
    print(json.dumps({'cases': len(selected_rows), 'expected_total': len(selected_rows),
        'selected_modules': selected_modules, 'module_counts': dict(Counter(
        module for row in selected_rows for module in case_modules(row)
        if module in selected_modules)),
        'models': settings['models'], 'csv_status': settings['csv_verification']['status'],
        'output': str(output), 'dry_run': args.dry_run}, ensure_ascii=False), flush=True)
    if args.dry_run:
        return
    saved = {}
    for row in manifest['rows']:
        path = output / f"q{row['csv_row']:05d}" / 'module_evaluation.json'
        if path.exists():
            result = json.loads(path.read_text())
            if any(result.get(key) != value for key, value in row.items()):
                raise ValueError('Cached annotation mismatch: ' + str(path))
            saved[row['csv_row']] = result
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(audit_case, row, args.scene_root.resolve(), output,
                               args.review_limit, args.retry_errors) for row in selected_rows]
        for future in as_completed(futures):
            result = future.result()
            saved[result['csv_row']] = result
            report(output, list(saved.values()), manifest['rows'])
            print(json.dumps({key: result.get(key) for key in ('csv_row', 'module', 'scene', 'prediction_source',
                'predicted_id', 'target_id', 'accepted', 'decision_status', 'final_missing', 'error')},
                ensure_ascii=False), flush=True)
    print(json.dumps(report(output, list(saved.values()), manifest['rows']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
