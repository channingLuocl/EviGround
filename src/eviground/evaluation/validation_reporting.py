"""Evaluation-only answer checks and batch score aggregation."""
import hashlib
import json

from eviground.evaluation.batch_nr3d import summarize
from eviground.evaluation.run_storage import ensure_run_writable, write_json

def verify_source(manifest, source):
    expected = manifest['csv_sha256']
    if not source.exists():
        return {'path': str(source), 'expected_sha256': expected, 'status': 'missing'}
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError('NR3D CSV hash does not match the selection')
    return {'path': str(source), 'expected_sha256': expected,
            'actual_sha256': actual, 'status': 'verified'}


def evaluate(row, prediction, log, object_table='gt'):
    data = json.loads(prediction.read_text())
    if (data['query'] != row['query'] or data.get('object_table') != object_table or
            data.get('mode') != 'active_diagnostic'):
        raise ValueError(f'Prediction provenance mismatch: {prediction}')
    trace = data.get('active_trace', {})
    return {**row, 'object_table': object_table, 'predicted_id': data.get('predicted_id'),
            'review_failed': data.get('review_failed', False),
            'api_attempts': data.get('api_attempts', data.get('api_calls')),
            'call_usage': data.get('call_usage', []),
            'api_stage_calls': [{k: c.get(k) for k in ('stage', 'api_attempts', 'cache_hit', 'latency_s', 'usage', 'api_diagnostic')}
                for c in json.loads((prediction.parent / 'run.json').read_text()).get('calls', [])]
                if (prediction.parent / 'run.json').exists() else [],
            'recovery_attempts': data.get('recovery_attempts', []),
            'recovery_budget_exhausted': data.get('recovery_budget_exhausted', False),
            'review_findings': [f for review in trace.get('final_reviews', []) for f in review.get('findings', [])],
            'historical_reviews': trace.get('final_reviews', []),
            'prediction_source': data.get('prediction_source'),
            'implementation_version': data.get('implementation_version'),
            'backtrack_count': data.get('backtrack_count', 0), 'final_review': data.get('final_review'),
            'internal_budget_usage': data.get('internal_budget_usage'),
            'v4_api_calls': data.get('v4_api_calls', 0), 'v4_latency_s': data.get('v4_latency_s', 0),
            'review_triggered': data.get('review_triggered', False),
            'backtrack_target_restored': any(row['target_id'] in b['restored_ids'] for b in trace.get('backtracks', [])),
            'review_predictions': [{'predicted_id': r.get('predicted_id'),
                                    'correct': r.get('predicted_id') == row['target_id']}
                                   for r in trace.get('final_reviews', [])],
            'correct': data.get('predicted_id') == row['target_id'],
            'accepted': data.get('accepted'), 'decision_status': data.get('decision_status'),
            'missing': trace.get('final_evidence_state', {}).get('missing', []),
            'stop_reason': trace.get('stop_reason'), 'api_calls': data.get('api_calls'),
            'pipeline_latency_s': data.get('pipeline_latency_s'),
            'prediction_path': str(prediction), 'log': str(log)}


def report(output, results, total):
    ensure_run_writable(output)
    summary = {**summarize(results), 'selected_total': total,
               'errors': sum('error' in row for row in results),
               'accepted_wrong': sum(row.get('accepted') is True and not row['correct']
                                     for row in results)}
    groups = {}
    for key in ('stage', 'size_form', 'easy', 'view_dep'):
        if not any(key in r for r in results):
            continue
        groups[key] = {value: summarize([r for r in results if r.get(key, 'unknown') == value])
                       for value in sorted({r.get(key, 'unknown') for r in results})}
    versions = sorted({r.get('implementation_version') or r.get('source_implementation_version', 'unknown')
                       for r in results})
    groups['implementation_version'] = {version: summarize([
        r for r in results if (r.get('implementation_version') or
                              r.get('source_implementation_version', 'unknown')) == version])
        for version in versions}
    summary['mixed_implementation_versions'] = len(versions) > 1
    if (output / 'correct_cache_sync.json').exists():
        summary['evaluation_policy'] = 'cumulative_ever_correct'
    write_json(output / 'results.json', {'summary': summary, 'groups': groups,
               'results': sorted(results, key=lambda r: r['csv_row'])})
    return summary
