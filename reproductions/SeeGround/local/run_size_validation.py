"""Run a fixed NR3D manifest in isolated processes; answers are evaluation-only."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from batch_nr3d import summarize
from run import ROOT
from v4_policy import settings as v4_settings, v5_enabled


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def verify_source(manifest, source):
    expected = manifest['csv_sha256']
    if not source.exists():
        return {'path': str(source), 'expected_sha256': expected, 'status': 'missing'}
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError('NR3D CSV hash does not match the selection')
    return {'path': str(source), 'expected_sha256': expected,
            'actual_sha256': actual, 'status': 'verified'}


def evaluate(row, prediction, log):
    data = json.loads(prediction.read_text())
    if (data['query'] != row['query'] or data.get('object_table') != 'gt' or
            data.get('mode') != 'active_diagnostic'):
        raise ValueError(f'Prediction provenance mismatch: {prediction}')
    trace = data.get('active_trace', {})
    return {**row, 'predicted_id': data.get('predicted_id'),
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


def run_case(row, scene_root, output, review_limit, retry_errors=False, retry_null=False,
             retry_false=False):
    directory = output / f"q{row['csv_row']:05d}"
    result_path = directory / 'evaluation.json'
    latest_result_path = None
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        if any(saved.get(key) != value for key, value in row.items()):
            raise ValueError(f'Cached annotation mismatch: {result_path}')
        if retry_null or retry_false:
            if saved.get('predicted_id') is not None and not retry_false:
                return saved
            # Manual retries get a fresh recovery budget and keep all earlier
            # artifacts, including checkpoints from failed attempts.
            latest_result_path = result_path
            retries = directory / ('false_retries' if retry_false else 'null_retries')
            retries.mkdir(exist_ok=True)
            attempt = 1
            while (retries / f'attempt_{attempt:03d}').exists():
                attempt += 1
            directory = retries / f'attempt_{attempt:03d}'
            directory.mkdir()
            write_json(directory / 'previous_evaluation.json', saved)
            result_path = directory / 'evaluation.json'
        elif not (retry_errors and (saved.get('error') or saved.get('review_failed'))):
            return saved
    directory.mkdir(parents=True, exist_ok=True)
    prediction = directory / 'prediction.json'
    log = directory / 'console.log'
    try:
        from v4_policy import v5_enabled
        resume_review = False
        if v5_enabled() and retry_errors and result_path.exists():
            saved = json.loads(result_path.read_text())
            if saved.get('recovery_budget_exhausted'):
                return saved
            if saved.get('error') and not (directory / 'pending_inference.json').exists():
                return saved
            if saved.get('review_failed') and (directory / 'pending_review.json').exists():
                resume_review = True
        if not prediction.exists() or resume_review:
            # Target/stimulus IDs are deliberately absent from the inference process arguments.
            command = [sys.executable, str(Path(__file__).resolve()), '--worker',
                       '--scene-dir', str(scene_root / row['scene']), '--query', row['query'],
                       '--csv-row', str(row['csv_row']), '--output', str(directory),
                       '--review-limit', str(review_limit)]
            with log.open('a' if retry_errors else 'w') as handle:
                completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise ValueError(f'Inference process exit code {completed.returncode}; see {log}')
        result = evaluate(row, prediction, log)
    except (OSError, ValueError, KeyError) as exc:
        run_path = directory / 'run.json'
        try:
            failed_run = json.loads(run_path.read_text()) if run_path.exists() else {}
        except (OSError, ValueError):
            failed_run = {}
        from v5_runtime import actual_attempts
        from qwen_api import _redact
        result = {**row, 'predicted_id': None, 'correct': False, 'accepted': False, 'error': _redact(str(exc)),
                  'inference_error': True, 'error_stage': failed_run.get('error_stage'),
                  'api_attempts': sum(actual_attempts(c) for c in failed_run.get('calls', [])),
                  'pipeline_latency_s': failed_run.get('pipeline_latency_s'),
                  'call_usage': [c.get('usage') for c in failed_run.get('calls', [])],
                  'api_stage_calls': [{k: c.get(k) for k in ('stage', 'api_attempts', 'cache_hit', 'latency_s', 'usage', 'api_diagnostic')}
                      for c in failed_run.get('calls', [])],
                  'recovery_budget_exhausted': failed_run.get('recovery_budget_exhausted', False),
                  'recovery_attempts': failed_run.get('recovery_attempts', []),
                  'prediction_path': str(prediction), 'log': str(log)}
        if v5_enabled() and prediction.exists() and (directory / 'pending_review.json').exists():
            try:
                # A failed resume or process crash cannot erase an already
                # saved prediction. Keep the failure visible in evaluation.
                result = {**evaluate(row, prediction, log), **{key: result[key] for key in
                    ('error', 'inference_error', 'error_stage')}}
            except (OSError, ValueError, KeyError):
                pass
    write_json(result_path, result)
    if latest_result_path is not None:
        write_json(latest_result_path, result)
    return result


def report(output, results, total):
    summary = {**summarize(results), 'selected_total': total,
               'errors': sum('error' in row for row in results),
               'accepted_wrong': sum(row.get('accepted') is True and not row['correct']
                                     for row in results)}
    groups = {}
    for key in ('stage', 'size_form'):
        groups[key] = {value: summarize([r for r in results if r.get(key) == value])
                       for value in sorted({r.get(key, 'unknown') for r in results})}
    write_json(output / 'results.json', {'summary': summary, 'groups': groups,
               'results': sorted(results, key=lambda r: r['csv_row'])})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path,
                        default=ROOT / 'local/validation/nr3d_size_300.json')
    parser.add_argument('--scene-root', type=Path, default=ROOT / 'local_outputs')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--api-concurrency', type=int,
                        help='Maximum simultaneous model requests; local workers remain independent')
    parser.add_argument('--review-limit', type=int, default=3)
    parser.add_argument('--limit', type=int, help='Run the first N; omit to resume the whole selection')
    parser.add_argument('--retry-errors', action='store_true', help='Retry cached failed API/process runs')
    parser.add_argument('--retry-null', action='store_true',
                        help='Rerun cached null predictions from scratch, preserving earlier attempts; skip existing IDs')
    parser.add_argument('--retry-false', action='store_true',
                        help='Rerun every cached incorrect evaluation, including nulls and wrong IDs; latest result wins')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--scene-dir', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--query', help=argparse.SUPPRESS)
    parser.add_argument('--csv-row', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.review_limit < 1 or args.workers < 1 or
            (args.limit is not None and args.limit < 1) or
            (args.api_concurrency is not None and args.api_concurrency < 1)):
        parser.error('workers, review-limit, api-concurrency and limit must be positive')
    if sum(bool(flag) for flag in (args.retry_null, args.retry_errors, args.retry_false)) > 1:
        parser.error('Choose one of --retry-null, --retry-errors or --retry-false')
    if (args.retry_null or args.retry_false) and args.output is None:
        parser.error('--retry-null/--retry-false require --output pointing to an existing run')
    if args.worker:
        from real_scene import run_query
        run_query(args.scene_dir, args.query, args.output,
                  query_source=f'Official NR3D original utterance; CSV row {args.csv_row}',
                  mode='active_diagnostic', object_table='gt', review_limit=args.review_limit)
        return
    manifest = json.loads(args.manifest.read_text())
    rows = manifest['rows']
    selected = rows[:args.limit] if args.limit else rows
    missing = [str(args.scene_root / scene / name)
               for scene in sorted({r['scene'] for r in selected})
               for name in ('scene.npz', 'objects_gt.json', 'scene_instances.json', 'provenance.json')
               if not (args.scene_root / scene / name).exists()]
    if missing:
        raise ValueError('Prepare scenes before inference. Missing files: ' + ', '.join(missing[:10]))
    for scene in {r['scene'] for r in selected}:
        objects = json.loads((args.scene_root / scene / 'objects_gt.json').read_text())
        known = {obj['bbox_id'] for obj in objects}
        if any(not set(r['stimulus_candidate_ids']).issubset(known)
               for r in selected if r['scene'] == scene):
            raise ValueError(f'NR3D candidate IDs missing in {scene}')
    source = ROOT / 'local_outputs/downloads/nr3d_official.csv'
    from qwen_api import api_config_summary, api_concurrency
    from query_syntax import syntax_settings
    source_check = verify_source(manifest, source)
    output = (args.output or ROOT / 'local_outputs' /
              ('nr3d_size_300_' + datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    if (args.retry_null or args.retry_false) and not (output / 'manifest.json').exists():
        parser.error('--retry-null/--retry-false require an existing run with manifest.json')
    if args.api_concurrency is not None:
        os.environ['SEEGROUND_API_CONCURRENCY'] = str(args.api_concurrency)
    os.environ.setdefault('SEEGROUND_API_SLOT_DIR', str(output / '.api_slots'))
    output.mkdir(parents=True, exist_ok=True)
    Path(os.environ['SEEGROUND_API_SLOT_DIR']).mkdir(parents=True, exist_ok=True)
    settings = {'selection': manifest, 'review_limit': args.review_limit,
        'workers': args.workers, 'api_concurrency': api_concurrency(),
        'api_slot_dir': os.environ['SEEGROUND_API_SLOT_DIR'],
        'query_syntax': syntax_settings(),
        'api_config': api_config_summary(),
        'scene_root': str(args.scene_root.resolve()), 'object_table': 'gt',
        'mode': 'active_diagnostic',
        'v4': v4_settings(),
        'models': {key: os.getenv(key, '') for key in (
            'SEEGROUND_REACT_BACKEND', 'SEEGROUND_JEV_MODEL', 'SEEGROUND_MODEL',
            'SEEGROUND_BASE_URL')},
        'code_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in sorted((ROOT / 'local').glob('*.py'))
                        if not path.name.startswith(('test_', '._'))}}
    if v5_enabled():
        from v5_runtime import file_identity
        settings['input_sha256'] = {scene: file_identity(args.scene_root / scene / name for name in
            ('scene.npz', 'objects_gt.json', 'scene_instances.json', 'provenance.json')) for scene in sorted({r['scene'] for r in rows})}
    settings_path = output / 'manifest.json'
    if settings_path.exists():
        previous_settings = json.loads(settings_path.read_text())
        if args.retry_null or args.retry_false:
            # Only the batch orchestrator may differ when adding null retries
            # to an existing run. Inference code, inputs and settings must match.
            previous_compare = {**previous_settings, 'code_sha256': {
                k: v for k, v in previous_settings['code_sha256'].items() if k != Path(__file__).name}}
            current_compare = {**settings, 'code_sha256': {
                k: v for k, v in settings['code_sha256'].items() if k != Path(__file__).name}}
            compatible = previous_compare == current_compare
        else:
            compatible = previous_settings == settings
        if not compatible:
            raise ValueError('Output has different code/settings/selection; choose a fresh output directory')
    write_json(output / 'csv_verification.json', source_check)
    if args.retry_null or args.retry_false:
        audit = output / ('false_retry_runs' if args.retry_false else 'null_retry_runs') / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        audit.mkdir(parents=True)
        write_json(audit / 'manifest.json', settings)
    else:
        write_json(settings_path, settings)
    results = []
    pending = selected
    if args.retry_null or args.retry_false:
        pending = []
        selected_ids = {r['csv_row'] for r in selected}
        for row in rows:
            cached = output / f"q{row['csv_row']:05d}" / 'evaluation.json'
            if not cached.exists():
                continue
            saved = json.loads(cached.read_text())
            if any(saved.get(key) != value for key, value in row.items()):
                raise ValueError(f'Cached annotation mismatch: {cached}')
            results.append(saved)
            if row['csv_row'] in selected_ids and (
                    (args.retry_false and saved.get('correct') is False) or
                    (args.retry_null and saved.get('predicted_id') is None)):
                pending.append(row)
    if args.retry_null or args.retry_false:
        write_json(audit / 'selection.json', {'csv_rows': [r['csv_row'] for r in pending]})
        report(output, results, len(rows))
    from batch_progress import BatchProgress
    progress = BatchProgress(len(pending))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_case, row, args.scene_root.resolve(), output, args.review_limit,
                               args.retry_errors, args.retry_null, args.retry_false)
                   for row in pending]
        for future in as_completed(futures):
            result = future.result()
            if args.retry_null or args.retry_false:
                results = [r for r in results if r['csv_row'] != result['csv_row']]
            results.append(result)
            report(output, results, len(rows))
            progress.update(result)
    report(output, results, len(rows))


if __name__ == '__main__':
    main()
