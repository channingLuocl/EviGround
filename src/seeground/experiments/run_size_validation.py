"""Run a fixed NR3D manifest in isolated processes; answers are evaluation-only."""
from seeground.paths import PACKAGE_ROOT, source_files
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from seeground.rendering.run import ROOT
from seeground.grounding.v4_policy import settings as v4_settings, v5_enabled
from seeground.evaluation.run_storage import ensure_run_writable, write_json
from seeground.evaluation.validation_cache import (check_cached_table, correct_cache_path, _correct_result,
                              same_sample, load_correct_cache, save_correct_cache,
                              remember_correct_result, apply_correct_cache)
from seeground.evaluation.validation_reuse import (ORCHESTRATION_FILES, GT_V71_EQUIVALENT_HASHES,
                              V73_LEGACY_GT_HASHES, comparable_settings, reuse_results)
from seeground.evaluation.validation_reporting import evaluate, report, verify_source

# Keep the historical public imports above for existing launchers and tests.
# Cache/reuse/reporting implementations live in their own modules.


def parse_bool(value):
    if value.lower() not in ('true', 'false'):
        raise argparse.ArgumentTypeError('Expected true or false')
    return value.lower() == 'true'


def run_case(row, scene_root, output, review_limit, retry_errors=False, retry_null=False,
             retry_false=False, object_table='gt'):
    ensure_run_writable(output)
    directory = output / f"q{row['csv_row']:05d}"
    result_path = directory / 'evaluation.json'
    latest_result_path = None
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        check_cached_table(saved, object_table)
        if any(saved.get(key) != value for key, value in row.items()):
            raise ValueError(f'Cached annotation mismatch: {result_path}')
        if _correct_result(row, saved):
            return saved
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
        from seeground.grounding.v4_policy import v5_enabled
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
            command.append('--gt=' + str(object_table == 'gt').lower())
            with log.open('a' if retry_errors else 'w') as handle:
                completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                           start_new_session=True)
            if completed.returncode:
                raise ValueError(f'Inference process exit code {completed.returncode}; see {log}')
        result = evaluate(row, prediction, log, object_table)
    except (OSError, ValueError, KeyError) as exc:
        run_path = directory / 'run.json'
        try:
            failed_run = json.loads(run_path.read_text()) if run_path.exists() else {}
        except (OSError, ValueError):
            failed_run = {}
        from seeground.grounding.v5_runtime import actual_attempts
        from seeground.api.qwen_api import _redact
        result = {**row, 'object_table': object_table,
                  'implementation_version': v4_settings()['version'],
                  'predicted_id': None, 'correct': False, 'accepted': False, 'error': _redact(str(exc)),
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
                result = {**evaluate(row, prediction, log, object_table), **{key: result[key] for key in
                    ('error', 'inference_error', 'error_stage')}}
            except (OSError, ValueError, KeyError):
                pass
    write_json(result_path, result)
    if latest_result_path is not None:
        write_json(latest_result_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path,
                        default=ROOT / 'data/validation/nr3d_size_300.json')
    parser.add_argument('--scene-root', type=Path, default=ROOT / 'local_outputs')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--reuse-from', type=Path, help='Import a completed compatible batch without API calls')
    parser.add_argument('--reuse-previous-version', action='store_true',
                        help='Explicitly import the verified v7.1 GT baseline into v7.3, preserving version provenance')
    parser.add_argument('--prepare-only', action='store_true',
                        help='Verify inputs/settings and import cached results without running inference')
    parser.add_argument('--gt', nargs='?', const=True, default=False, type=parse_bool,
                        help='Use GT category labels (default: false, author-provided predicted labels); '
                             'accepts --gt, --gt=true or --gt=false')
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
    if args.output is not None:
        ensure_run_writable(args.output)
    if args.reuse_previous_version and not args.reuse_from:
        parser.error('--reuse-previous-version requires --reuse-from')
    object_table = 'gt' if args.gt else 'pred'
    objects_file = 'objects_gt.json' if args.gt else 'objects.json'
    if (args.review_limit < 1 or args.workers < 1 or
            (args.limit is not None and args.limit < 1) or
            (args.api_concurrency is not None and args.api_concurrency < 1)):
        parser.error('workers, review-limit, api-concurrency and limit must be positive')
    if sum(bool(flag) for flag in (args.retry_null, args.retry_errors, args.retry_false)) > 1:
        parser.error('Choose one of --retry-null, --retry-errors or --retry-false')
    if (args.retry_null or args.retry_false) and args.output is None:
        parser.error('--retry-null/--retry-false require --output pointing to an existing run')
    if args.worker:
        from seeground.grounding.real_scene import run_query
        run_query(args.scene_dir, args.query, args.output,
                  query_source=f'Official NR3D original utterance; CSV row {args.csv_row}',
                  mode='active_diagnostic', object_table=object_table, review_limit=args.review_limit)
        return
    manifest = json.loads(args.manifest.read_text())
    rows = manifest['rows']
    selected = rows[:args.limit] if args.limit else rows
    missing = [str(args.scene_root / scene / name)
               for scene in sorted({r['scene'] for r in rows})
               for name in ('scene.npz', objects_file, 'scene_instances.json', 'provenance.json')
               if not (args.scene_root / scene / name).exists()]
    if missing:
        raise ValueError('Prepare scenes before inference. Missing files: ' + ', '.join(missing[:10]))
    for scene in {r['scene'] for r in selected}:
        objects = json.loads((args.scene_root / scene / objects_file).read_text())
        known = {obj['bbox_id'] for obj in objects}
        if any(not set(r['stimulus_candidate_ids']).issubset(known)
               for r in selected if r['scene'] == scene):
            raise ValueError(f'NR3D candidate IDs missing in {scene}')
    source = ROOT / 'local_outputs/downloads/nr3d_official.csv'
    from seeground.api.qwen_api import api_config_summary, api_concurrency
    from seeground.query.query_syntax import syntax_settings
    source_check = verify_source(manifest, source)
    output = (args.output or ROOT / 'local_outputs' /
              ('nr3d_size_300_' + datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    ensure_run_writable(output)
    if (args.retry_null or args.retry_false) and not (output / 'manifest.json').exists():
        parser.error('--retry-null/--retry-false require an existing run with manifest.json')
    if args.api_concurrency is not None:
        os.environ['SEEGROUND_API_CONCURRENCY'] = str(args.api_concurrency)
    os.environ.setdefault('SEEGROUND_API_SLOT_DIR', str(output / '.api_slots'))
    ensure_run_writable(Path(os.environ['SEEGROUND_API_SLOT_DIR']))
    output.mkdir(parents=True, exist_ok=True)
    Path(os.environ['SEEGROUND_API_SLOT_DIR']).mkdir(parents=True, exist_ok=True)
    settings = {'selection': manifest, 'review_limit': args.review_limit,
        'workers': args.workers, 'api_concurrency': api_concurrency(),
        'api_slot_dir': os.environ['SEEGROUND_API_SLOT_DIR'],
        'query_syntax': syntax_settings(),
        'api_config': api_config_summary(),
        'scene_root': str(args.scene_root.resolve()), 'object_table': object_table,
        'mode': 'active_diagnostic',
        'v4': v4_settings(),
        'models': {key: os.getenv(key, '') for key in (
            'SEEGROUND_REACT_BACKEND', 'SEEGROUND_JEV_MODEL', 'SEEGROUND_MODEL',
            'SEEGROUND_BASE_URL')},
        'code_sha256': {path.relative_to(PACKAGE_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in sorted(source_files())
                        if not path.name.startswith(('test_', '._'))}}
    if v5_enabled():
        from seeground.grounding.v5_runtime import file_identity
        settings['input_sha256'] = {scene: file_identity(args.scene_root / scene / name for name in
            ('scene.npz', objects_file, 'scene_instances.json', 'provenance.json')) for scene in sorted({r['scene'] for r in rows})}
    settings_path = output / 'manifest.json'
    if settings_path.exists():
        previous_settings = json.loads(settings_path.read_text())
        if comparable_settings(previous_settings) != comparable_settings(settings):
            raise ValueError('Output has different code/settings/selection; choose a fresh output directory')
    if args.reuse_from:
        reuse_results(args.reuse_from, output, selected, settings, args.reuse_previous_version)
    correct_cache = load_correct_cache(args.scene_root, manifest, object_table)
    cache_update = apply_correct_cache(correct_cache, output, rows)
    write_json(output / 'correct_cache_sync.json', {
        'cache_file': str(correct_cache_path(args.scene_root, object_table)),
        **cache_update})
    print(f'Correct cache: {cache_update["cached_correct_rows"]} successes retained; '
          f'{len(cache_update["imported_rows"])} imported, '
          f'{len(cache_update["replaced_rows"])} incorrect results replaced.',
          file=sys.stderr, flush=True)
    write_json(output / 'csv_verification.json', source_check)
    if args.retry_null or args.retry_false:
        audit = output / ('false_retry_runs' if args.retry_false else 'null_retry_runs') / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        audit.mkdir(parents=True)
        write_json(audit / 'manifest.json', settings)
    elif not settings_path.exists():
        write_json(settings_path, settings)
    results = []
    pending = []
    selected_ids = {r['csv_row'] for r in selected}
    for row in rows:
        cached = output / f"q{row['csv_row']:05d}" / 'evaluation.json'
        if cached.exists():
            saved = json.loads(cached.read_text())
            check_cached_table(saved, object_table)
            if any(saved.get(key) != value for key, value in row.items()):
                raise ValueError(f'Cached annotation mismatch: {cached}')
            results.append(saved)
            if row['csv_row'] in selected_ids and not _correct_result(row, saved) and (
                    (args.retry_false and saved.get('correct') is False) or
                    (args.retry_null and saved.get('predicted_id') is None) or
                    (args.retry_errors and (saved.get('error') or saved.get('review_failed')))):
                pending.append(row)
        elif row['csv_row'] in selected_ids and not (args.retry_null or args.retry_false):
            pending.append(row)
    if args.retry_null or args.retry_false:
        write_json(audit / 'selection.json', {'csv_rows': [r['csv_row'] for r in pending]})
    report(output, results, len(rows))
    if args.prepare_only:
        print(f'Prepared {len(rows)} cases: {len(results)} cached, {len(pending)} pending; '
              'no inference started.', file=sys.stderr, flush=True)
        return
    from seeground.evaluation.batch_progress import BatchProgress
    pending_ids = {r['csv_row'] for r in pending}
    row_by_id = {row['csv_row']: row for row in pending}
    completed = [r for r in results if r['csv_row'] not in pending_ids]
    progress = BatchProgress(len(completed) + len(pending), initial_results=completed)
    last_report = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_case, row, args.scene_root.resolve(), output, args.review_limit,
                               args.retry_errors, args.retry_null, args.retry_false, object_table)
                   for row in pending]
        try:
            for future in as_completed(futures):
                result = future.result()
                remember_correct_result(correct_cache, args.scene_root,
                                        row_by_id[result['csv_row']], result, output)
                results = [r for r in results if r['csv_row'] != result['csv_row']]
                results.append(result)
                if time.monotonic() - last_report >= 30:
                    report(output, results, len(rows))
                    last_report = time.monotonic()
                progress.update(result)
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            print('\nStopping new cases; waiting for active cases to save checkpoints. '
                  'Run the same command to resume.', file=sys.stderr, flush=True)
    # Include workers that finished while handling an interrupt.
    for row in pending:
        cached = output / f"q{row['csv_row']:05d}/evaluation.json"
        if cached.exists():
            results = [r for r in results if r['csv_row'] != row['csv_row']]
            saved = json.loads(cached.read_text())
            results.append(saved)
            remember_correct_result(correct_cache, args.scene_root, row, saved, output)
    report(output, results, len(rows))


if __name__ == '__main__':
    main()
