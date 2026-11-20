"""Offline asset preflight and user-started sequential batch for the frozen direct-Qwen protocol."""
from __future__ import annotations
import argparse
from collections import Counter
import contextlib
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from eviground.evaluation.run_storage import ensure_run_writable, write_json
from eviground.evaluation.batch_progress import BatchProgress

SELECTION = ROOT / 'data/queries/nr3d_val.json'
ASSETS = ROOT / 'data/qwen_direct/full-assets.json'
SCENES = Path('/Volumes/KESU/论文/paper/reproductions/SeeGround/local_outputs')
EXPORT = Path('/Users/luocl/Desktop/seeground/outputs/20261020_r8_object_predictions')
CACHE = Path('/Volumes/KESU/EviGround/experiments/nr3d_dag_monitor_qwen_v5/runs/E0/key_slot_1/category_inputs')
OUTPUT = ROOT / 'outputs/qwen-direct/full-nr3d-20261120-v3'
EXPECTED_PREDICTIONS = '34aa94cbc455324fdd28ddfd21a3832bba743eac25c32d58037c9bc460b1da9e'
EXPECTED_CONFIG = 'de36aaaaea37c20062c3e4f9c9cc9b5fb039a810ddfa644b9a40f124dd6d6dba'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


class ConsoleLog:
    """Keep parent progress visible and tail-able without a shell pipe lifecycle."""
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, text):
        self.terminal.write(text); self.log.write(text)
        return len(text)

    def flush(self):
        self.terminal.flush(); self.log.flush()

    def isatty(self):
        return False


def inference_selection(path, expected=7805):
    # Discard target annotations during decoding; scoring loads them ONLY after
    # the corresponding prediction/failure has a saved immutable freeze receipt.
    def without_annotations(pairs):
        return {key: value for key, value in pairs if key != 'target_id'}
    data = json.loads(Path(path).read_text(), object_pairs_hook=without_annotations)
    rows = [{key: row[key] for key in ('scene', 'csv_row', 'query')} for row in data['rows']]
    if len(rows) != expected or len({r['csv_row'] for r in rows}) != expected:
        raise ValueError(f'Need exactly {expected} distinct csv_row entries, found {len(rows)}')
    if any(Path(r['scene']).name != r['scene'] or not r['query'].strip() for r in rows):
        raise ValueError('Invalid scene/query selection')
    return rows


def asset_preflight(selection, scene_root, cache_root, export, asset_file):
    import numpy as np
    from eviground.categories.export import load_export, prepare_tables
    from eviground.categories.hypotheses import load_category_objects
    report_path = asset_file.parent / 'full-asset-preflight.json'
    rows = inference_selection(selection)
    scenes = sorted({r['scene'] for r in rows})
    needed = [scene_root / s / name for s in scenes for name in ('scene.npz', 'scene_instances.json')]
    needed += [export / n for n in ('EXPORT_VERIFIED.json', 'predictions.json', 'config.json')]
    missing = [str(p) for p in needed if not p.is_file()]
    if missing:
        report = {'ready': False, 'queries': len(rows), 'scenes': len(scenes), 'missing_paths': missing,
                  'model_calls': 0, 'checked_utc': now()}
        save(report_path, report)
        raise ValueError(f'Preflight missing {len(missing)} paths; see {report_path}: {missing[:3]}')
    predictions, cloud_sources, hashes, config = load_export(export)
    if hashes['predictions.json'] != EXPECTED_PREDICTIONS or hashes['config.json'] != EXPECTED_CONFIG:
        raise ValueError('Category source differs from diagnostic003 frozen Top5 source; no silent frontend substitution')
    by_scene = {}
    for row in predictions:
        by_scene.setdefault(row['scene'], {})[row['bbox_id']] = row
    local_cache = asset_file.parent / 'full_categories'
    origin = {}; absent = []
    for scene in scenes:
        existing = cache_root / scene / 'objects_top5.json'
        target = local_cache / scene / 'objects_top5.json'
        if existing.is_file():
            value = read(existing)
            if (value['source_predictions_sha256'] != EXPECTED_PREDICTIONS or
                    value['source_config_sha256'] != EXPECTED_CONFIG or value['gate']['top_k'] != 5):
                raise ValueError(f'Wrong-source category cache: {existing}')
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != existing.read_bytes():
                raise ValueError(f'Local prepared cache changed: {target}')
            if not target.exists():
                shutil.copyfile(existing, target)
            origin[scene] = {'mode': 'reuse_complete_original_cache', 'source': str(existing), 'source_sha256': sha(existing)}
        else:
            absent.append(scene)
            origin[scene] = {'mode': 'materialize_verified_original_prediction_rows_without_inference', 'source': str(export)}
    if absent:
        prepare_tables(export, scene_root, local_cache, absent, 5)
    entries = {}; total_bytes = 0; objects_count = 0
    for i, scene in enumerate(scenes, 1):
        paths = {name: scene_root / scene / name for name in ('scene.npz', 'scene_instances.json')}
        cloud_hash = sha(paths['scene.npz'])
        if cloud_hash != cloud_sources[scene]:
            raise ValueError(f'Cloud hash does not match original classifier input: {scene}')
        table_path = local_cache / scene / 'objects_top5.json'
        table = read(table_path); objects = load_category_objects(table_path)
        if (table['cloud_sha256'] != cloud_hash or table['source_predictions_sha256'] != EXPECTED_PREDICTIONS
                or table['source_config_sha256'] != EXPECTED_CONFIG or table['gate']['top_k'] != 5):
            raise ValueError(f'Category/source/scene mismatch: {scene}')
        expected = by_scene[scene]
        if {o['bbox_id'] for o in table['objects']} != set(expected):
            raise ValueError(f'Category object coverage differs from full export: {scene}')
        if any(o['top5'] != expected[o['bbox_id']]['top5'] or o['bbox_3d'] != expected[o['bbox_id']]['bbox_3d'] for o in table['objects']):
            raise ValueError(f'Category rows or geometry differ from original predictions: {scene}')
        instances = read(paths['scene_instances.json'])
        instance_map = {o['bbox_id']: o['bbox_3d'] for o in instances}
        with np.load(paths['scene.npz'], allow_pickle=False) as cloud:
            xyz, rgb, ids = cloud['xyz'], cloud['rgb'], cloud['ids']
            if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != xyz.shape or ids.shape != (len(xyz),) or not np.isfinite(xyz).all():
                raise ValueError(f'Invalid point-cloud arrays: {scene}')
            known = set(map(int, np.unique(ids)))
        if not set(expected) <= known or any(instance_map.get(o['bbox_id']) != o['bbox_3d'] for o in table['objects']):
            raise ValueError(f'Category/point-cloud/instance geometry mismatch: {scene}')
        if not known - {-1} <= set(instance_map):
            raise ValueError(f'Missing instance geometry for raw points: {scene}')
        paths['category'] = table_path
        entries[scene] = {'paths': {k: str(v.resolve()) for k, v in paths.items()},
                          'hashes': {str(v.resolve()): (cloud_hash if k == 'scene.npz' else sha(v)) for k, v in paths.items()},
                          'category_origin': origin[scene], 'category_objects': len(objects)}
        total_bytes += paths['scene.npz'].stat().st_size; objects_count += len(objects)
        if i % 20 == 0 or i == len(scenes):
            print(f'[offline asset preflight] {i}/{len(scenes)} scenes verified; model_calls=0', flush=True)
    value = {'ready': True, 'selection_path': str(selection.resolve()), 'selection_sha256': sha(selection),
             'queries': len(rows), 'scenes': len(scenes), 'category_objects': objects_count,
             'scene_cloud_bytes_total': total_bytes, 'expected_predictions_sha256': EXPECTED_PREDICTIONS,
             'expected_config_sha256': EXPECTED_CONFIG, 'category_score_type': config['score_type'],
             'reused_category_scenes': len(scenes) - len(absent), 'materialized_scenes': absent,
             'scene_entries': entries, 'GT_supplied_to_workers': False, 'model_calls': 0,
             'public_scene_policy': 'Read-only shared originals, hash-checked each worker; one manifest, no per-query cloud copy.'}
    save(asset_file, value)
    save(report_path, {k: v for k, v in value.items() if k != 'scene_entries'})
    return value


def source_closure():
    paths = [*ROOT.glob('src/**/*.py'), ROOT / 'scripts/qwen_direct_fresh.py', Path(__file__),
             ROOT / 'scripts/run_qwen_direct_full.command',
             ROOT / 'data/qwen_direct/known-gt-exposure.json',
             ROOT / 'tests/unit/test_qwen_direct_fresh.py', ROOT / 'tests/unit/test_qwen_direct_batch.py',
             ROOT / 'tests/fixtures/qwen_direct_q00171_progress.json',
             ROOT / 'docs/qwen-direct-v3-loop-fix.md',
             ROOT / 'pyproject.toml', ROOT / 'docs/qwen-direct-protocol.md', ROOT / 'docs/qwen-direct-batch.md']
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def freeze_run(output, selection, assets_path):
    from qwen_direct_fresh import build_frozen_manifest
    assets = read(assets_path)
    if not assets['ready'] or assets['selection_sha256'] != sha(selection):
        raise ValueError('Missing/stale asset preflight; use --preflight first')
    rows = inference_selection(selection)
    if {r['scene'] for r in rows} != set(assets['scene_entries']):
        raise ValueError('Asset scene domain differs from selected7805 queries')
    missing = [path for entry in assets['scene_entries'].values() for path in entry['hashes'] if not Path(path).is_file()]
    if missing:
        raise ValueError(f'Asset not mounted or missing: {missing[:5]}')
    # Verify every referenced scene BEFORE any API request, even on resume.
    for i, (scene, entry) in enumerate(sorted(assets['scene_entries'].items()), 1):
        if any(sha(Path(path)) != expected for path, expected in entry['hashes'].items()):
            raise ValueError(f'Changed shared input: {scene}; no worker was started')
        if i % 20 == 0 or i == len(assets['scene_entries']):
            print(f'[startup input integrity] {i}/{len(assets["scene_entries"])} scenes; no inference yet', flush=True)
    closure = source_closure()
    protocol = build_frozen_manifest()
    # Single-sample local input IDs do not describe this run; shared asset hashes do.
    for key in ('source_closure', 'inputs', 'scheduler_sha256'):
        protocol.pop(key, None)
    protocol.update(candidate='qwen-direct-full-v3-historical-coverage-explicit-replanning',
                    scope='User-started full7805 Nr3D local exploration; diagnostic only, not official/paper score',
                    shared_scene_storage='Read-only public scene refs checked against asset hashes',
                    worker_isolation='One fresh subprocess per sample; workers=1; no shared monkeypatch/Monitor',
                    scoring='Parent evaluator after immutable per-sample prediction/failure freeze; null/errors count wrong',
                    development_sample={'csv_row': 12, 'scene': 'scene0565_00', 'GT_revealed_before_full_run': True})
    protocol['previously_gt_revealed_samples'] = read(ROOT/'data/qwen_direct/known-gt-exposure.json')['samples']
    protocol['diagnostic_trace_seen_without_gt'] = [{'csv_row': 171,
        'scope': 'v2 interrupted trace; no annotation read during v3 offline loop regression',
        'historically_gt_exposed_in_v1': True}]
    manifest = {'protocol': protocol, 'source_closure': closure, 'selection_sha256': sha(selection),
                'assets_sha256': sha(assets_path), 'python_executable': sys.executable,
                'python_prefix': sys.prefix,
                'workers': 1, 'total': len(rows)}
    path = output / 'manifest.json'
    if path.exists():
        if read(path) != manifest:
            raise ValueError('Frozen source/selection/assets/runtime changed; choose a new output. Old results not overwritten.')
        for name, expected in closure.items():
            if sha(output / 'source' / name) != expected:
                raise ValueError(f'Frozen source changed: {name}')
    else:
        if any(output.glob('q[0-9]*')):
            raise ValueError('Partial results without shared frozen manifest; refusing implicit resume')
        for name, expected in closure.items():
            target = output / 'source' / name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
            if sha(target) != expected:
                raise ValueError(f'Source changed during freeze: {name}')
        save(output / 'inference-selection.json', {'rows': rows, 'GT_fields': []})
        save(output / 'asset-index.json', assets)
        # Parent-only evaluation source, never a worker input or model message.
        target = output / 'parent-evaluation' / 'selection.json'; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(selection, target)
        save(path, manifest)
    if sha(output / 'parent-evaluation/selection.json') != manifest['selection_sha256'] or sha(output / 'asset-index.json') != manifest['assets_sha256']:
        raise ValueError('Shared frozen inputs changed')
    return manifest, rows, read(output / 'asset-index.json')


def call_accounting(directory):
    rows = []
    for file in sorted(directory.glob('calls/*/receipt.json')):
        try:
            row = read(file)
        except ValueError:
            return {'ambiguous_journal': True, 'client_attempts': None, 'http_responses': None, 'unknown_usage': [str(file)]}
        rows.append(row)
    return {'client_attempts': sum(bool(r.get('http_sent')) for r in rows),
            'http_responses': sum(r.get('http_status') is not None for r in rows),
            'unknown_usage': [r.get('tag') for r in rows if r.get('http_sent') and r.get('usage') is None],
            'remote_execution_unknown': [r.get('tag') for r in rows if r.get('http_sent') and r.get('http_status') is None],
            'request_errors': sum(r.get('status') == 'failed' for r in rows),
            'known_prompt_tokens': sum((r.get('usage') or {}).get('prompt_tokens', 0) for r in rows),
            'known_completion_tokens': sum((r.get('usage') or {}).get('completion_tokens', 0) for r in rows),
            'structural_repairs': len(list(directory.glob('controller-contracts/*/repair-*.json'))) + len(list(directory.glob('tentative-final/structural-rejection-*.json')))}


def parent_evaluate(row, directory, annotation_path, manifest_sha):
    prediction_path = directory / 'result.json'
    result = read(prediction_path)
    if result.get('status') == 'interrupted':
        raise ValueError('Interrupted sample is unfinished, not completed denominator; inspect before any retry')
    freeze_path = directory / 'prediction-freeze.json'
    freeze = {'result_sha256': sha(prediction_path), 'sample': row, 'shared_manifest_sha256': manifest_sha,
              'annotation_read': False}
    if freeze_path.exists() and read(freeze_path) != freeze:
        raise ValueError('Prediction/failure changed after freeze')
    if not freeze_path.exists():
        save(freeze_path, freeze)
    # Annotation decoding happens strictly after this sample's frozen state exists.
    annotation = [r for r in read(annotation_path)['rows'] if r['csv_row'] == row['csv_row']]
    if len(annotation) != 1 or any(annotation[0][k] != row[k] for k in ('scene', 'query')):
        raise ValueError('Annotation identity mismatch')
    target = int(annotation[0]['target_id'])
    shared = read(directory.parent/'manifest.json') if (directory.parent/'manifest.json').exists() else {}
    exposed = shared.get('protocol', {}).get('previously_gt_revealed_samples', [{'scene': 'scene0565_00', 'csv_row': 12}])
    valid = result['status'] in ('completed', 'tentative', 'abstained', 'tentative_abstained')
    predicted = result.get('predicted_id')
    return {'scene': row['scene'], 'csv_row': row['csv_row'], 'predicted_id': predicted, 'target_id': target,
            'correct': bool(valid and predicted is not None and predicted == target),
            'status': result['status'], 'strict': result.get('strict_monitor_accepted', False),
            'terminal_mode': result.get('terminal_mode', 'failed'), 'budgets': result.get('budgets', {}),
            'error': result.get('prediction', {}).get('failure'), 'usage': call_accounting(directory),
            'failure_type': result.get('prediction', {}).get('failure_type'),
            'prediction_freeze_sha256': sha(freeze_path), 'frozen_result_sha256': freeze['result_sha256'],
            'shared_manifest_sha256': manifest_sha,
            'previously_development_sample': {'scene': row['scene'], 'csv_row': row['csv_row']} in exposed,
            'full_semantic_evidence_sufficient': 'not_independently_audited_in_batch', 'formal_result': False}


class StopState:
    def __init__(self):
        self.requested = False; self.interrupts = 0

    def signal(self, _number, _frame):
        self.requested = True; self.interrupts += 1
        print('\n[STOP] Stop scheduling. First Ctrl+C drains current sample normally; second interrupts its worker. Saved records remain.', flush=True)


def worker_environment(command):
    environment = {k: v for k, v in os.environ.items() if not k.startswith(('EVIGROUND_', 'SEEGROUND_'))}
    environment.update(PYTHONPATH=str(Path(command[1]).parents[1] / 'src'), PYTHONUNBUFFERED='1',
                       PYTHON_DOTENV_DISABLED='1', PYTHONDONTWRITEBYTECODE='1')
    return environment


def run_worker(command, directory, stop):
    log = directory / 'console.log'
    environment = worker_environment(command)
    started = time.monotonic(); sent_interrupt = False; last_heartbeat = 0
    with log.open('w') as writer:
        process = subprocess.Popen(command, stdout=writer, stderr=subprocess.STDOUT, env=environment,
                                   cwd=Path(command[1]).parents[1], start_new_session=True)
        save(directory / 'worker-start.json', {'pid': process.pid, 'started_utc': now(), 'GT_input': False})
        with log.open() as reader:
            while process.poll() is None:
                text = reader.read()
                if text:
                    print(text, end='', flush=True)
                if stop.interrupts >= 2 and not sent_interrupt:
                    process.send_signal(signal.SIGINT); sent_interrupt = True
                if time.monotonic() - last_heartbeat >= 20:
                    live_path = directory / 'episode/live-progress.json'
                    state = read(live_path) if live_path.exists() else {}
                    print('[sample heartbeat] ' + json.dumps({'elapsed_s': round(time.monotonic()-started), **state}, ensure_ascii=False), flush=True)
                    last_heartbeat = time.monotonic()
                time.sleep(.2)
            tail = reader.read()
            if tail:
                print(tail, end='', flush=True)
    return process.returncode, time.monotonic() - started


def bootstrap_worker(output, rows, assets):
    """Same frozen worker/environment as full run, stopped before its first API."""
    row = rows[0]
    entry = assets['scene_entries'][row['scene']]
    request = {'sample': row, 'scene_dir': str(Path(entry['paths']['scene.npz']).parent),
               'category_path': entry['paths']['category'], 'shared_manifest': str(output/'manifest.json'),
               'input_hashes': entry['hashes']}
    directory = output/'offline-bootstrap'
    directory.mkdir(exist_ok=True)
    request_path = directory/'worker-input.json'
    if request_path.exists() and read(request_path) != request:
        raise ValueError('Bootstrap request identity changed')
    save(request_path, request)
    index = len(list(directory.glob('attempt-*'))) + 1
    attempt = directory/f'attempt-{index:03d}'; attempt.mkdir()
    command = [sys.executable, str(output/'source/scripts/qwen_direct_fresh.py'), '--run-dir', str(attempt/'episode'),
               '--worker-input', str(request_path), '--bootstrap-check']
    try:
        result = subprocess.run(command, cwd=output/'source', env=worker_environment(command),
                                capture_output=True, text=True)
        (attempt/'console.log').write_text(result.stdout + result.stderr)
        check_path = attempt/'episode/bootstrap-check.json'
        terminal = read(attempt/'episode/result.json') if (attempt/'episode/result.json').exists() else {}
        ready = result.returncode == 0 and check_path.exists() and read(check_path).get('ready') is True and terminal.get('status') == 'bootstrap_ready'
        report = {'ready': ready, 'worker_interpreter': sys.executable, 'worker_prefix': sys.prefix,
                  'shared_manifest_sha256': sha(output/'manifest.json'), 'attempt': str(attempt),
                  'sample': row, 'model_calls': 0,
                  'failure': terminal.get('prediction') if not ready else None,
                  'check': read(check_path) if check_path.exists() else None}
        if not ready and report['failure'] is None:
            report['failure'] = {'failure_type': 'WorkerBootstrapProcessError',
                                 'failure': (result.stderr or result.stdout)[-2000:]}
        if ready:
            if list((attempt/'episode').glob('calls/*/receipt.json')) or terminal.get('model_call_attempts') != 0:
                raise ValueError('Offline bootstrap unexpectedly recorded API attempts')
        save(output/'worker-bootstrap-status.json', report)
        if not ready:
            print('[FATAL BOOTSTRAP] ' + json.dumps(report, ensure_ascii=False), flush=True)
            raise ValueError(f'Real worker startup failed before API; no samples scheduled. See {attempt}/console.log')
        print('[offline real-worker bootstrap] dependencies/scene/category/parser/schema OK; first API boundary reached; HTTP=0', flush=True)
        return report
    except OSError as exc:
        save(output/'worker-bootstrap-status.json', {'ready': False, 'error_type': type(exc).__name__, 'error': str(exc), 'model_calls': 0})
        raise


def totals(results, total, state):
    statuses = Counter(r['status'] for r in results)
    return {'state': state, 'completed': len(results), 'total': total,
            'correct': sum(r['correct'] for r in results),
            'accuracy_completed': sum(r['correct'] for r in results) / len(results) if results else None,
            'strict': sum(bool(r['strict']) for r in results),
            'tentative': sum(r['terminal_mode'] == 'tentative' for r in results),
            'abstained': sum(r['predicted_id'] is None and r['status'] in ('abstained', 'tentative_abstained') for r in results),
            'failed': statuses['failed'],
            'client_attempts_known': sum(r['usage'].get('client_attempts') or 0 for r in results),
            'http_responses_known': sum(r['usage'].get('http_responses') or 0 for r in results),
            'structural_repairs': sum(r['usage'].get('structural_repairs', 0) for r in results),
            'request_errors': sum(r['usage'].get('request_errors', 0) for r in results),
            'unknown_usage_calls': sum(len(r['usage']['unknown_usage']) for r in results),
            'known_prompt_tokens': sum(r['usage'].get('known_prompt_tokens', 0) for r in results),
            'known_completion_tokens': sum(r['usage'].get('known_completion_tokens', 0) for r in results),
            'formal_result': False, 'includes_known_development_sample': True, 'updated_utc': now()}


def run_batch(output, manifest, rows, assets, worker_command=run_worker, evaluator=parent_evaluate, stop=None):
    stop = stop or StopState(); results = []; manifest_hash = sha(output / 'manifest.json')
    summary_path = output / 'summary.json'
    def checkpoint(state):
        summary = totals(results, len(rows), state)
        completed_ids = {r['csv_row'] for r in results}
        unfinished = []
        for row in rows:
            directory = output / f"q{row['csv_row']:05d}"
            if row['csv_row'] not in completed_ids and (directory/'worker-start.json').exists():
                unfinished.append({'scene': row['scene'], 'csv_row': row['csv_row'],
                                   'usage': call_accounting(directory/'episode')})
        summary['unfinished_cases'] = unfinished
        summary['unknown_usage_calls_including_unfinished'] = summary['unknown_usage_calls'] + sum(len(r['usage']['unknown_usage']) for r in unfinished)
        summary['known_prompt_tokens_including_unfinished'] = summary['known_prompt_tokens'] + sum(r['usage'].get('known_prompt_tokens', 0) for r in unfinished)
        summary['known_completion_tokens_including_unfinished'] = summary['known_completion_tokens'] + sum(r['usage'].get('known_completion_tokens', 0) for r in unfinished)
        save(summary_path, summary)
        save(output/'results.json', {'summary': summary, 'results': results})
        return summary
    for row in rows:
        directory = output / f"q{row['csv_row']:05d}"
        cached = directory / 'evaluation.json'
        if cached.exists():
            value = read(cached)
            if (value['shared_manifest_sha256'] != manifest_hash or value['scene'] != row['scene']
                    or value['csv_row'] != row['csv_row'] or sha(directory/'result.json') != value['frozen_result_sha256']
                    or sha(directory/'prediction-freeze.json') != value['prediction_freeze_sha256']):
                raise ValueError(f'Cached evaluation changed identity: {cached}')
            results.append(value)
    progress = BatchProgress(len(rows), initial_results=results)
    done_ids = {r['csv_row'] for r in results}
    checkpoint('running')
    startup_failures = [r for r in results if r['status'] == 'failed' and r['usage'].get('client_attempts') == 0]
    if startup_failures:
        checkpoint('attention_global_zero_api_startup_failure')
        print('[FATAL STARTUP previously saved; scheduling paused] ' + json.dumps({
            'csv_row': startup_failures[0]['csv_row'], 'error_type': startup_failures[0].get('failure_type'),
            'error': startup_failures[0].get('error'), 'instruction': 'Inspect and use a new frozen output after fixing global startup; never silently skip zero-API engineering failures.'}, ensure_ascii=False), flush=True)
        return results
    for row in rows:
        if stop.requested:
            break
        if row['csv_row'] in done_ids:
            continue
        directory = output / f"q{row['csv_row']:05d}"
        result_path = directory / 'result.json'
        directory.mkdir(parents=True, exist_ok=True)
        if not result_path.exists() and (directory/'episode/result.json').exists():
            shutil.copyfile(directory/'episode/result.json', result_path)
        if (directory / 'worker-start.json').exists() and not result_path.exists():
            checkpoint('attention_interrupted_worker')
            raise ValueError(f'Unfinished worker at {directory}; inspect receipts, no automatic retry of unknown work.')
        if result_path.exists() and read(result_path).get('status') == 'interrupted':
            checkpoint('attention_interrupted_worker')
            raise ValueError(f'Interrupted worker at {directory}; preserved, no blind re-send. Choose explicit new run only after inspection.')
        elapsed = 0
        if not result_path.exists():
            entry = assets['scene_entries'][row['scene']]
            request = {'sample': row, 'scene_dir': str(Path(entry['paths']['scene.npz']).parent),
                       'category_path': entry['paths']['category'], 'shared_manifest': str((output / 'manifest.json').resolve()),
                       'input_hashes': entry['hashes']}
            request_path = directory / 'worker-input.json'; save(request_path, request)
            command = [sys.executable, str(output / 'source/scripts/qwen_direct_fresh.py'), '--run-dir', str(directory/'episode'), '--worker-input', str(request_path)]
            # All episode artifacts live in episode; evaluator uses that same folder.
            episode = directory / 'episode'
            if episode.exists():
                raise ValueError(f'Unfinished artifacts at {episode}; refusing implicit overwrite or retry')
            save(directory/'worker-start.json', {'state': 'launching', 'started_utc': now(), 'GT_input': False})
            try:
                rc, elapsed = worker_command(command, directory, stop)
            except OSError as exc:
                rc = 127
                save(directory/'worker-launch-error.json', {'error_type': type(exc).__name__, 'error': str(exc), 'http_started': False})
            episode_result = episode / 'result.json'
            if episode_result.exists():
                # Parent-side result pointer is copied once, raw worker result never rewritten.
                shutil.copyfile(episode_result, result_path)
            elif stop.requested or rc == 130:
                checkpoint('stopped_unfinished')
                return results
            else:
                save(result_path, {'status': 'failed', 'predicted_id': None, 'strict_monitor_accepted': False,
                    'terminal_mode': 'failed', 'budgets': {}, 'prediction': {'failure_type': 'WorkerProcessError', 'failure': f'Worker exited{rc} before terminal result; raw console/receipts preserved'}})
        if read(result_path).get('status') == 'interrupted':
            checkpoint('stopped_unfinished')
            return results
        # Evaluate frozen parent pointer; account from episode receipts below.
        value = evaluator(row, directory, output/'parent-evaluation/selection.json', manifest_hash)
        value['usage'] = call_accounting(directory/'episode')
        value['elapsed_this_session_s'] = elapsed
        save(directory/'evaluation.json', value)
        results.append(value); done_ids.add(row['csv_row'])
        summary = checkpoint('stopping_after_current' if stop.requested else 'running')
        progress.update(value)
        details = {**summary, 'scene': row['scene'], 'csv_row': row['csv_row'], 'pred': value['predicted_id'],
                   'GT': value['target_id'], 'correct_this': value['correct'], 'BRT': value['budgets'],
                   'sample_usage': value['usage'], 'elapsed_s': round(elapsed, 1)}
        print(f"正确数/已完成数 {summary['correct']}/{summary['completed']} = {summary['accuracy_completed']:.2%}"
              f" | 已完成 {summary['completed']}/{len(rows)} | " + json.dumps(details, ensure_ascii=False), flush=True)
        with (output/'progress.jsonl').open('a') as log:
            log.write(json.dumps(details, ensure_ascii=False)+'\n'); log.flush(); os.fsync(log.fileno())
        if value['status'] == 'failed' and value['usage'].get('client_attempts') == 0:
            checkpoint('attention_global_zero_api_startup_failure')
            print('[FATAL STARTUP; scheduling paused] ' + json.dumps({
                'scene': row['scene'], 'csv_row': row['csv_row'],
                'error_type': value.get('failure_type'), 'error': value.get('error'),
                'API_attempts': 0, 'classification': 'engineering startup failure; not model-quality evidence'}, ensure_ascii=False), flush=True)
            return results
        if value['usage'].get('remote_execution_unknown') or value['usage'].get('ambiguous_journal'):
            checkpoint('attention_remote_execution_unknown')
            print('[ATTENTION] Unknown remote execution saved as failed/completed when terminal; no resend. Stopping new scheduling for inspection.', flush=True)
            return results
    checkpoint('stopped' if stop.requested else 'completed')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true', help='Offline only: verify7805 queries/scenes/categories and prepare missing category tables; no model calls')
    parser.add_argument('--prepare-only', action='store_true', help='Verify all input hashes and freeze shared output/source, but do NOT start workers or call any model')
    parser.add_argument('--manifest', type=Path, default=SELECTION)
    parser.add_argument('--scene-root', type=Path, default=SCENES)
    parser.add_argument('--category-cache', type=Path, default=CACHE)
    parser.add_argument('--prediction-export', type=Path, default=EXPORT)
    parser.add_argument('--assets-file', type=Path, default=ASSETS)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--workers', type=int, choices=[1], default=1, help='Sequential fresh subprocesses only; no shared global Monitor')
    args = parser.parse_args()
    if args.preflight:
        report = asset_preflight(args.manifest, args.scene_root, args.category_cache, args.prediction_export, args.assets_file)
        print(json.dumps({k: v for k, v in report.items() if k != 'scene_entries'}, ensure_ascii=False), flush=True)
        return
    output = args.output.resolve(); ensure_run_writable(output); output.mkdir(parents=True, exist_ok=True)
    with (output/'.batch.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another parent is already running this output; do not duplicate API requests')
        with (output/'runner.log').open('a', buffering=1) as logfile:
            with contextlib.redirect_stdout(ConsoleLog(sys.stdout, logfile)), contextlib.redirect_stderr(ConsoleLog(sys.stderr, logfile)):
                manifest, rows, assets = freeze_run(output, args.manifest, args.assets_file)
                bootstrap_worker(output, rows, assets)
                if args.prepare_only:
                    print(f'Prepared {len(rows)} samples; one shared frozen source, zero workers/model calls.', flush=True)
                    return
                stop = StopState(); old = signal.signal(signal.SIGINT, stop.signal)
                old_term = signal.signal(signal.SIGTERM, stop.signal)
                try:
                    run_batch(output, manifest, rows, assets, stop=stop)
                finally:
                    signal.signal(signal.SIGINT, old); signal.signal(signal.SIGTERM, old_term)


if __name__ == '__main__':
    main()
