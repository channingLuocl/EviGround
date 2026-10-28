"""Manual null-only retries, with separate outputs and frozen r18 inference."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, redirect_stdout
from datetime import datetime, timezone
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from retry_state import read, save, success
from verify import ROOT, verify, digest


def now():
    return datetime.now(timezone.utc).isoformat()


def null_ids(results):
    # Selection does not inspect correctness, IoU, target IDs or GT classes.
    return [r['csv_row'] for r in results if r.get('predicted_id') is None]


def check_inference_settings(expected, current):
    for section in ('api', 'policy', 'query_syntax', 'models'):
        old, new = expected[section], current[section]
        if section == 'api':
            # Each output has its own lock directory; it is not a model setting.
            old = {k: v for k, v in old.items() if k != 'api_slot_dir'}
            new = {k: v for k, v in new.items() if k != 'api_slot_dir'}
        if old != new:
            changed = sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
            raise ValueError(f'Inference settings differ from source: {section}: {changed}')


def validate_runtime_settings(source, output):
    # Match the initialization performed by the frozen batch runner before it
    # records its manifest, including its explicit --api-concurrency 8 option.
    os.environ['SEEGROUND_API_CONCURRENCY'] = '8'
    os.environ['SEEGROUND_API_SLOT_DIR'] = str(output / '.api_slots')
    import qwen_api, v4_policy
    from query_syntax import syntax_settings
    qwen_api._settings()
    assert v4_policy.settings() == read(ROOT / 'provenance/HISTORICAL_NR3D.json')['policy']
    assert syntax_settings()['backend'] == 'spacy'
    expected = read(source / 'manifest.json')
    current = {'api': qwen_api.api_config_summary(), 'policy': v4_policy.settings(),
               'query_syntax': syntax_settings(),
               'models': {k: os.getenv(k, '') for k in expected['models']}}
    check_inference_settings(expected, current)


def validate(results, rows):
    expected = {r['csv_row']: r for r in rows}
    observed = {r['csv_row']: r for r in results}
    if len(results) != len(observed) or set(expected) != set(observed):
        raise ValueError('Missing, duplicate or unexpected query IDs')
    for key, row in expected.items():
        if any(observed[key].get(k) != v for k, v in row.items()):
            raise ValueError(f'Query identity changed: {key}')


def snapshot(source, rows):
    pending = source / 'retry_pending.json'
    state = read(pending) if pending.exists() else None
    archived = Path(state['archive']) if state else None
    original = read((archived or source) / 'results.json')['results']
    validate(original, rows)
    selected = set(state['failed_ids']) if state else set()
    current = read(source / 'results.json')['results']
    current_ids = {r['csv_row'] for r in current}
    validate(current, [r for r in rows if r['csv_row'] in current_ids])
    cached = {r['csv_row']: r for r in current}
    # Only interrupted-round directories can contain newer evaluations than the
    # last aggregate write. Avoid reopening 9508 unchanged small files on exFAT.
    live = {p.name for p in source.iterdir() if p.name.startswith('q')}

    def latest(row):
        key = row['csv_row']
        path = source / f'q{key:05d}' / 'evaluation.json'
        if key in selected and path.parent.name in live and path.is_file():
            return read(path), str(path)
        if key in selected:
            if key in cached and cached[key] != row:
                raise ValueError(f'Completed replacement lost its evaluation: {key}')
            path = archived / f'q{key:05d}' / 'evaluation.json'
            return row, str(path)
        if cached.get(key) != row:
            raise ValueError(f'Preserved result changed or missing: {key}')
        return row, str(path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(latest, original))
    results = [v[0] for v in values]
    validate(results, rows)
    history = read(source / 'retry_history.json')['rounds'] if (source / 'retry_history.json').exists() else []
    used = max(len(history), state['round'] if state else 0)
    return results, {'source': str(source), 'recorded_at': now(),
                     'interrupted_false_round': state, 'previous_rounds_used': used,
                     'interrupted_round_is_not_a_complete_round_score': bool(state),
                     'evaluation_paths': {str(r['csv_row']): p for r, p in values},
                     'source_results_sha256': digest(source / 'results.json'),
                     'source_pending_sha256': digest(pending) if state else None}


def merge(before, replacements, selected):
    if {r['csv_row'] for r in replacements} != set(selected) or len(replacements) != len(selected):
        raise ValueError('Null retry is incomplete or has unexpected IDs')
    updated = {r['csv_row']: r for r in replacements}
    return [updated.get(r['csv_row'], r) for r in before]


def report(output, manifest, source):
    from scanrefer_metrics import accounting, report as original_report
    args = SimpleNamespace(manifest=manifest, results=output / 'results.json',
                           proposals='gt', output=output / 'scanrefer_metrics.json')
    with redirect_stdout(io.StringIO()):
        original_report(args)
    value = read(args.output)
    protocol = read(output / 'protocol.json')
    history = read(output / 'history.json')['rounds']
    # The source tree retains all first-pass and interrupted-false trajectories.
    # New null attempts live only in their separate round directories.
    usage = [accounting(source)] + [accounting(Path(r['output'])) for r in history]
    counters = ('api_attempts', 'prompt_tokens', 'completion_tokens', 'total_tokens',
                'cached_tokens', 'reasoning_tokens', 'trajectories',
                'attempts_without_token_usage', 'wall_elapsed_s_completed_sessions',
                'incomplete_sessions')
    value.update(evaluation_policy='cumulative_mixed_interrupted_false_then_null',
                 protocol_change_declared=True, retry_rounds=history,
                 retry_in_progress=(output / 'pending.json').exists(),
                 previous_rounds_used=protocol['previous_rounds_used'],
                 first_pass=read(source / 'first_pass/scanrefer_metrics.json'),
                 accounting_by_segment=[{'output': str(source), **usage[0]}] +
                     [{'output': h['output'], **u} for h, u in zip(history, usage[1:])],
                 accounting={**{k: sum(u.get(k, 0) for u in usage) for k in counters},
                             'cost': None, 'cost_note': 'Provider billing unavailable; no estimate'})
    save(args.output, value)
    print(json.dumps({'report': str(args.output), 'overall': value['overall'],
                      'policy': value['evaluation_policy']}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'preflight', 'report'))
    parser.add_argument('split', choices=('full',))
    parser.add_argument('--version', required=True)
    parser.add_argument('--retry-null', action='store_true', required=True)
    args = parser.parse_args()
    source = ROOT / 'outputs' / args.version / 'full'
    output = ROOT / 'outputs' / args.version / 'null_retry'
    output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for path in (source / '.run.lock', output / '.run.lock'):
            handle = stack.enter_context(path.open('a'))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit('Another run is active. Stop it before retry-null.')
        frozen, manifest = verify('full', args.version)
        if args.action != 'report':
            validate_runtime_settings(source, output)
        rows = read(manifest)['rows']
        identities = {name: digest(ROOT / name) for name in
                      ('retry_null.py', 'retry_null.sh', 'retry_null_progress.py')}
        protocol_path = output / 'protocol.json'
        if protocol_path.exists():
            protocol = read(protocol_path)
            if protocol['entry_sha256'] != identities or protocol['version_sha256'] != digest(ROOT / 'versions' / (args.version + '.json')):
                raise ValueError('Null retry code/version changed; use a new protocol output')
            origin = protocol['origin']
            if digest(source / 'results.json') != origin['source_results_sha256']:
                raise ValueError('Original output changed after branching; inspect before continuing')
            pending = source / 'retry_pending.json'
            if (digest(pending) if pending.exists() else None) != origin['source_pending_sha256']:
                raise ValueError('Original pending round changed after branching')
        else:
            print('Reading saved evaluations; no model calls.', flush=True)
            results, origin = snapshot(source, rows)
            protocol = {'version': args.version, 'created_at': now(), 'origin': origin,
                        'previous_rounds_used': origin['previous_rounds_used'],
                        'max_total_retry_rounds': frozen['max_retry_rounds'],
                        'selection': 'predicted_id is null, including errors and abstentions',
                        'entry_sha256': identities,
                        'version_sha256': digest(ROOT / 'versions' / (args.version + '.json')),
                        'policy_change': 'User requested null-only after interrupted false retry',
                        'not_original_predeclared_false_protocol': True}
            save(output / 'baseline_results.json', {'results': results})
            save(output / 'results.json', {'results': results})
            save(output / 'history.json', {'rounds': []})
            save(protocol_path, protocol)
        if args.action == 'report':
            report(output, manifest, source)
            return
        pending_path = output / 'pending.json'
        if pending_path.exists():
            state = read(pending_path)
        else:
            results = read(output / 'results.json')['results']
            validate(results, rows)
            history = read(output / 'history.json')['rounds']
            number = protocol['previous_rounds_used'] + len(history) + 1
            selected = null_ids(results)
            print(json.dumps({'selected_null': len(selected), 'preserved_nonnull': len(results)-len(selected),
                              'official_denominator': 9508, 'next_round': number,
                              'max_total_retry_rounds': frozen['max_retry_rounds'],
                              'inference_settings_match': True,
                              'model_calls_sent': 0}), flush=True)
            if args.action == 'preflight':
                return
            if not selected:
                raise SystemExit('No null predictions remain.')
            if number > frozen['max_retry_rounds']:
                raise SystemExit('Total retry-round cap reached, counting interrupted false rounds.')
            directory = output / f'round{number:03d}'
            directory.mkdir(exist_ok=True)
            save(directory / 'before_results.json', {'results': results})
            save(directory / 'selection.json', {**read(manifest), 'rows': [r for r in rows if r['csv_row'] in set(selected)]})
            state = {'round': number, 'output': str(directory), 'selected_ids': selected,
                     'started_at': now(), 'correct_before': sum(success(r) for r in results),
                     'selection': 'null_only', 'before_sha256': digest(directory / 'before_results.json')}
            save(pending_path, state)
        if args.action == 'preflight':
            print(json.dumps({'resume_round': state['round'], 'selected_null': len(state['selected_ids']),
                              'inference_settings_match': True, 'model_calls_sent': 0}))
            return
        directory = Path(state['output'])
        command = [sys.executable, '-u', str(ROOT / 'retry_null_progress.py'),
                   '--baseline', str(directory / 'before_results.json'),
                   '--manifest', str(directory / 'selection.json'),
                   '--predictions', str(ROOT / 'inputs/r8_predictions'),
                   '--scene-root', str(ROOT / 'inputs/scenes'), '--output', str(directory),
                   '--category-top-k', '5', '--workers', '8', '--api-concurrency', '8', '--review-limit', '3']
        session_dir = directory / 'execution_sessions'
        session_dir.mkdir(exist_ok=True)
        session_path = session_dir / f'{time.time_ns()}.json'
        session = {'started_at': now(), 'retry_round': state['round'], 'command': command}
        save(session_path, session)
        start = time.monotonic()
        try:
            result = subprocess.run(command)
            session['returncode'] = result.returncode
        finally:
            session.update(elapsed_s=time.monotonic()-start, finished_at=now())
            save(session_path, session)
        if result.returncode:
            raise SystemExit(result.returncode)
        before = read(directory / 'before_results.json')['results']
        if digest(directory / 'before_results.json') != state['before_sha256']:
            raise ValueError('Saved round baseline changed')
        replacements = read(directory / 'results.json')['results']
        combined = merge(before, replacements, state['selected_ids'])
        validate(combined, rows)
        correct = sum(success(r) for r in combined)
        entry = {**state, 'finished_at': now(), 'status': 'complete',
                 'correct_after': correct, 'newly_correct': correct-state['correct_before']}
        history = read(output / 'history.json')
        if not any(r['round'] == entry['round'] for r in history['rounds']):
            history['rounds'].append(entry)
            save(output / 'history.json', history)
        save(output / 'results.json', {'results': combined})
        save(directory / 'completion.json', entry)
        pending_path.unlink()
        report(output, manifest, source)


if __name__ == '__main__':
    main()
