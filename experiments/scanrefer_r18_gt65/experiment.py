"""Versioned ScanRefer execution; ground truth is confined to evaluation/retry selection."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from verify import ROOT, verify
from retry_state import prepare, finalize, save, checked_results


def report(output, manifest):
    subprocess.run([sys.executable, str(ROOT / 'scanrefer_metrics.py'), 'report',
                    '--manifest', str(manifest), '--results', str(output / 'results.json'),
                    '--proposals', 'gt', '--output', str(output / 'scanrefer_metrics.json')], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'run', 'report'))
    parser.add_argument('split', choices=('32', '84', 'full'))
    parser.add_argument('--version', required=True)
    parser.add_argument('--retry-false', action='store_true')
    args = parser.parse_args()
    frozen, manifest = verify(args.split, args.version)
    output = ROOT / 'outputs' / args.version / ('full' if args.split == 'full' else 'dev' + args.split)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / '.run.lock'
    # flock survives an interrupted shell without leaving a stale directory lock.
    import fcntl
    with lock.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('This version/split already has an active command')
        if args.action == 'report':
            report(output, manifest)
            return
        import qwen_api, v4_policy
        from query_syntax import syntax_settings
        qwen_api._settings()
        history = json.loads((ROOT / 'provenance/HISTORICAL_NR3D.json').read_text())
        assert v4_policy.settings() == history['policy'], 'Historical policy changed'
        assert syntax_settings()['backend'] == 'spacy'
        command = [sys.executable, '-u', str(ROOT / 'source_r18/local/run_grounding_v9.py'),
                   '--manifest', str(manifest), '--predictions', str(ROOT / 'inputs/r8_predictions'),
                   '--scene-root', str(ROOT / 'inputs/scenes'), '--output', str(output),
                   '--category-top-k', '5', '--workers', '8', '--api-concurrency', '8', '--review-limit', '3']
        # The original runner checks its complete immutable model/input/source manifest
        # before any failed trajectory is moved out for retry.
        subprocess.run(command + ['--prepare-only'], check=True)
        if args.retry_false and not (output / 'retry_pending.json').exists():
            checked_results(output, manifest)
            report(output, manifest)
            first = output / 'first_pass'
            if not first.exists() and not (output / 'retry_history.json').exists():
                first.mkdir()
                for name in ('results.json', 'scanrefer_metrics.json', 'manifest.json'):
                    shutil.copyfile(output / name, first / name)
        if args.retry_false or (output / 'retry_pending.json').exists():
            prepare(output, manifest, preview=args.action == 'preflight',
                    max_rounds=frozen['max_retry_rounds'])
        if args.action == 'preflight':
            return
        session = {'started_at': datetime.now(timezone.utc).isoformat(),
                   'version': args.version, 'git_commit': frozen['git_commit'],
                   'retry_round': (json.loads((output / 'retry_pending.json').read_text())['round']
                                   if (output / 'retry_pending.json').exists() else 0)}
        sessions = output / 'execution_sessions.jsonl'
        session_dir = output / 'execution_sessions'
        session_dir.mkdir(exist_ok=True)
        session_path = session_dir / (str(time.time_ns()) + '.json')
        save(session_path, session)
        started = time.monotonic()
        try:
            process = subprocess.run(command)
            session['returncode'] = process.returncode
        finally:
            session['elapsed_s'] = time.monotonic() - started
            session['finished_at'] = datetime.now(timezone.utc).isoformat()
            save(session_path, session)
            with sessions.open('a') as stream:
                stream.write(json.dumps(session) + '\n')
        if process.returncode:
            raise SystemExit(process.returncode)
        finalize(output, manifest)
        report(output, manifest)
        if not (output / 'retry_history.json').exists():
            first = output / 'first_pass'
            first.mkdir(exist_ok=True)
            for name in ('results.json', 'scanrefer_metrics.json', 'manifest.json', 'execution_sessions.jsonl'):
                if not (first / name).exists():
                    shutil.copyfile(output / name, first / name)


if __name__ == '__main__':
    main()
