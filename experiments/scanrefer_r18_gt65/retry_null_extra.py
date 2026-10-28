"""One manually launched, separately declared null retry after the frozen cap."""
import argparse
from contextlib import redirect_stdout
import fcntl
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

from retry_null import merge, now, null_ids, validate, validate_runtime_settings
from retry_state import read, save, success
from verify import ROOT, digest, verify

VERSION = 'r18-evidence-v1'
SOURCE = ROOT / 'outputs' / VERSION / 'null_retry'
OUTPUT = ROOT / 'outputs/r18-evidence-v1-extra-null-v1'
ROUND = OUTPUT / 'round004'
ENTRIES = ('retry_null_extra.py', 'retry_null_extra.sh', 'retry_null_progress.py', 'retry_null.py')


def prepare(manifest):
    frozen = read(SOURCE / 'FROZEN.json')
    assert frozen['rounds_used'] == frozen['maximum_rounds'] == 3
    references = {SOURCE / 'FROZEN.json': digest(SOURCE / 'FROZEN.json'),
                  ROOT / 'versions' / (VERSION + '.json'): digest(ROOT / 'versions' / (VERSION + '.json'))}
    for name in ('results.json', 'history.json', 'protocol.json'):
        path = SOURCE / name
        assert digest(path) == frozen['artifact_sha256'][name]
        references[path] = digest(path)
    terminal = SOURCE / 'terminal_stage/FINAL_EVALUATION.json'
    assert digest(terminal) == frozen['artifact_sha256']['FINAL_EVALUATION.json']
    references[terminal] = digest(terminal)
    baseline = read(SOURCE / 'results.json')['results']
    full = read(manifest)
    validate(baseline, full['rows'])
    selected = null_ids(baseline)
    if not selected:
        raise ValueError('No frozen null predictions remain')
    shutil.copyfile(SOURCE / 'results.json', OUTPUT / 'baseline_results.json')
    save(OUTPUT / 'selection.json', {**full, 'rows': [r for r in full['rows'] if r['csv_row'] in set(selected)]})
    protocol = {'name': OUTPUT.name, 'created_at': now(), 'inference_version': VERSION,
                'baseline_rounds_used': 3, 'max_additional_rounds': 1, 'display_round': 4,
                'selection': 'Frozen predicted_id is null; no correctness-based selection',
                'selected_ids': selected, 'official_denominator': 9508,
                'preserved_nonnull': len(baseline)-len(selected),
                'authorization': 'User authorized preparing one separate manual null-only retry; no inference launch authorized here',
                'manual_launch_only': True, 'rl_enabled': False, 'candidate_recall_changed': False,
                'not_within_original_three_round_protocol': True,
                'source_reference_sha256': {str(p): h for p, h in references.items()},
                'entry_sha256': {name: digest(ROOT / name) for name in ENTRIES},
                'baseline_sha256': digest(OUTPUT / 'baseline_results.json'),
                'selection_sha256': digest(OUTPUT / 'selection.json'),
                'manifest_sha256': digest(manifest)}
    save(OUTPUT / 'protocol.json', protocol)


def checked(manifest):
    protocol = read(OUTPUT / 'protocol.json')
    assert protocol['max_additional_rounds'] == 1 and protocol['baseline_rounds_used'] == 3
    for name, expected in protocol['entry_sha256'].items():
        assert digest(ROOT / name) == expected, 'Extra entry changed after preparation'
    for name, expected in protocol['source_reference_sha256'].items():
        assert digest(Path(name)) == expected, 'Frozen source artifact changed'
    assert digest(OUTPUT / 'baseline_results.json') == protocol['baseline_sha256']
    assert digest(OUTPUT / 'selection.json') == protocol['selection_sha256']
    assert digest(manifest) == protocol['manifest_sha256']
    baseline = read(OUTPUT / 'baseline_results.json')['results']
    validate(baseline, read(manifest)['rows'])
    assert null_ids(baseline) == protocol['selected_ids']
    assert [r['csv_row'] for r in read(OUTPUT / 'selection.json')['rows']] == protocol['selected_ids']
    return protocol, baseline


def report_extra(manifest, protocol, baseline):
    from scanrefer_metrics import accounting, report
    args = SimpleNamespace(manifest=manifest, results=OUTPUT / 'results.json',
                           proposals='gt', output=OUTPUT / 'scanrefer_metrics.json')
    with redirect_stdout(io.StringIO()):
        report(args)
    value = read(args.output)
    prior = read(SOURCE / 'terminal_stage/FINAL_EVALUATION.json')['accounting']
    extra = accounting(ROUND)
    numeric = {k for k, v in prior.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    total = {k: prior.get(k, 0)+extra.get(k, 0) for k in numeric}
    value.update(evaluation_policy='post_cap_extra_null_only_cumulative_not_single_pass',
                 original_frozen_result_unchanged=True, outside_original_three_round_protocol=True,
                 baseline_correct25=sum(success(r) for r in baseline),
                 newly_correct25=value['overall']['correct25']-sum(success(r) for r in baseline),
                 additional_rounds=1, selected_null=len(protocol['selected_ids']),
                 rl_enabled=False, candidate_recall_changed=False,
                 accounting_by_segment={'frozen_baseline': prior, 'extra_round004': extra},
                 accounting={**total, 'cost': None, 'cost_note': 'Provider billing unavailable; no estimate'})
    save(args.output, value)
    print(json.dumps({'report': str(args.output), 'overall': value['overall'],
                      'newly_correct25': value['newly_correct25'], 'policy': value['evaluation_policy']}), flush=True)


def run_extra(protocol, baseline):
    # Terminal outcomes, including errors and abstentions, are cached equally.
    # An unfinished physical trajectory needs audit before any fresh request.
    for directory in ROUND.glob('q*'):
        if directory.is_dir() and not (directory / 'evaluation.json').exists() and not (directory / 'prediction.json').exists():
            raise ValueError(f'Interrupted trajectory requires audited recovery: {directory}')
    ROUND.mkdir(exist_ok=True)
    command = [sys.executable, '-u', str(ROOT / 'retry_null_progress.py'),
               '--baseline', str(OUTPUT / 'baseline_results.json'),
               '--manifest', str(OUTPUT / 'selection.json'),
               '--predictions', str(ROOT / 'inputs/r8_predictions'),
               '--scene-root', str(ROOT / 'inputs/scenes'), '--output', str(ROUND),
               '--category-top-k', '5', '--workers', '8', '--api-concurrency', '8', '--review-limit', '3']
    sessions = ROUND / 'execution_sessions'
    sessions.mkdir(exist_ok=True)
    path = sessions / f'{time.time_ns()}.json'
    session = {'started_at': now(), 'command': command, 'post_cap_additional_round': 1}
    save(path, session)
    start = time.monotonic()
    try:
        result = subprocess.run(command)
        session['returncode'] = result.returncode
    finally:
        session.update(finished_at=now(), elapsed_s=time.monotonic()-start)
        save(path, session)
    if result.returncode:
        raise SystemExit(result.returncode)
    combined = merge(baseline, read(ROUND / 'results.json')['results'], protocol['selected_ids'])
    save(OUTPUT / 'results.json', {'results': combined})
    save(OUTPUT / 'completion.json', {'finished_at': now(), 'additional_rounds_completed': 1,
         'selected_null': len(protocol['selected_ids']), 'results_sha256': digest(OUTPUT / 'results.json')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'preflight', 'run', 'report'))
    args = parser.parse_args()
    _, manifest = verify('full', VERSION)
    OUTPUT.mkdir(exist_ok=True)
    with (OUTPUT / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == 'prepare' and not (OUTPUT / 'protocol.json').exists():
            prepare(manifest)
        protocol, baseline = checked(manifest)
        if args.action != 'report':
            validate_runtime_settings(ROOT / 'outputs' / VERSION / 'full', ROUND)
            assert not any(os.getenv(k) for k in ('SEEGROUND_RL_CHECKPOINT', 'SEEGROUND_RL_RECALL'))
        print(json.dumps({'extra_output': str(OUTPUT), 'selected_null': len(protocol['selected_ids']),
                          'preserved_nonnull': protocol['preserved_nonnull'], 'official_denominator': 9508,
                          'max_additional_rounds': 1, 'rl_enabled': False, 'model_calls_sent_by_preflight': 0}), flush=True)
        if args.action in ('prepare', 'preflight'):
            return
        if not (OUTPUT / 'completion.json').exists():
            if args.action == 'report':
                raise SystemExit('Extra round is incomplete; no final cumulative score is available yet.')
            run_extra(protocol, baseline)
        complete = read(OUTPUT / 'completion.json')
        assert digest(OUTPUT / 'results.json') == complete['results_sha256']
        validate(read(OUTPUT / 'results.json')['results'], read(manifest)['rows'])
        report_extra(manifest, protocol, baseline)


if __name__ == '__main__':
    main()
