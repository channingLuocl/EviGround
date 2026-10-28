"""One manual independent retry of every Acc@0.25 failure with gpt-6.1-sol."""
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

from retry_null import check_inference_settings, merge, now, validate
from retry_state import read, save, success
from verify import ROOT, digest, verify

VERSION = 'r18-evidence-v1'
SOURCE = ROOT / 'outputs/r18-visual109-v1'
OUTPUT = ROOT / 'outputs/r18-all-false-gpt61-sol-v1'
ROUND = OUTPUT / 'round001'
RUNTIME = ROOT / 'visual109_runtime'
MODEL = 'gpt-6.1-sol'
BASE_URL = 'http://106.14.25.135:3000/v1'
CREDENTIAL = Path.home() / '.config/seeground/scanrefer-gpt61-sol.env'
ENTRIES = ('retry_all_false_gpt61.py', 'retry_all_false_gpt61.sh',
           'retry_all_false_gpt61_progress.py', 'ALL_FALSE_GPT61_RULES.md',
           'test_all_false_gpt61.py')


def selected_ids(rows):
    # Explicit evaluation-based development subset; never pass outcomes to inference.
    return sorted(r['csv_row'] for r in rows if not success(r))


def validate_new_settings():
    import qwen_api, v4_policy
    from query_syntax import syntax_settings

    os.environ['SEEGROUND_API_CONCURRENCY'] = '8'
    os.environ['SEEGROUND_API_SLOT_DIR'] = str(ROUND / '.api_slots')
    base, model, _ = qwen_api._settings()
    if (base, model) != (BASE_URL, MODEL):
        raise ValueError('Endpoint or model differs from this fixed retry')
    if v4_policy.settings() != read(ROOT / 'provenance/HISTORICAL_NR3D.json')['policy']:
        raise ValueError('Frozen r18 policy changed')
    source = read(ROOT / 'outputs' / VERSION / 'full/manifest.json')
    expected = {'api': {**source['api'], 'allow_http': True},
                'policy': source['policy'], 'query_syntax': source['query_syntax'],
                'models': {**source['models'], 'SEEGROUND_BASE_URL': BASE_URL,
                           'SEEGROUND_MODEL': MODEL}}
    current = {'api': qwen_api.api_config_summary(), 'policy': v4_policy.settings(),
               'query_syntax': syntax_settings(),
               'models': {k: os.getenv(k, '') for k in expected['models']}}
    check_inference_settings(expected, current)
    if any(os.getenv(k) for k in ('SEEGROUND_RL_CHECKPOINT', 'SEEGROUND_RL_RECALL')):
        raise ValueError('RL must remain disabled')
    if not CREDENTIAL.is_file() or CREDENTIAL.stat().st_mode & 0o077:
        raise ValueError('Private credential file is missing or not owner-only')
    return current


def prepare(manifest):
    baseline = read(SOURCE / 'results.json')['results']
    validate(baseline, read(manifest)['rows'])
    assert digest(SOURCE / 'results.json') == read(SOURCE / 'completion.json')['results_sha256']
    selected = selected_ids(baseline)
    if len(baseline) != 9508 or sum(success(r) for r in baseline) != 6131 or len(selected) != 3377:
        raise ValueError('Completed 64.48% baseline or false selection differs')
    source_protocol = read(SOURCE / 'protocol.json')
    references = {SOURCE / name: digest(SOURCE / name) for name in
                  ('results.json', 'completion.json', 'protocol.json', 'scanrefer_metrics.json',
                   'round001/results.json', 'selection.json')}
    for name, expected in source_protocol['entry_sha256'].items():
        path = ROOT / name
        if digest(path) != expected:
            raise ValueError(f'Completed visual109 entry changed: {name}')
        references[path] = expected
    for name, expected in source_protocol['runtime_sha256'].items():
        path = RUNTIME / name
        if digest(path) != expected:
            raise ValueError(f'Completed visual109 runtime changed: {name}')
        references[path] = expected
    for name in ('retry_null.py', 'retry_state.py', 'retry_null_progress.py',
                 'scanrefer_metrics.py', 'verify.py'):
        references[ROOT / name] = digest(ROOT / name)
    shutil.copyfile(SOURCE / 'results.json', OUTPUT / 'baseline_results.json')
    full = read(manifest)
    selected_set = set(selected)
    save(OUTPUT / 'selection.json', {**full, 'rows': [r for r in full['rows'] if r['csv_row'] in selected_set]})
    save(OUTPUT / 'protocol.json', {
        'name': OUTPUT.name, 'created_at': now(), 'baseline': str(SOURCE),
        'selection': 'All 3377 Acc@0.25 failures in the latest completed 9508-query cumulative baseline',
        'selection_uses_evaluation': True, 'evaluation_answers_in_worker': False,
        'generalization_claim': 'Targeted retry of known failures; not an independent-test gain',
        'selected_ids': selected, 'official_denominator': 9508,
        'baseline_correct25': sum(success(r) for r in baseline),
        'preserved_other_rows': len(baseline) - len(selected), 'max_additional_rounds': 1,
        'manual_launch_only': True, 'rl_enabled': False,
        'model_changed': True, 'candidate_recall_changed': False,
        'recall_policy': 'all_instances_atlas_binding_v1',
        'model': MODEL, 'api_base': BASE_URL,
        'credential_sha256': digest(CREDENTIAL),
        'runtime_settings': validate_new_settings(),
        'fixed_policy': __import__('visual109_policy').POLICY,
        'additional_resources': '3 overviews, ceil(N/12) atlas pages, N independent 300px renders per grounded case; 3 dynamic actions',
        'merge_policy': 'Replace all 3377 outcomes once; no best-of selection or correctness-based cache',
        'ground_truth_used_for_recall': False, 'tool_budget': 3,
        'outside_original_three_round_protocol': True,
        'source_reference_sha256': {str(p): h for p, h in references.items()},
        'runtime_sha256': source_protocol['runtime_sha256'],
        'entry_sha256': {name: digest(ROOT / name) for name in ENTRIES},
        'baseline_sha256': digest(OUTPUT / 'baseline_results.json'),
        'selection_sha256': digest(OUTPUT / 'selection.json'), 'manifest_sha256': digest(manifest),
    })


def checked(manifest):
    protocol = read(OUTPUT / 'protocol.json')
    for name, expected in protocol['source_reference_sha256'].items():
        assert digest(Path(name)) == expected, 'Completed baseline or frozen result changed'
    for parent, field in ((ROOT, 'entry_sha256'), (RUNTIME, 'runtime_sha256')):
        for name, expected in protocol[field].items():
            assert digest(parent / name) == expected, 'Prepared visual runtime or entry changed'
    for name, field in (('baseline_results.json', 'baseline_sha256'), ('selection.json', 'selection_sha256')):
        assert digest(OUTPUT / name) == protocol[field]
    assert digest(manifest) == protocol['manifest_sha256']
    assert digest(CREDENTIAL) == protocol['credential_sha256'], 'Credential file changed after preparation'
    baseline = read(OUTPUT / 'baseline_results.json')['results']
    validate(baseline, read(manifest)['rows'])
    assert len(baseline) == 9508 and sum(success(r) for r in baseline) == protocol['baseline_correct25']
    assert selected_ids(baseline) == protocol['selected_ids']
    assert [r['csv_row'] for r in read(OUTPUT / 'selection.json')['rows']] == protocol['selected_ids']
    return protocol, baseline


def run_visual(protocol, baseline):
    for directory in ROUND.glob('q*'):
        if directory.is_dir() and not any((directory / name).exists() for name in ('evaluation.json', 'prediction.json')):
            raise ValueError(f'Interrupted trajectory requires audited recovery: {directory}')
    ROUND.mkdir(exist_ok=True)
    command = [sys.executable, '-u', str(ROOT / 'retry_all_false_gpt61_progress.py'),
               '--baseline', str(OUTPUT / 'baseline_results.json'),
               '--manifest', str(OUTPUT / 'selection.json'),
               '--predictions', str(ROOT / 'inputs/r8_predictions'),
               '--scene-root', str(ROOT / 'inputs/scenes'), '--output', str(ROUND),
               '--category-top-k', '5', '--workers', '8', '--api-concurrency', '8', '--review-limit', '3']
    sessions = ROUND / 'execution_sessions'
    sessions.mkdir(exist_ok=True)
    path = sessions / f'{time.time_ns()}.json'
    session = {'started_at': now(), 'command': command}
    save(path, session)
    started = time.monotonic()
    try:
        result = subprocess.run(command)
        session['returncode'] = result.returncode
    finally:
        session.update(finished_at=now(), elapsed_s=time.monotonic() - started)
        save(path, session)
    if result.returncode:
        raise SystemExit(result.returncode)
    replacements = read(ROUND / 'results.json')['results']
    save(OUTPUT / 'results.json', {'results': merge(baseline, replacements, protocol['selected_ids'])})
    save(OUTPUT / 'completion.json', {'finished_at': now(), 'additional_rounds_completed': 1,
         'selected_failed': len(protocol['selected_ids']), 'results_sha256': digest(OUTPUT / 'results.json')})


def report_visual(manifest, protocol):
    from scanrefer_metrics import accounting, report
    args = SimpleNamespace(manifest=manifest, results=OUTPUT / 'results.json',
                           proposals='gt', output=OUTPUT / 'scanrefer_metrics.json')
    with redirect_stdout(io.StringIO()):
        report(args)
    value = read(args.output)
    extra = accounting(ROUND)
    prior = read(SOURCE / 'scanrefer_metrics.json')['accounting']
    numeric = {k for segment in (prior, extra) for k, v in segment.items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)}
    rows = read(ROUND / 'results.json')['results']
    value.update(evaluation_policy='all_false_gpt61_sol_cumulative_not_single_pass',
        baseline=str(SOURCE), baseline_correct25=protocol['baseline_correct25'],
        newly_correct25=value['overall']['correct25'] - protocol['baseline_correct25'],
        selected_failed=len(protocol['selected_ids']), rl_enabled=False,
        model_changed=True, candidate_recall_changed=False, model=MODEL, api_base=BASE_URL,
        original_frozen_result_unchanged=True, outside_original_three_round_protocol=True,
        fallback_cases=sum(r.get('candidate_gate', {}).get('fallback_applied', False) for r in rows),
        remaining_null_reasons={reason: sum(r.get('null_reason') == reason for r in rows)
                               for reason in ('empty_candidate_pool', 'model_abstention', 'runtime_or_api_error')},
        selection_uses_evaluation=True, generalization_claim=protocol['generalization_claim'],
        visual109_resources={key: sum(r.get('visual109_resource_usage', {}).get(key, 0) for r in rows)
            for key in ('overview_images', 'atlas_pages', 'atlas_instances', 'atlas_instance_renders')},
        accounting_by_segment={'completed_baseline': prior, 'all_false_gpt61_round001': extra},
        accounting={**{k: prior.get(k, 0) + extra.get(k, 0) for k in numeric}, 'cost': None,
                    'cost_note': 'Provider billing unavailable; no estimate'})
    save(args.output, value)
    print(json.dumps({'report': str(args.output), 'overall': value['overall'],
        'newly_correct25': value['newly_correct25'], 'remaining_null_reasons': value['remaining_null_reasons']}), flush=True)


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
            assert validate_new_settings() == protocol['runtime_settings']
        print(json.dumps({'output': str(OUTPUT), 'selected_failed': len(protocol['selected_ids']),
            'baseline_correct25': protocol['baseline_correct25'], 'official_denominator': 9508,
            'model': MODEL, 'api_base': BASE_URL, 'rl_enabled': False,
            'candidate_recall_changed': False,
            'model_calls_sent_by_preflight': 0}), flush=True)
        if args.action in ('prepare', 'preflight'):
            return
        if not (OUTPUT / 'completion.json').exists():
            if args.action == 'report':
                raise SystemExit('Independent visual109 run is incomplete; no final cumulative score is available')
            run_visual(protocol, baseline)
        assert digest(OUTPUT / 'results.json') == read(OUTPUT / 'completion.json')['results_sha256']
        validate(read(OUTPUT / 'results.json')['results'], read(manifest)['rows'])
        report_visual(manifest, protocol)


if __name__ == '__main__':
    main()
