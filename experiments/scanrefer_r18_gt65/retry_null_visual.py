"""One manual null-only visual-recall run, separate from every frozen result."""
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

from retry_null import merge, null_ids, now, validate, validate_runtime_settings
from retry_state import read, save, success
from verify import ROOT, digest, verify

VERSION = 'r18-evidence-v1'
SOURCE = ROOT / 'outputs/r18-evidence-v1-extra-null-v1'
OUTPUT = ROOT / 'outputs/r18-null-visual-recall-v1'
ROUND = OUTPUT / 'round001'
RUNTIME = ROOT / 'null_visual_runtime'
ENTRIES = ('retry_null_visual.py', 'retry_null_visual.sh',
           'retry_null_visual_progress.py', 'null_visual_recall.py')


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('Frozen source anchor must occur exactly once')
    return text.replace(before, after, 1)


def runtime_sources():
    local = ROOT / 'source_r18/local'
    scene = (local / 'real_scene.py').read_text()
    scene = replace_once(scene, '            trace={}\n',
        "            from null_visual_recall import recover_candidates\n"
        "            candidates, candidate_gate = recover_candidates(candidates, objects)\n"
        "            record['candidate_gate'] = candidate_gate\n"
        "            if candidate_gate['fallback_applied']:\n"
        "                parsed['candidate_generation'].update(\n"
        "                    source='broad_fallback', status='unknown',\n"
        "                    visual_recall=candidate_gate)\n"
        "            trace={}\n")
    scene = replace_once(scene, "            if is_v9:\n                from categories_v9 import sha256\n",
        "            prediction['candidate_gate'] = candidate_gate\n"
        "            prediction['inference_variant'] = 'r18-empty-pool-visual-recall-v1'\n"
        "            if is_v9:\n                from categories_v9 import sha256\n")
    react = (local / 'r18_react.py').read_text()
    react = replace_once(react, "    payload['target_category_bindings'] = [\n",
        "    recall = parsed.get('candidate_generation', {}).get('visual_recall', {})\n"
        "    if recall.get('fallback_applied'):\n"
        "        payload['candidate_generation'] = recall\n"
        "        payload['category_policy'] = (\n"
        "            'The category filter returned no candidates. All classifier-covered scene '\n"
        "            'instances are allowed for visual inspection; none is category-verified. '\n"
        "            'Top-5 absence is not proof of absence. Use the original description and '\n"
        "            'visible evidence to bind the target and references. Keep uncertainty '\n"
        "            'when evidence is missing; an allowed ID is not an answer.')\n"
        "    payload['target_category_bindings'] = [\n")
    runner = (local / 'run_grounding_v9.py').read_text()
    runner = replace_once(runner, '    write_json(saved, result)\n',
        "    from null_visual_recall import null_reason\n"
        "    data = json.loads(prediction.read_text()) if prediction.exists() else {}\n"
        "    run_path = directory / 'run.json'\n"
        "    run = json.loads(run_path.read_text()) if run_path.exists() else {}\n"
        "    gate = data.get('candidate_gate', run.get('candidate_gate', {}))\n"
        "    effective = gate.get('effective_candidate_ids', data.get('category_target_ids', []))\n"
        "    result.update(candidate_gate=gate, inference_candidate_count=len(effective),\n"
        "                  effective_candidate_recall=row['target_id'] in effective,\n"
        "                  null_reason=null_reason({**data, 'error': result.get('error')}),\n"
        "                  inference_variant='r18-empty-pool-visual-recall-v1')\n"
        "    write_json(saved, result)\n")
    return {'real_scene.py': scene, 'r18_react.py': react, 'run_grounding_v9.py': runner}


def prepare(manifest):
    baseline = read(SOURCE / 'results.json')['results']
    validate(baseline, read(manifest)['rows'])
    assert digest(SOURCE / 'results.json') == read(SOURCE / 'completion.json')['results_sha256']
    selected = null_ids(baseline)
    if not selected:
        raise ValueError('No null predictions remain in the completed baseline')
    sources = runtime_sources()
    RUNTIME.mkdir(exist_ok=True)
    for name, text in sources.items():
        compile(text, name, 'exec')
        path = RUNTIME / name
        if path.exists() and path.read_text() != text:
            raise ValueError('Visual runtime already differs; use a new version')
        if not path.exists():
            path.write_text(text)
    shutil.copyfile(SOURCE / 'results.json', OUTPUT / 'baseline_results.json')
    full = read(manifest)
    selected_set = set(selected)
    save(OUTPUT / 'selection.json', {**full, 'rows': [r for r in full['rows'] if r['csv_row'] in selected_set]})
    references = {SOURCE / name: digest(SOURCE / name) for name in
                  ('results.json', 'completion.json', 'protocol.json', 'scanrefer_metrics.json')}
    original = ROOT / 'outputs' / VERSION / 'null_retry'
    references[original / 'FROZEN.json'] = digest(original / 'FROZEN.json')
    frozen = read(original / 'FROZEN.json')['artifact_sha256']
    for name, expected in frozen.items():
        path = original / 'terminal_stage' / name
        assert digest(path) == expected
        references[path] = expected
    for name in ('results.json', 'history.json', 'protocol.json'):
        assert digest(original / name) == frozen[name]
        references[original / name] = frozen[name]
    save(OUTPUT / 'protocol.json', {
        'name': OUTPUT.name, 'created_at': now(), 'baseline': str(SOURCE),
        'selection': 'Completed baseline predicted_id is null; no correctness-based selection',
        'selected_ids': selected, 'official_denominator': 9508,
        'baseline_correct25': sum(success(r) for r in baseline),
        'preserved_nonnull': len(baseline) - len(selected), 'max_additional_rounds': 1,
        'manual_launch_only': True, 'rl_enabled': False,
        'candidate_recall_changed': True, 'recall_policy': 'empty_pool_visual_recall_v1',
        'fallback': 'Only an empty category pool admits all classifier-covered instances; nonempty pools unchanged',
        'ground_truth_used_for_recall': False, 'tool_budget': 3,
        'outside_original_three_round_protocol': True,
        'source_reference_sha256': {str(p): h for p, h in references.items()},
        'runtime_sha256': {name: digest(RUNTIME / name) for name in sources},
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
    baseline = read(OUTPUT / 'baseline_results.json')['results']
    validate(baseline, read(manifest)['rows'])
    assert null_ids(baseline) == protocol['selected_ids']
    assert [r['csv_row'] for r in read(OUTPUT / 'selection.json')['rows']] == protocol['selected_ids']
    return protocol, baseline


def run_visual(protocol, baseline):
    for directory in ROUND.glob('q*'):
        if directory.is_dir() and not any((directory / name).exists() for name in ('evaluation.json', 'prediction.json')):
            raise ValueError(f'Interrupted trajectory requires audited recovery: {directory}')
    ROUND.mkdir(exist_ok=True)
    command = [sys.executable, '-u', str(ROOT / 'retry_null_visual_progress.py'),
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
         'selected_null': len(protocol['selected_ids']), 'results_sha256': digest(OUTPUT / 'results.json')})


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
    value.update(evaluation_policy='post_cap_null_visual_recall_cumulative_not_single_pass',
        baseline=str(SOURCE), baseline_correct25=protocol['baseline_correct25'],
        newly_correct25=value['overall']['correct25'] - protocol['baseline_correct25'],
        selected_null=len(protocol['selected_ids']), rl_enabled=False, candidate_recall_changed=True,
        original_frozen_result_unchanged=True, outside_original_three_round_protocol=True,
        fallback_cases=sum(r.get('candidate_gate', {}).get('fallback_applied', False) for r in rows),
        remaining_null_reasons={reason: sum(r.get('null_reason') == reason for r in rows)
                               for reason in ('empty_candidate_pool', 'model_abstention', 'runtime_or_api_error')},
        accounting_by_segment={'completed_baseline': prior, 'visual_recall_round001': extra},
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
            validate_runtime_settings(ROOT / 'outputs' / VERSION / 'full', ROUND)
            assert not any(os.getenv(k) for k in ('SEEGROUND_RL_CHECKPOINT', 'SEEGROUND_RL_RECALL'))
        print(json.dumps({'output': str(OUTPUT), 'selected_null': len(protocol['selected_ids']),
            'baseline_correct25': protocol['baseline_correct25'], 'official_denominator': 9508,
            'rl_enabled': False, 'candidate_recall_changed': True,
            'model_calls_sent_by_preflight': 0}), flush=True)
        if args.action in ('prepare', 'preflight'):
            return
        if not (OUTPUT / 'completion.json').exists():
            if args.action == 'report':
                raise SystemExit('Visual recall run is incomplete; no final cumulative score is available')
            run_visual(protocol, baseline)
        assert digest(OUTPUT / 'results.json') == read(OUTPUT / 'completion.json')['results_sha256']
        validate(read(OUTPUT / 'results.json')['results'], read(manifest)['rows'])
        report_visual(manifest, protocol)


if __name__ == '__main__':
    main()
