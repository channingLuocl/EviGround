"""One manual, fixed-rule retry of the 109 audited visual-recall failures."""
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

from retry_null import merge, now, validate, validate_runtime_settings
from retry_state import read, save, success
from verify import ROOT, digest, verify

VERSION = 'r18-evidence-v1'
SOURCE = ROOT / 'outputs/r18-null-visual-recall-v1'
OUTPUT = ROOT / 'outputs/r18-visual109-v1'
ROUND = OUTPUT / 'round001'
RUNTIME = ROOT / 'visual109_runtime'
ENTRIES = ('retry_visual109.py', 'retry_visual109.sh', 'retry_visual109_progress.py',
           'visual109_policy.py', 'VISUAL109_RULES.md', 'test_visual109.py')


def selected_ids(rows):
    # Explicit evaluation-based development subset; never pass outcomes to inference.
    return sorted(r['csv_row'] for r in rows if not success(r))


def runtime_sources():
    from retry_null_visual import runtime_sources as previous_sources, replace_once
    sources = previous_sources()
    for name in sources:
        sources[name] = sources[name].replace('from null_visual_recall import', 'from visual109_policy import')
        sources[name] = sources[name].replace('r18-empty-pool-visual-recall-v1', 'r18-all-instances-atlas-binding-v1')
    scene = sources['real_scene.py']
    scene = replace_once(scene,
        "            if candidate_gate['fallback_applied']:\n"
        "                parsed['candidate_generation'].update(\n"
        "                    source='broad_fallback', status='unknown',\n"
        "                    visual_recall=candidate_gate)\n",
        "            parsed['candidate_generation'].update(\n"
        "                source='all_instances_visual_pool', status='unknown',\n"
        "                visual_recall=candidate_gate)\n")
    sources['real_scene.py'] = scene
    react = sources['r18_react.py']
    react = replace_once(react, 'def observer_frame(frame, objects, selected):',
        'from visual109_policy import candidate_atlas, parse_action, POLICY, INSTRUCTIONS as COVERAGE_INSTRUCTIONS\n'
        'INSTRUCTIONS = COVERAGE_INSTRUCTIONS + INSTRUCTIONS\n\n\n'
        'def observer_frame(frame, objects, selected):')
    react = replace_once(react,
        "        'focus_ids': array({'type': 'integer', 'enum': sorted(known)}),",
        "        'focus_ids': {**array({'type': 'integer', 'enum': sorted(known)}), 'maxItems': POLICY['max_focus_ids']},")
    react = replace_once(react, '    views = [overview[0], overview[-1]]\n',
        "    atlas = candidate_atlas(candidates, xyz, rgb, ids, out)\n"
        "    views = list(overview) + atlas\n"
        "    trace['r18_initial_resources'] = {\n"
        "        'overview_images': len(overview), 'atlas_pages': len(atlas),\n"
        "        'atlas_instances': sum(len(v['focus_ids']) for v in atlas),\n"
        "        'atlas_instance_renders': sum(v['views_rendered'] for v in atlas),\n"
        "        'dynamic_tool_budget': limit, 'max_focus_ids': POLICY['max_focus_ids'],\n"
        "        'atlas_order': POLICY['atlas_order']}\n"
        "    trace['r18_initial_atlas'] = [{k: v for k, v in view.items() if k != 'image'} for view in atlas]\n"
        "    trace['internal_usage']['render'] += len(overview) + sum(v['views_rendered'] for v in atlas)\n")
    start = react.index("    if recall.get('fallback_applied'):")
    stop = react.index("    payload['target_category_bindings']", start)
    react = react[:start] + (
        "    payload['candidate_generation'] = recall\n"
        "    payload['category_policy'] = ('All classifier-covered instances are allowed; category matches are priors, never exclusions. None is category-verified.')\n"
        "    payload['coverage_policy'] = POLICY\n"
        "    payload['decision_checklist'] = ['target noun phrase vs reference/support/accessory',\n"
        "        'survey all atlas IDs before narrowing', 'inspect winner and strongest plausible rival',\n"
        "        'bind reference IDs and preserve every clause of the original utterance']\n"
    ) + react[stop:]
    react = replace_once(react,
        '            if best_ranks.get(oid) in (2, 3, 4, 5) and oid not in inspected:',
        '            if oid is not None and oid not in inspected:')
    react = replace_once(react,
        "    best_ranks = {b['id']: min(b['matched_ranks'], default=None)\n"
        "                  for b in payload['target_category_bindings']}\n", '')
    react = replace_once(react,
        "                gap = 'The selected target is an alternative-only category match and has not been visibly inspected. Inspect it with plausible primary matches; distance alone is not category evidence.'",
        "                gap = 'Dynamically inspect the selected target together with its strongest plausible rival/reference; the survey atlas does not resolve the decisive clause.'")
    react = replace_once(react,
        "    trace['internal_usage']['render'] += len(overview)\n", '')
    react = replace_once(react,
        'from qwen_api import chat, image_part, parse_json_object',
        'from qwen_api import chat, image_part')
    react = replace_once(react,
        "        action = parse_json_object(response['_content'])",
        "        action = parse_action(response['_content'])")
    sources['r18_react.py'] = react
    runner = sources['run_grounding_v9.py']
    runner = replace_once(runner, '    write_json(saved, result)\n',
        "    result['visual109_resource_usage'] = data.get('active_trace', run.get('active_diagnostic', {})).get('r18_initial_resources', {})\n"
        '    write_json(saved, result)\n')
    sources['run_grounding_v9.py'] = runner
    return sources


def prepare(manifest):
    baseline = read(SOURCE / 'results.json')['results']
    validate(baseline, read(manifest)['rows'])
    assert digest(SOURCE / 'results.json') == read(SOURCE / 'completion.json')['results_sha256']
    previous_round = read(SOURCE / 'round001/results.json')['results']
    validate(previous_round, read(SOURCE / 'selection.json')['rows'])
    full_by_id = {r['csv_row']: r for r in baseline}
    assert all(full_by_id[r['csv_row']] == r for r in previous_round)
    selected = selected_ids(previous_round)
    assert len(previous_round) == 383 and len(selected) == 109
    assert sum(success(r) for r in baseline) == 6108
    if not selected:
        raise ValueError('No failures remain in the completed source round')
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
                  ('results.json', 'completion.json', 'protocol.json', 'scanrefer_metrics.json',
                   'round001/results.json', 'selection.json')}
    for name, expected in read(SOURCE / 'protocol.json')['entry_sha256'].items():
        assert digest(ROOT / name) == expected
        references[ROOT / name] = expected
    for name, expected in read(SOURCE / 'protocol.json')['runtime_sha256'].items():
        path = ROOT / 'null_visual_runtime' / name
        assert digest(path) == expected
        references[path] = expected
    for name in ('retry_null.py', 'retry_state.py', 'retry_null_progress.py', 'scanrefer_metrics.py', 'verify.py'):
        references[ROOT / name] = digest(ROOT / name)
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
        'selection': 'All Acc@0.25 failures in the previous 383-case visual round; evaluation-based development subset',
        'selection_uses_evaluation': True, 'evaluation_answers_in_worker': False,
        'generalization_claim': 'No independent-test gain claim; rules developed on these 109 cases',
        'selected_ids': selected, 'official_denominator': 9508,
        'baseline_correct25': sum(success(r) for r in baseline),
        'preserved_other_rows': len(baseline) - len(selected), 'max_additional_rounds': 1,
        'manual_launch_only': True, 'rl_enabled': False,
        'candidate_recall_changed': True, 'recall_policy': 'all_instances_atlas_binding_v1',
        'fallback': 'All classifier-covered instances are allowed, including when category pool is nonempty',
        'fixed_policy': __import__('visual109_policy').POLICY,
        'additional_resources': 'All 3 overviews plus ceil(N/12) atlas pages, N independent 300px instance renders; dynamic actions stay 3',
        'merge_policy': 'Replace all 109 outcomes once; no best-of selection or correctness-based cache',
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
    assert selected_ids(read(SOURCE / 'round001/results.json')['results']) == protocol['selected_ids']
    assert [r['csv_row'] for r in read(OUTPUT / 'selection.json')['rows']] == protocol['selected_ids']
    return protocol, baseline


def run_visual(protocol, baseline):
    for directory in ROUND.glob('q*'):
        if directory.is_dir() and not any((directory / name).exists() for name in ('evaluation.json', 'prediction.json')):
            raise ValueError(f'Interrupted trajectory requires audited recovery: {directory}')
    ROUND.mkdir(exist_ok=True)
    command = [sys.executable, '-u', str(ROOT / 'retry_visual109_progress.py'),
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
    value.update(evaluation_policy='targeted_visual109_fixed_rules_cumulative_not_single_pass',
        baseline=str(SOURCE), baseline_correct25=protocol['baseline_correct25'],
        newly_correct25=value['overall']['correct25'] - protocol['baseline_correct25'],
        selected_failed=len(protocol['selected_ids']), rl_enabled=False, candidate_recall_changed=True,
        original_frozen_result_unchanged=True, outside_original_three_round_protocol=True,
        fallback_cases=sum(r.get('candidate_gate', {}).get('fallback_applied', False) for r in rows),
        remaining_null_reasons={reason: sum(r.get('null_reason') == reason for r in rows)
                               for reason in ('empty_candidate_pool', 'model_abstention', 'runtime_or_api_error')},
        selection_uses_evaluation=True, generalization_claim=protocol['generalization_claim'],
        visual109_resources={key: sum(r.get('visual109_resource_usage', {}).get(key, 0) for r in rows)
            for key in ('overview_images', 'atlas_pages', 'atlas_instances', 'atlas_instance_renders')},
        accounting_by_segment={'completed_baseline': prior, 'visual109_round001': extra},
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
        print(json.dumps({'output': str(OUTPUT), 'selected_failed': len(protocol['selected_ids']),
            'baseline_correct25': protocol['baseline_correct25'], 'official_denominator': 9508,
            'rl_enabled': False, 'candidate_recall_changed': True,
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
