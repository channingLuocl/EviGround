"""Posthoc acceptance accounting; reads no evaluation labels and calls no models."""
from collections import Counter
import json
from pathlib import Path
import laya_joint as joint

cases = []
for folder in sorted((joint.ROOT / 'cases').glob('v*-nr3d'), key=lambda p: int(p.name.split('-')[0][1:])):
    if not (folder / 'result.json').exists():
        continue
    result = json.loads((folder / 'result.json').read_text())
    trace = json.loads((folder / 'trace.json').read_text())
    receipts = [json.loads(p.read_text()) for p in sorted(folder.glob('calls/*/receipt.json'))]
    events = trace['attempts']
    capacity = [r for r in receipts if r.get('status') == 'failed' and not r.get('http_sent')
        and r.get('preflight', {}).get('maximum', 0) > 32768]
    fast = [r for r in receipts if r['role'] == 'fast_action_selector']
    sent_qwen = [r for r in receipts if r['role'] != 'fast_action_selector' and r.get('http_sent')]
    assert all(r.get('reported_model') == joint.MODEL for r in sent_qwen if r['status'] == 'completed')
    assert all(r['preflight']['maximum'] <= 32768 for r in sent_qwen)
    assert all(r['usage']['prompt_tokens'] + r['preflight']['output_reserved'] <= 32768
        for r in sent_qwen if r.get('usage'))
    assert all(not r['usage']['truncated'] and r['usage']['state_tokens_dropped'] == 0
        for r in fast if r['status'] == 'completed')
    assert result['successful_tools'] <= 10 and result['ancillary_slots'] <= 6 and result['primary_decisions'] <= 17
    assert result['inherited_primary_decisions'] == 0
    nodes = (trace['monitor_plan'] or {}).get('nodes', [])
    completed = [n['id'] for n in nodes if n['status'] == 'completed']
    exclusions = sorted({oid for n in nodes for oid in n.get('excluded_target_ids', [])})
    predicted = result.get('predicted_id')
    cases.append({'case': folder.name, 'raw_status': result['status'],
        'acceptance_status': 'capacity_preflight_blocked' if capacity else result['status'],
        'physical_successes_B': result['successful_tools'], 'ancillary_used_R': result['ancillary_slots'],
        'decisions_used_T': result['primary_decisions'], 'budget_ceiling_B_R_T': [10, 6, 17],
        'fast_model_calls': len(fast), 'qwen_http_attempts': len(sent_qwen),
        'qwen_generations_completed': sum(r['status'] == 'completed' for r in sent_qwen),
        'visual_model_calls': sum(r['role'] == 'visual' for r in receipts),
        'calls_blocked_before_http': sum(not r.get('http_sent', False) for r in receipts),
        'accepted_actions': [a['action']['action'] for a in events if a.get('monitor_feedback', {}).get('action_accepted')],
        'invalid_or_failed_decision_attempts': sum('feedback' in a and
            not ('no HTTP sent' in a['feedback'] and capacity) for a in events)
            + int(result.get('status') == 'unhandled_contract_failure'),
        'repeated_physical_actions_executed': sum(bool(a.get('repeated_physical_action_executed')) for a in events),
        'completed_nodes': completed, 'target_exclusions': exclusions,
        'independent_evidence_audit': 'pending independent review' if predicted is not None else 'not accepted: no supported final prediction; inspect actual observations/trace separately',
        'predicted_id': predicted, 'localization_correctness': 'not_evaluated_prediction_requires_freeze_and_evidence_audit' if predicted is not None else 'not_evaluated_no_prediction; no GT read',
        'valid_final_submission': result.get('status') in ('completed', 'abstained'),
        'terminal_record_posthoc': result.get('posthoc_terminal_record', False),
        'joint_episode_completed': False, 'formal_result': False,
        'capacity_blockers': [{'call_id': r['call_id'], 'bound_including_reserved_output': r['preflight']['maximum'],
            'no_http_sent': True, 'not_proof_of_actual_model_overflow': True} for r in capacity],
        'trace_sha256': joint.file_sha(folder / 'trace.json'),
        'model_ledger_sha256': joint.file_sha(folder / 'model-ledger.json')})

semantic = []
for version in ['v1', 'v2', 'v3']:
    source = joint.ROOT / f'semantic-probes-{version}/results.json'
    data = json.loads(source.read_text())
    witnesses = []
    groups = {}
    for row in data['results']:
        groups.setdefault((row['case'], row['position']), []).append(row)
    for (name, position), group in groups.items():
        if len({r['preflight']['full_original_state_sha256'] for r in group}) == 1 and len({r['chosen'] for r in group}) > 1:
            witnesses.append({'case': name, 'position': position,
                'same_accepted_state_sha256': group[0]['preflight']['full_original_state_sha256'],
                'actions_by_order': {str(r.get('order')): r['chosen'] for r in group}})
    semantic.append({'version': version, 'requests': data['laya_calls'],
        'strict_expected_matches': data['passed'], 'not_an_accuracy_benchmark': True,
        'selected_choice_counts': dict(Counter(r['choice'] for r in data['results'])),
        'actions_by_case': {name: dict(Counter(r['chosen'] for r in data['results'] if r['case'] == name))
            for name in sorted({r['case'] for r in data['results']})},
        'order_sensitivity_witnesses': witnesses,
        'all_untruncated': all(not r['usage']['truncated'] for r in data['results']),
        'source_sha256': joint.file_sha(source)})

fixed = []
for source in sorted(joint.ROOT.glob('semantic-fixed-*/results.json')):
    data = json.loads(source.read_text())
    groups = {}
    for row in data['records']:
        groups.setdefault((row['case'], row['position']), []).append(row)
    witnesses = [{'case': name, 'position': pos,
        'state_sha256': group[0]['preflight']['full_original_state_sha256'],
        'choices': [{'order': r['order'], 'chosen': r['chosen'], 'expected': r['expected']} for r in group]}
        for (name, pos), group in groups.items() if len({r['chosen'] for r in group}) > 1]
    fixed.append({'version': source.parent.name, 'requests': data['calls'],
        'expected_matches': data['passed'], 'synthetic_usefulness_only_not_scene_accuracy': True,
        'all_untruncated': all(not r['usage']['truncated'] and not r['usage']['state_tokens_dropped'] for r in data['records']),
        'input_tokens_range': [min(r['usage']['input_tokens'] for r in data['records']), max(r['usage']['input_tokens'] for r in data['records'])],
        'per_case': {name: {'expected_matches': sum(r['passed'] for r in data['records'] if r['case'] == name),
            'actions': dict(Counter(r['chosen'] for r in data['records'] if r['case'] == name))}
            for name in sorted({r['case'] for r in data['records']})},
        'order_sensitivity_witnesses': witnesses, 'source_sha256': joint.file_sha(source)})
control_path = joint.ROOT / 'semantic-positive-control-v1/results.json'
control = json.loads(control_path.read_text())
report = {'status': 'blocked_not_accepted', 'deployment_usable': True,
    'real_joint_episode_success': False, 'second_dataset_started': False,
    'model': joint.MODEL, 'cases': cases, 'synthetic_semantic_probes': semantic,
    'fixed_semantic_probes': fixed,
    'positive_control': {'requests': control['calls'], 'expected_matches': control['passed'],
        'source_sha256': joint.file_sha(control_path), 'scope': 'short customer tickets, not planning'},
    'blockers': ['Capacity remains request-specific: instruction consolidation certifies the old v4 blocked request under the same conservative counter; no exact tokenizer or arbitrary-history capacity claim.',
        'Three preregistered synthetic contracts retain action-order and information-gain failures despite an 18/18 short-classification control.',
        'Fresh task runs reject invalid model-owned alternatives/graphs under unchanged R=6; these failures are not program-repaired into success.',
        'A real v10 image request passed preflight and completed; the following full-state request is conservatively rejected at 35733, not evidence of actual model overflow.'],
    'protocol_comparability': 'Diagnostic extension: added slow consultations consume original R slots. Not a formal or pure-Qwen comparison.',
    'exact_tokenizer_limit': 'The tested /tokenize and /count_tokens routes returned URL errors; this does not establish that every provider counting interface is unavailable.',
    'counter_assumptions': joint.window(None, [], 0)['assumptions'],
    'scope_of_negative_conclusion': 'The tested current interfaces fail necessary information-gain and order-robustness controls. This is a reproducible obstacle to accepting this division of roles, not proof that every Laya prompt or architecture is impossible.',
    'capacity_consolidation': json.loads((joint.ROOT / 'capacity-primary-consolidation.json').read_text()),
    'accounting_caveats': [
        'v5 uncaught ValidationError terminated before the original additional invalid-response R charge; raw R=1 is retained, mapped R would be 2. Later versions wrap this as ContractFailure and charge the original repair slot.',
        'v9 local schema preparation assertion occurred before call-folder creation; both rejected attempts and their R/T costs remain in trace. See local-preparation-failures.json; no Qwen request was sent for those attempts.',
        'v5 result.json is a clearly marked posthoc terminal record from immutable trace/receipts, not a continuation.'],
    'evaluation_labels_read': False, 'program_selected_target_or_completed_nodes': False,
    'audit_model_calls': 0, 'audit_script_sha256': joint.file_sha(__file__)}
joint.save(joint.ROOT / 'acceptance.json', report)
print(json.dumps({'status': report['status'], 'cases': [{k: c[k] for k in
    ['case', 'acceptance_status', 'physical_successes_B', 'ancillary_used_R', 'decisions_used_T', 'accepted_actions']}
    for c in cases], 'semantic_probe_requests': sum(s['requests'] for s in semantic)}, ensure_ascii=False, indent=2))
