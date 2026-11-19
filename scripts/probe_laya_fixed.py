"""Preregistered synthetic action-usefulness tests; no Qwen, GT, or scene tools."""
from copy import deepcopy
from itertools import permutations
import argparse
import json
import time

import requests
from tokenizers import Tokenizer
import laya_joint as joint


CASES = [
    ('material_supported', 'supported', 'commit',
     'Current task: record the material conclusion. Delivered image e1-v0 clearly shows leather on '
     'chair 1 and plastic on chair 2. Both objects are fully visible. No uncertainty remains about '
     'material. Heights were already measured. The shared proposed update records this evidence; '
     'commit submits that change. New images or repeated heights add no missing evidence.'),
    ('material_insufficient', 'insufficient', 'inspect',
     'Current task: establish which chair is leather. No material image or material report has been '
     'delivered. The archive contains only category scores and already-measured heights. Neither '
     'scores nor heights establishes material. Need a new image. Reading archive rows or repeating '
     'heights adds no material evidence.'),
    ('material_refuted', 'refuted', 'consult',
     'Current task: repair an accepted material binding. Accepted state binds chair 1 as leather; '
     'delivered image e1-v0 clearly shows plastic on chair 1 and leather on chair 2. No visual '
     'uncertainty remains. Need the semantic Monitor to reopen the false binding and descendants. '
     'Neither repeating geometry nor another identical image repairs the accepted state.'),
    ('geometry_supported', 'supported', 'commit',
     'Current task: record the taller-chair comparison. Delivered calibrated measurements e1 give '
     'chair 1 height 1.2m and chair 2 height 0.8m, same datum, complete group {1,2}. No uncertainty '
     'remains. The shared proposed update records chair 1 as taller; commit submits it. Repeating '
     'these measurements or acquiring material images adds no missing evidence.'),
    ('geometry_insufficient', 'insufficient', 'measure',
     'Current task: determine the taller chair in complete group {1,2}. Delivered image e1-v0 '
     'establishes material only. Isolated panels have independent scales. NO calibrated height '
     'measurement exists; archive rows contain category scores only. Need heights of BOTH chairs. '
     'Another material image or rereading category scores cannot resolve height.'),
    ('geometry_refuted', 'refuted', 'consult',
     'Current task: resolve accepted-state conflict. Accepted state says chair 1 is taller and its '
     'dependent node is completed. New delivered calibrated measurement e1 gives chair 1 height '
     '0.7m and chair 2 height 1.3m, same datum, complete group {1,2}. Need semantic Monitor help '
     'to reopen the false comparison and descendants. Repeating the exact measurement or another '
     'material image adds no missing evidence and does not repair accepted state.'),
]


def fixture(name, condition, expected, facts, position, order):
    rows = ['Unrelated archive metadata: this row contains no current-task observations.'] * 9
    rows.insert({'start': 0, 'middle': 4, 'end': 9}[position], facts)
    state = {'description': 'Resolve the current chair evidence task using the delivered observations.',
        'current_task': name, 'evidence_condition': condition, 'records': rows,
        'object_directory': [{'id': 1}, {'id': 2}], 'allowed_target_ids': [1, 2],
        'accepted_bindings': {'chair_candidates': [1, 2]}, 'accepted_exclusions': [],
        'remaining_tools': 8, 'remaining_consult_or_repair': 3,
        'source_directory': [{'evidence_id': 'initial', 'sections': ['native_rows']},
                             {'evidence_id': 'e1', 'sections': ['objects', 'images']}],
        'scope': 'Synthetic usefulness selection. No scene/DagMonitor action is executed or certified.'}
    common = {'question': '', 'focus_ids': [], 'context': None, 'azimuth_deg': None,
        'elevation_deg': None, 'frame': None, 'source_ids': [], 'sections': [],
        'image_ids': [], 'predicted_id': None, 'evidence': [], 'unresolved': []}
    options = {
        'inspect': {**common, 'action': 'inspect', 'focus_ids': [1, 2], 'context': 'isolated',
            'azimuth_deg': 90, 'elevation_deg': 30, 'reason': 'Acquire images of the candidate chairs.'},
        'measure': {**common, 'action': 'measure', 'measurement_type': 'object_geometry',
            'target_ids': [1, 2], 'reference_ids': [], 'node_id': 'n0',
            'reason': 'Measure both chair heights from calibrated geometry.'},
        'consult': {**common, 'action': 'consult', 'source_ids': ['initial'],
            'sections': ['native_rows'],
            'question': 'Review current evidence and revise or reopen the accepted semantic state if needed.',
            'reason': 'Ask the semantic Monitor to review and repair the state.'},
        'commit': {**common, 'action': 'commit',
            'reason': 'Record the proposed supported conclusion without a new acquisition.'},
    }
    names = ['commit', 'inspect', 'measure'] if condition == 'supported' else ['inspect', 'measure', 'consult']
    actions = [deepcopy(options[names[i]]) for i in order]
    proposal = {'state_version': joint.digest(state),
        'monitor': {'current_node_id': 'n0', 'synthetic_proposal': 'Record the delivered evidence if supported; otherwise keep the predicate unresolved.'},
        'decision_basis': 'Choose the action that resolves the current task with new information or an evidence-backed state update. Preserve uncertainty and rollback requirements.',
        'options': deepcopy(actions)}
    return state, proposal, actions


def run(root, url, compact=False, conditional=False):
    root.mkdir(parents=True, exist_ok=False)
    (root / 'probe.py').write_bytes(__import__('pathlib').Path(__file__).read_bytes())
    joint.save(root / 'preregistered.json', {'cases': CASES, 'positions': ['start', 'middle', 'end'],
        'orders': list(permutations(range(3))), 'requests_planned': 108,
        'expected_chosen_before_http': True, 'synthetic_usefulness_only': True,
        'qwen_selected_actions': False, 'actual_choices_never_corrected': True,
        'compact_contract': compact,
        'conditional_option_wording': conditional,
        'compact_scope': 'Same complete synthetic accepted state/Monitor/actions; redundant wrapper metadata removed. No evidence or parameters selected or omitted.',
        'probe_sha256': joint.file_sha(__file__), 'entry_sha256': joint.file_sha(joint.__file__)})
    tokenizer = Tokenizer.from_file(str(joint.ROOT / 'laya-tokenizer/tokenizer.json'))
    tokenizer.no_truncation(); tokenizer.no_padding()
    session = requests.Session(); session.trust_env = False
    records = []
    for name, condition, expected, facts in CASES:
        for position in ['start', 'middle', 'end']:
            for order in permutations(range(3)):
                state, proposal, actions = fixture(name, condition, expected, facts, position, order)
                if conditional:
                    reasons = {
                        'inspect': 'Acquire missing material images; avoid already observed appearance.',
                        'measure': 'Obtain unknown heights; avoid already measured heights.',
                        'consult': 'Repair accepted state contradicted by delivered evidence.',
                        'commit': 'Record a fully supported conclusion with no evidence missing.',
                    }
                    for action in actions:
                        action['reason'] = reasons[action['action']]
                body, preflight = joint.laya_request(state, proposal, actions, tokenizer)
                if compact:
                    body['state'] = {**state, 'semantic_monitor_proposal': proposal['monitor'],
                        'consultant_analysis': proposal['decision_basis'],
                        'actions': body['state']['actions']}
                    preflight['state_tokens'] = len(tokenizer.encode(json.dumps(body['state'], ensure_ascii=False), add_special_tokens=False).ids)
                    preflight['input_tokens'] = preflight['state_tokens'] + preflight['head_tokens'] + 1
                    preflight['visibility_changed'] = True
                    preflight['change'] = 'Unnest accepted state and remove only constant wrapper scaffolding; complete synthetic facts/actions/Monitor retained'
                folder = root / f'{name}-{position}-' / ''.join(map(str, order))
                joint.save(folder / 'request.json', body)
                receipt = {'case': name, 'condition': condition, 'expected': expected,
                    'position': position, 'order': list(order), 'preflight': preflight,
                    'status': 'started', 'http_sent': True}
                joint.save(folder / 'receipt.json', receipt)
                started = time.monotonic()
                try:
                    response = session.post(url + '/v1/systemone', json=body, timeout=(10, 120))
                    receipt['http_status'] = response.status_code
                    (folder / 'response.txt').write_text(response.text)
                    response.raise_for_status()
                    wire = response.json(); usage = wire['usage']
                    assert not usage['truncated'] and usage['state_tokens_dropped'] == 0
                    assert usage['input_tokens'] == preflight['input_tokens']
                    assert usage['state_tokens'] == preflight['state_tokens']
                    choice = wire['answers']['next_action']['choice']
                    chosen = body['state']['actions'][choice]['action']
                    receipt.update(status='completed', chosen=chosen, choice=choice,
                        passed=chosen == expected, usage=usage,
                        probabilities=wire['answers']['next_action']['probabilities'])
                except Exception as exc:
                    receipt.update(status='failed', error_type=type(exc).__name__, error=str(exc))
                    raise
                finally:
                    receipt['latency_s'] = round(time.monotonic() - started, 3)
                    joint.save(folder / 'receipt.json', receipt)
                    records.append(receipt)
                    joint.save(root / 'results.json', {'records': records, 'qwen_calls': 0,
                        'scene_actions_executed': 0, 'calls': len(records),
                        'passed': sum(r.get('passed', False) for r in records)})
                print(name, position, order, chosen, chosen == expected, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', required=True)
    parser.add_argument('--url', default='http://127.0.0.1:18180')
    parser.add_argument('--compact', action='store_true')
    parser.add_argument('--conditional', action='store_true')
    args = parser.parse_args()
    run(joint.ROOT / ('semantic-fixed-' + args.version), args.url.rstrip('/'), args.compact, args.conditional)
