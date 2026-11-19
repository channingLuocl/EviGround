"""Synthetic action-selection probes; no scene actions or Qwen calls are executed."""
from copy import deepcopy
import json
import time
import requests
from tokenizers import Tokenizer
import laya_joint as joint

tokenizer = Tokenizer.from_file(str(joint.ROOT / 'laya-tokenizer/tokenizer.json'))
tokenizer.no_truncation(); tokenizer.no_padding()
session = requests.Session(); session.trust_env = False
root = joint.ROOT / 'semantic-probes-v3'
root.mkdir(parents=True, exist_ok=False)
records = []
cases = [
    ('unknown', 'Current node: verify leather material. Geometry has already been measured. Material is UNKNOWN; no material image was delivered. A repeated size measurement adds no information.', 'inspect'),
    ('refuted', 'Current node: repair the leather-material hypothesis. Image e02 clearly shows polymer and refutes leather. Geometry is already measured. Need semantic plan revision; neither more size measurements nor another identical view repairs the false binding.', 'consult'),
    ('missing_geometry', 'Current node: determine which of objects 1 and 2 is taller. Image e02 confirms leather for both. NO height measurements exist. Need a new geometry measurement, not a material inspection.', 'measure'),
    ('deleted_fact', 'Current node: verify leather material. Geometry has already been measured. There is an image source index e02 but its observations have NOT been delivered. The index proves no material fact.', 'inspect'),
]
for name, facts, expected in cases:
    for position, order in [(p, o) for p in ['start', 'middle', 'end'] for o in range(3)]:
        padding = ['Unrelated archived scene metadata; no evidence about this task.'] * (400 if name == 'unknown' else 6)
        index = {'start': 0, 'middle': len(padding)//2, 'end': len(padding)}[position]
        padding.insert(index, 'CURRENT TASK FACTS: ' + facts)
        state = {'description': 'Find the leather chair and verify relative height when that stage is ready.',
            'current_node': facts.split('.')[0], 'accepted_bindings': {'chair_candidates': [1, 2]},
            'accepted_exclusions': [], 'remaining_tools': 8, 'remaining_consult_or_repair': 3,
            'source_directory': [{'evidence_id': 'e02', 'sections': ['images']}],
            'records': padding, 'object_directory': [{'id': 1}, {'id': 2}],
            'allowed_target_ids': [1, 2]}
        options = [
            {'action': 'inspect', 'focus_ids': [1, 2], 'context': 'isolated',
                'azimuth_deg': 90, 'elevation_deg': 30, 'reason': 'Acquire new images to establish unknown material.'},
            {'action': 'measure', 'measurement_type': 'object_geometry', 'target_ids': [1, 2],
                'reference_ids': [], 'frame': None, 'node_id': 'n0', 'reason': 'Obtain exact heights and extents.'},
            {'action': 'consult', 'source_ids': ['e02'], 'sections': ['images'],
                'image_ids': ['e02-v0'], 'question': 'Review contradiction and explicitly revise the semantic plan.',
                'reason': 'Ask the semantic reasoner to review evidence and revise its plan.'},
        ]
        if order == 1:
            options = [options[2], options[0], options[1]]
        if order == 2:
            options.reverse()
        proposal = {'state_version': joint.digest(state), 'monitor': {'current_node_id': 'n0'},
            'decision_basis': 'Choose an information-gaining action that resolves the current node, preserving unknown versus contradicted evidence.',
            'options': deepcopy(options)}
        # Each option uses a complete fixed synthetic domain; only selection is probed here.
        body, preflight = joint.laya_request(state, proposal, options, tokenizer)
        folder = root / f'{name}-{position}-order{order}'
        joint.save(folder / 'request.json', body)
        started = time.monotonic()
        response = session.post('http://127.0.0.1:18180/v1/systemone', json=body, timeout=(10, 120))
        response.raise_for_status()
        result = response.json()
        joint.save(folder / 'response.json', result)
        usage = result['usage']
        assert not usage['truncated'] and usage['state_tokens_dropped'] == 0
        assert usage['state_tokens'] == preflight['state_tokens']
        assert usage['input_tokens'] == preflight['input_tokens']
        label = result['answers']['next_action']['choice']
        chosen = body['state']['actions'][label]['action']
        row = {'case': name, 'position': position, 'order': order, 'expected': expected, 'chosen': chosen,
            'passed': chosen == expected, 'choice': label, 'usage': usage,
            'probabilities': result['answers']['next_action']['probabilities'],
            'latency_s': round(time.monotonic()-started, 3), 'preflight': preflight}
        records.append(row)
        joint.save(root / 'results.json', {'scope': 'synthetic diagnostic; not an episode or a long-context quality benchmark',
            'entry_sha256': joint.file_sha(joint.__file__), 'probe_sha256': joint.file_sha(__file__),
            'qwen_calls': 0, 'laya_calls': len(records), 'scene_actions_executed': 0,
            'passed': sum(r['passed'] for r in records), 'results': records})
        print(json.dumps(row), flush=True)
