"""Short ticket classification control, separate from action planning."""
from itertools import permutations
import json
import requests
from tokenizers import Tokenizer
import laya_joint as joint

root = joint.ROOT / 'semantic-positive-control-v1'
root.mkdir(parents=True, exist_ok=False)
cases = [('technical', 'The app crashes immediately after login. I cannot open its settings.'),
         ('billing', 'The service works, but my credit card was charged twice. Please refund the duplicate payment.'),
         ('other', 'Thank you! I like the new logo. No question or problem to report.')]
labels = {'technical': 'Application malfunction or technical support',
          'billing': 'Payment, invoice, charge, or refund issue',
          'other': 'Other feedback unrelated to technical or billing problems'}
instruction = 'Classify this customer support ticket.'
tokenizer = Tokenizer.from_file(str(joint.ROOT / 'laya-tokenizer/tokenizer.json'))
tokenizer.no_truncation(); tokenizer.no_padding()
enc = lambda s: len(tokenizer.encode(s, add_special_tokens=False).ids)
session = requests.Session(); session.trust_env = False
joint.save(root / 'preregistered.json', {'cases': cases, 'labels': labels,
    'orders': list(permutations(labels)), 'calls_planned': 18,
    'script_sha256': joint.file_sha(__file__), 'no_scene_actions_or_qwen': True})
records = []
for expected, ticket in cases:
    for order in permutations(labels):
        state = {'ticket': ticket}
        criteria = {f'c{i}': labels[k] for i, k in enumerate(order)}
        body = {'model': 'english', 'state': state, 'max_len': 8192, 'head_max_len': 512,
            'questions': {'next_action': {'type': 'choice', 'instructions': instruction, 'criteria': criteria}}}
        head = 3 + enc('choice question: ' + instruction) + sum(enc(k + ': ' + v) + 1 for k, v in criteria.items())
        state_tokens = enc(json.dumps(state, ensure_ascii=False))
        preflight = {'head_tokens': head, 'state_tokens': state_tokens,
            'input_tokens': head + state_tokens + 1, 'option_tokens': [enc(k + ': ' + v) for k, v in criteria.items()]}
        assert max(preflight['option_tokens']) <= 48 and preflight['input_tokens'] <= 8192
        folder = root / (expected + '-' + '-'.join(order))
        joint.save(folder / 'request.json', body)
        response = session.post('http://127.0.0.1:18180/v1/systemone', json=body, timeout=(10, 120))
        (folder / 'response.txt').write_text(response.text)
        response.raise_for_status()
        wire = response.json(); usage = wire['usage']
        assert not usage['truncated'] and usage['state_tokens_dropped'] == 0
        assert usage['input_tokens'] == preflight['input_tokens'] and usage['state_tokens'] == state_tokens
        choice = wire['answers']['next_action']['choice']; chosen = order[int(choice[1:])]
        row = {'expected': expected, 'chosen': chosen, 'order': order, 'choice': choice,
            'passed': chosen == expected, 'usage': usage, 'preflight': preflight}
        records.append(row)
        joint.save(folder / 'receipt.json', row)
        joint.save(root / 'results.json', {'calls': len(records),
            'passed': sum(r['passed'] for r in records), 'records': records})
        print(expected, order, chosen, flush=True)
