"""Isolated Laya action selection with a bounded Token Plan semantic consultant."""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
import io
import json
from pathlib import Path
import shlex
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / 'outputs/laya-joint-20261119'
FROZEN = ROOT / 'frozen'
MODEL = 'qwen3.8-max'
LIMITS = {'B': 10, 'R': 6, 'T': 17, 'decision_output_reserved': 4000}
TEXT_OVERHEAD = 1024
# Qwen3.8 official vision documentation: at most 16384 patches plus two markers.
IMAGE_UPPER_BOUND = 16386
VISION_SOURCE = 'https://help.aliyun.com/zh/model-studio/vision'


def packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(packed(value).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def factor_schema(schema):
    """Losslessly share repeated schema objects, including bare enum domains."""
    counts, names, definitions = {}, {}, {}

    def visit(value, replace=False, root=False):
        if isinstance(value, list):
            return [visit(item, replace) for item in value]
        if not isinstance(value, dict):
            return value
        key = packed(value)
        if not replace:
            counts[key] = counts.get(key, 0) + 1
        schema_node = (isinstance(value.get('type'), (str, list)) or
            isinstance(value.get('anyOf'), list) or isinstance(value.get('enum'), list))
        if replace and schema_node and not root and len(key) >= 80 and counts.get(key, 0) > 1:
            if key not in names:
                name = 'part_' + str(len(names))
                names[key] = name
                definitions[name] = visit(value, True, True)
            return {'$ref': '#/$defs/' + names[key]}
        return {k: v if k in ('enum', 'const', 'default', 'examples') else visit(v, replace)
                for k, v in value.items()}

    assert '$defs' not in schema, 'Factor only the original unfactored contract'
    visit(schema)
    result = visit(schema, True, True)
    result['$defs'] = definitions
    return result if len(packed(result)) < len(packed(schema)) else schema


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


class TransportStop(RuntimeError):
    """A transport/provider failure requires inspection before any further send."""


def window(_cpu, messages, reserve, response_format=None):
    """Conservative UTF-8 byte bound, not a surrogate model's exact token count."""
    text_messages, images = deepcopy(messages), []
    from PIL import Image
    for message, original in zip(text_messages, messages):
        if not isinstance(message['content'], list):
            continue
        for part, original_part in zip(message['content'], original['content']):
            if part['type'] != 'image_url':
                continue
            uri = original_part['image_url']['url']
            assert uri.startswith('data:image/') and ';base64,' in uri
            data = base64.b64decode(uri.split(',', 1)[1], validate=True)
            with Image.open(io.BytesIO(data)) as image:
                width, height = image.size
                assert width > 0 and height > 0
            images.append({'sha256': hashlib.sha256(data).hexdigest(), 'width': width,
                'height': height, 'visual_tokens_upper_bound': IMAGE_UPPER_BOUND})
            part['image_url']['url'] = '<image tokens accounted separately>'
    text_bytes = len(packed({'messages': text_messages,
        'response_format': response_format}).encode())
    maximum = text_bytes + TEXT_OVERHEAD + IMAGE_UPPER_BOUND * len(images) + reserve
    return {'method': 'utf8_byte_upper_bound_plus_full_image_cap',
        'assumptions': ['Qwen text uses byte-level BPE; bytes bound text tokens',
            '1024-token allowance for chat/schema framing',
            'each actual image uses at most 16384 visual tokens plus two markers'],
        'exact_tokenizer': False, 'text_bytes': text_bytes,
        'framing_allowance': TEXT_OVERHEAD, 'maximum': maximum,
        'output_reserved': reserve, 'actual_image_parts': len(images), 'images': images,
        'vision_source': VISION_SOURCE}


def validate_proposal(proposal, version, original_schema):
    from jsonschema import validate
    assert proposal['state_version'] == version, 'Stale accepted state version'
    assert 2 <= len(proposal['options']) <= 3, 'Need two or three model-proposed actions'
    actions, identities = [], set()
    for option in proposal['options']:
        action = {**deepcopy(option), 'monitor': deepcopy(proposal['monitor'])}
        validate(action, original_schema)
        if action['action'] == 'measure':
            left, right = action['target_ids'], action['reference_ids']
            assert len(set(left)) == len(left) and len(set(right)) == len(right)
            assert not set(left) & set(right), 'Target/reference groups overlap'
            assert len(set(left + right)) <= 8
            current = proposal['monitor'].get('current_node_id',
                proposal['monitor'].get('replace_plan', {}).get('current_node_id'))
            assert action['node_id'] == current, f"All alternatives must address shared current node {current}; measure.node_id={action['node_id']} is a different stage"
        if 'focus_ids' in action:
            assert len(set(action['focus_ids'])) == len(action['focus_ids'])
        identity = digest({k: v for k, v in option.items() if k not in ('reason', 'question')})
        assert identity not in identities, 'Narration-only alternatives are not distinct actions'
        identities.add(identity)
        actions.append(action)
    return actions


def charge_slow_consultation(used):
    assert type(used) is int and 0 <= used < 6, 'No slow consultation/repair slots remain'
    return used + 1


def shared_monitor_schema(action_schema):
    variants = {}
    for branch in action_schema['anyOf']:
        monitor = branch['properties']['monitor']
        variants.setdefault(packed(monitor), deepcopy(monitor))
    return next(iter(variants.values())) if len(variants) == 1 else {'anyOf': list(variants.values())}


def node_map_schema(schema):
    """Move a node's literal ID to its object key; preserve all body constraints."""
    if isinstance(schema, list):
        return [node_map_schema(v) for v in schema]
    if not isinstance(schema, dict):
        return schema
    result = {k: v if k in ('enum', 'const', 'default', 'examples') else node_map_schema(v)
        for k, v in schema.items()}
    nodes = result.get('properties', {}).get('nodes')
    if nodes and nodes.get('type') == 'array':
        groups = {}
        for branch in nodes['items']['anyOf']:
            branch = deepcopy(branch)
            ids = branch['properties'].pop('id')['enum']
            assert ids and all(isinstance(name, str) for name in ids)
            branch['required'].remove('id')
            for name in ids:
                groups.setdefault(name, []).append(deepcopy(branch))
        result['properties']['nodes'] = {'type': 'object', 'properties': {
            name: bs[0] if len(bs) == 1 else {'anyOf': bs} for name, bs in groups.items()},
            'required': [], 'additionalProperties': False,
            'minProperties': nodes['minItems'], 'maxProperties': nodes['maxItems']}
    return result


def decode_node_maps(value):
    if isinstance(value, list):
        return [decode_node_maps(v) for v in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, child in value.items():
        if key == 'nodes':
            assert isinstance(child, dict), 'Wire nodes must be an ID-keyed object'
            result[key] = [{'id': name, **decode_node_maps(body)} for name, body in child.items()]
        else:
            result[key] = decode_node_maps(child)
    return result


def unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON field: ' + key)
        result[key] = value
    return result


def laya_request(state, proposal, actions, tokenizer):
    instruction = 'Choose the best legal next action using the complete state and consultant analysis.'
    criteria = {}
    for i, action in enumerate(actions):
        keys = {'inspect': ('focus_ids', 'context'),
            'measure': ('measurement_type', 'target_ids', 'reference_ids', 'frame'),
            'consult': ('source_ids', 'sections'), 'commit': (),
            'finish': ('predicted_id',)}[action['action']]
        parameters = {k: action[k] for k in keys if k != 'frame'}
        criteria[f'c{i}'] = action['action'] + ': ' + action['reason'] + ' ' + packed(parameters)
    contract = {'accepted_state': state, 'accepted_state_version': digest(state),
        'semantic_monitor_proposal': proposal['monitor'],
        'consultant_analysis': proposal['decision_basis'],
        'legal_actions': ['inspect', 'measure', 'consult', 'commit', 'finish'],
        'parameter_contract': {'inspect_max_focus': 6, 'measure_max_union': 8,
            'azimuth': '[0,360)', 'elevation': '[10,75]',
            'object_ids': 'exact IDs in accepted_state.object_directory',
            'target_ids': 'exact accepted_state.allowed_target_ids for final predictions',
            'evidence_ids': 'exact accepted_state.source_directory; index is not observation',
            'all_options_validated_against_full_original_schema': True},
        'actions': {f'c{i}': action for i, action in enumerate(actions)},
        'visibility': 'All original planner state retained verbatim; no raw pixels. Semantic evidence judgments are the consultant responsibility.'}
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False).ids
    option_counts = [len(enc(f'{k}: {v}')) for k, v in criteria.items()]
    assert max(option_counts) <= 48, 'Laya caps each option text at 48 tokens'
    assert len(set(criteria.values())) == len(criteria), 'Option descriptions must distinguish concrete alternatives'
    instruction_count = len(enc('choice question: ' + instruction))
    assert instruction_count <= max(8, 512 - sum(n + 1 for n in option_counts)), 'Question would truncate'
    state_count = len(enc(json.dumps(contract, ensure_ascii=False)))
    head_count = 3 + instruction_count + sum(n + 1 for n in option_counts)
    assert head_count + state_count + 1 <= 8192, f'Laya complete contract needs {head_count + state_count + 1} tokens >8192; nothing was dropped'
    return {'model': 'english', 'state': contract, 'max_len': 8192, 'head_max_len': 512,
        'questions': {'next_action': {'type': 'choice', 'instructions': instruction, 'criteria': criteria}}}, {
        'state_tokens': state_count, 'head_tokens': head_count,
        'input_tokens': head_count + state_count + 1, 'option_tokens': option_counts,
        'full_original_state_sha256': digest(state), 'visibility_changed': False}


SLOW_RULES = '''You are the slow semantic reasoner and semantic Monitor for a 3D grounding task.
The fast Laya model makes the final concrete action choice. Return a shared explicit monitor proposal
and TWO or THREE genuinely distinct complete legal next-action options for the SAME current node.
Use the exact state_version provided. Put your evidence interpretation and action-selection reasoning
in decision_basis (<=600 chars). Each option must have all original action fields except monitor.
Do not rank options by an identifier, encode a preferred option, choose a final option for Laya, or
make alternatives differ only in narration. A finish with target and a finish null are different options.
All state updates, exclusions, target/reference/comparison membership and parameters must be YOUR
explicit proposals. The executor never fills missing bindings or semantic node completions.

Use only supplied evidence. Category ranks are hypotheses, not appearance observations. Complete a
node only when its observed predicate and cited evidence support it; unknown is not a contradiction.
Reference bindings cannot exclude targets. Each target exclusion needs a covered observation that
contradicts that rival. Preserve every original requirement, uncertainty, comparison group and frame.
For a query without an observer, use the authorized room-center-facing-model-selected-reference
convention; camera-left/world-X is not language-left. Numerical facts require measurements.
Initial source indexes are not seen images. Request actual images before claiming appearance.
For isolated inspect, use at most four focus IDs so visual reports can share the output allowance.
Keep complete data for relevant groups; do not programmatically infer a missing reference or group.

Only verification permits finish. Non-null finish requires all nodes complete, all clauses supported,
every rival resolved by evidence, unresolved=[], and final evidence quoting the complete description.
If evidence is insufficient, abstain honestly. Rollback reopens descendants and reactivates exclusions.
Use commit to submit evidence-backed state changes; commit does not acquire evidence or finish.
Do not repeat tools or archive reads. Typed measurement uses the exact target/reference groups;
object_geometry has reference_ids=[]/frame=null; pair_geometry groups are nonempty and disjoint.
All options must use the exact original schema, IDs, evidence indexes and accepted state version.
In EVERY measure option, node_id must equal the shared monitor.current_node_id, never a later node.
Keep answers compact. Each new slow primary consultation costs one of the original six R slots.
Each option reason must be a short natural-language selection description (at most 80 characters);
the entire head label, including action type and parameters, must fit 48 Laya tokens. Full arguments
and your shared semantic analysis remain in the state; never omit constraints to make a short label.
Physical success B<=10, action decisions T<=17. Total role output reservations per decision <=4000.
'''

PRIMARY_RULES = '''Text-only 3D grounding reasoner/Monitor; Laya selects. Obey FULL JSON decoder: exact state_version,
shared monitor, 2-3 complete options, decision_basis. YOU propose bindings/groups/exclusions/updates/
parameters; code validates/executes. No ranking/selection/preference encoding. Options differ in
parameters, not narration; target/null finishes differ. SAME current node; measure.node_id equals it. Reasons<=80 chars;
verb+reason+parameters<=48 Laya tokens; retain full parameters/analysis.

Only delivered evidence counts; indexes/catalogs are not observations. Unknown/unseen is not false.
Categories are uncertain/exclusive per ID. Top-1 is a display prior; bind targets/references using
retained hypotheses. Rank/gap are classifier evidence, not truth/calibrated thresholds; frontend scores
are incomparable. Lower ranks need description+visual/spatial support. Retention/
rejection does not prove presence/absence. Actual images are required for appearance/material/color.
Reports are fallible; retain source/image IDs; cite ALL images per visual claim. Images give qualitative
appearance/direction; numbers need supplied measurements. Isolated panels have independent scales/
cameras; placement is not scene position. Camera-left/world-X is not
language-left. No observer: use authorized room-center-facing-model-selected-reference policy.
Preserve clauses/uncertainty/negation/references/complete comparison groups/frames.

Authoritative task_dag has separate REQUIRED verification n11. q0 (whole description) maps to n11,
never a prerequisite binding. Cover EVERY q; derive tasks/order from query/evidence.
Small DAG; unique node IDs; dependency/requirement references must exist. Nonempty revision_reason.
Dependencies/current_node_id use NODE IDs. Verification depends transitively on ALL tasks and alone
allows finish. Tool node must be pending, allow tool, have completed dependencies; never use n11.
Exact source substrings; actual bindings, not all inspected alternatives; no duplicate list entries.
Pending evidence_ids/conclusion/excluded_target_ids are empty. Completion needs observed conclusions
and cited delivered support, never future tools/errors/feedback/copied query. Reference binding
excludes NO targets. Exclude only in a completed node with cited evidence covering AND contradicting
that rival. Keep unresolved conditions pending; never manufacture uniqueness.

Existing DAG: submit deltas; omit unchanged nodes (empty allowed). Retain exact goals,
dependencies, quotes and requirement assignments. To change those/add/remove tasks, replace_plan
supplies a COMPLETE revised DAG. Overturned bindings/conclusions require reopen_node_ids; descendants
reset, exclusions reactivate. Read feedback/ready_node_ids; retain unchanged completions
verbatim; do not repeat rejected graphs. Required category reinspection reopens an evidence node.
Update supported tasks before choosing next action. commit records supported changes only; no new
evidence or finish. Tool calls/reports alone never complete tasks. Only finish may complete n11.
Non-null finish: ALL tasks completed, EVERY requirement supported, current_node_id=n11, unresolved=[],
supported exclusions leaving exactly predicted_id; final evidence quotes COMPLETE description
and explains strongest rivals. Update EVERY still-pending task (n11 first); omitted tasks stay pending.
Essential evidence missing: finish null; no deadline-forced exclusions. Verification: finish-only,
pending during tool use.

No repeated acquisitions/reads. Full relevant groups; exact schema IDs/sources.
object_geometry: reference_ids=[],frame=null. pair_geometry: nonempty disjoint target/reference groups.
Isolated inspect<=4 IDs. B<=10/R<=6/T<=17; slow consultation costs R1; roles share 4000 output tokens.
Tables: pair columns/row cells; merge constants into EVERY row. missing=[R,[C,...]] omits zero-based
columns, not null. nested_columns supplies columns for nested cell rows. Preserve values/order.
view_image_indices maps views to zero-based actual image parts; reused indices mean identical bytes.
'''


def prepare(version):
    destination = ROOT / 'versions' / version
    destination.mkdir(parents=True, exist_ok=False)
    source = FROZEN / 'prototype/runner.py'
    code = source.read_text()
    anchor = '            action = run.call(dataset + f\'-t{turn:02d}\', \'primary\', rules, state,'
    assert code.count(anchor) == 1
    code = code.replace(anchor,
        "            if used_ancillary >= 6:\n"
        "                result['status'] = 'slow_consultation_budget_exhausted'\n"
        "                break\n"
        "            used_ancillary = run.charge_slow_consultation(used_ancillary)\n"
        "            event['slow_consultation_R_cost'] = 1\n"
        "            state['remaining_consult_or_repair'] = 6-used_ancillary\n" + anchor)
    (destination / 'runner.py').write_text(code)
    (destination / 'entry.py').write_bytes(Path(__file__).read_bytes())
    manifest = {'version': version, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_snapshot': str(FROZEN), 'source_runner_sha256': file_sha(source),
        'runner_sha256': file_sha(destination / 'runner.py'), 'entry_sha256': file_sha(__file__),
        'limits': LIMITS, 'model': MODEL, 'laya': 'english; max_len8192; head_max_len512',
        'counting': {'primary': 'one final Laya action choice costs T=1',
            'new_slow_primary_consultation': 'R=1 before send, including failed responses',
            'visual': 'original tool/consult worker; shares the same 4000-token output reservation',
            'invalid_action_or_repair': 'original additional R=1',
            'physical_and_archive': 'unchanged frozen runner B/R definitions',
            'parser': 'original separate stage; additional catalog repair R=1'},
        'protocol_extension': 'R now also charges new slow semantic consultations; exploratory diagnostic, not a controlled pure-Qwen comparison',
        'visibility': 'Qwen gets full planner state; Laya gets identical state plus complete proposed actions/Monitor. No state summarization or truncation.',
        'counter': window(None, [], 0), 'formal_result': False,
        'inherited_model_calls_or_state': False,
        'frozen_file_hashes': {str(p.relative_to(FROZEN)): file_sha(p)
            for p in sorted(FROZEN.rglob('*')) if p.is_file() and '__pycache__' not in str(p)},
        'tokenizer_sha256': file_sha(ROOT / 'laya-tokenizer/tokenizer.json')}
    save(destination / 'manifest.json', manifest)
    return manifest


def load_runtime(version):
    candidate = ROOT / 'versions' / version
    manifest = json.loads((candidate / 'manifest.json').read_text())
    assert manifest['entry_sha256'] == file_sha(__file__), 'Entry changed after freeze'
    assert manifest['runner_sha256'] == file_sha(candidate / 'runner.py')
    for name, expected in manifest['frozen_file_hashes'].items():
        assert file_sha(FROZEN / name) == expected, name
    assert file_sha(ROOT / 'laya-tokenizer/tokenizer.json') == manifest['tokenizer_sha256']
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(candidate), str(FROZEN / 'prototype'), str(FROZEN / 'src')]
    import common
    common.PROJECT, common.window = FROZEN, window
    runner = importlib.import_module('runner')
    runner.INPUTS = FROZEN / 'outputs/side_context_real_20261119_01/scenes'
    runner.window = window
    runner.ROOT_RULES = SLOW_RULES
    return common, runner


def run_episode(version, dataset, laya_url):
    common, runner = load_runtime(version)
    import requests
    from jsonschema import validate, Draft202012Validator, ValidationError
    from tokenizers import Tokenizer
    from eviground.api.client import _wire_schema
    output = ROOT / 'cases' / f'{version}-{dataset}'
    output.mkdir(parents=True, exist_ok=False)
    common.ROOT = output

    class JointRun(common.Run):
        charge_slow_consultation = staticmethod(charge_slow_consultation)

        def __init__(self):
            self.cpu = None
            self.model, self.backend = MODEL, 'tokenplan'
            self.receipts, self.last_call_id = [], None
            self.http = requests.Session()
            self.http.trust_env = False
            config = {}
            for line in Path('/Users/luocl/.config/seeground/qwen-token-plan-20261022.env').read_text().splitlines():
                key, sep, value = line.strip().removeprefix('export ').partition('=')
                if sep and not key.startswith('#'):
                    config[key.strip()] = shlex.split(value, comments=True)[0] if value.strip() else ''
            self.base, self.key = config['SEEGROUND_BASE_URL'].rstrip('/'), config['SEEGROUND_API_KEY']
            assert self.base == 'https://token-plan.maas.qianwenaiapi.com/compatible-mode/v1'
            self.tokenizer = Tokenizer.from_file(str(ROOT / 'laya-tokenizer/tokenizer.json'))
            self.tokenizer.no_truncation()
            self.tokenizer.no_padding()

        def qwen(self, tag, role, rules, data, schema, reserve, images=(), messages=None):
            reserved = getattr(self, 'decision_output_reserved', 0) + reserve
            if reserved > 4000:
                raise common.BudgetExceeded('Shared decision output reservation exceeds 4000; no HTTP sent')
            self.decision_output_reserved = reserved
            messages = self.messages(role, rules, data, images) if messages is None else messages
            original_system = messages[0]['content']
            wire_schema = schema
            if role == 'semantic_consultant':
                messages[0]['content'] = PRIMARY_RULES + '\nWire: nodes is an object keyed by YOUR node IDs; each value omits id. Omit unused keys. All node bodies, dependencies and Monitor semantics are unchanged.\n'
                wire_schema = node_map_schema(schema)
            response_format = {'type': 'json_schema', 'json_schema':
                _wire_schema({'name': role + '_answer', 'strict': True, 'schema': factor_schema(wire_schema)})}
            Draft202012Validator.check_schema(response_format['json_schema']['schema'])
            preflight = window(None, messages, reserve, response_format)
            name = f'{len(self.receipts)+1:03d}-{tag}-{role}'
            folder = output / 'calls' / name
            folder.mkdir(parents=True, exist_ok=False)
            self.last_call_id = name
            receipt = {'call_id': name, 'role': role, 'tag': tag, 'status': 'preflight',
                'requested_model': self.model, 'preflight': preflight,
                'decision_output_reserved': reserved, 'usage': None, 'http_sent': False}
            payload = {'model': self.model, 'messages': messages, 'max_tokens': reserve,
                'temperature': 0, 'enable_thinking': False, 'response_format': response_format}
            save(folder / 'request.json', payload)
            if role == 'semantic_consultant':
                (folder / 'original-system.txt').write_text(original_system)
            save(folder / 'schema.json', schema)
            save(folder / 'wire-schema.json', wire_schema)
            save(folder / 'receipt.json', receipt)
            self.receipts.append(receipt)
            started = time.monotonic()
            try:
                if preflight['maximum'] > 32768:
                    raise common.BudgetExceeded(f'Conservative Qwen preflight {preflight["maximum"]}>32768; full schema, text, actual images and output included; no HTTP sent')
                print('SEND', name, 'window<=', preflight['maximum'], flush=True)
                receipt['http_sent'] = True
                response = self.http.post(self.base + '/chat/completions', json=payload,
                    headers={'Authorization': 'Bearer ' + self.key}, timeout=(45, 180), allow_redirects=False)
                receipt['http_status'] = response.status_code
                clean = response.text.replace(self.key, '[REDACTED]')
                (folder / 'response.txt').write_text(clean)
                if response.status_code != 200:
                    raise TransportStop('Provider HTTP ' + str(response.status_code) + '; stopped without automatic retry')
                wire = json.loads(clean)
                receipt.update(reported_model=wire.get('model'), usage=wire.get('usage'),
                    finish_reason=wire['choices'][0].get('finish_reason'))
                assert wire['model'] == MODEL, 'Generation model ID mismatch'
                assert receipt['finish_reason'] == 'stop', 'Incomplete model output'
                assert type(wire['usage']['prompt_tokens']) is int
                assert wire['usage']['prompt_tokens'] <= preflight['maximum'] - reserve, 'Provider exceeded conservative input bound'
                assert wire['usage']['prompt_tokens'] + reserve <= 32768
                answer = json.loads(wire['choices'][0]['message']['content'], object_pairs_hook=unique_json_object)
                try:
                    validate(answer, wire_schema)
                    if role == 'semantic_consultant':
                        save(folder / 'wire-parsed.json', answer)
                        answer = decode_node_maps(answer)
                    validate(answer, schema)
                except ValidationError as exc:
                    error = common.contract_error(exc)
                    raise common.ContractFailure(f'Invalid model response at {list(error.absolute_path)}: {error.message}') from exc
                save(folder / 'parsed.json', answer)
                receipt['status'] = 'completed'
                return answer
            except Exception as exc:
                receipt.update(status='failed', error_type=type(exc).__name__, error=str(exc))
                raise
            finally:
                receipt['latency_s'] = round(time.monotonic() - started, 3)
                save(folder / 'receipt.json', receipt)
                save(output / 'model-ledger.json', self.receipts)

        def call(self, tag, role, rules, data, schema, reserve, images=(), *, messages=None):
            if role != 'primary':
                return self.qwen(tag, role, rules, data, schema, reserve, images, messages)
            state_version = digest(data)
            monitor_schema = shared_monitor_schema(schema)
            variants = deepcopy(schema['anyOf'])
            for branch in variants:
                branch['properties'].pop('monitor')
                branch['required'].remove('monitor')
            slow_schema = common.obj({'state_version': common.enum([state_version]),
                'decision_basis': {'type': 'string', 'minLength': 1, 'maxLength': 600},
                'monitor': monitor_schema, 'options': common.arr({'anyOf': variants}, 3, 2)})
            proposal = self.qwen(tag, 'semantic_consultant', rules,
                {**data, 'state_version': state_version}, slow_schema, reserve)
            actions = validate_proposal(proposal, state_version, schema)
            body, preflight = laya_request(data, proposal, actions, self.tokenizer)
            name = f'{len(self.receipts)+1:03d}-{tag}-laya'
            folder = output / 'calls' / name
            folder.mkdir(parents=True, exist_ok=False)
            receipt = {'call_id': name, 'role': 'fast_action_selector', 'status': 'started',
                'requested_model': 'english', 'state_version': state_version,
                'preflight': preflight, 'output_reserved': 0, 'http_sent': True}
            self.receipts.append(receipt)
            self.last_call_id = name
            save(folder / 'request.json', body)
            try:
                response = self.http.post(laya_url.rstrip('/') + '/v1/systemone', json=body,
                    timeout=(10, 120), allow_redirects=False)
                receipt['http_status'] = response.status_code
                save(folder / 'response.json', response.json())
                response.raise_for_status()
                wire = response.json()
                usage = wire['usage']
                assert usage['truncated'] is False and usage['state_tokens_dropped'] == 0
                assert usage['state_tokens'] == preflight['state_tokens']
                assert usage['input_tokens'] == preflight['input_tokens']
                choice = wire['answers']['next_action']['choice']
                assert choice in body['state']['actions']
                action = body['state']['actions'][choice]
                receipt.update(status='completed', usage=usage, selected_choice=choice,
                    model_report=wire['routing'], state_version=state_version)
                save(folder / 'parsed.json', action)
                return action
            except Exception as exc:
                receipt.update(status='failed', error_type=type(exc).__name__, error=str(exc))
                raise
            finally:
                save(folder / 'receipt.json', receipt)
                save(output / 'model-ledger.json', self.receipts)

    run = JointRun()
    save(output / 'episode.json', {'version': version, 'dataset': dataset, 'limits': LIMITS,
        'manifest_sha256': file_sha(ROOT / 'versions' / version / 'manifest.json'),
        'fresh_start': True, 'inherited_nodes': [], 'evaluation_labels_visible': False,
        'formal_result': False, 'started_utc': datetime.now(timezone.utc).isoformat()})
    try:
        result = runner.run_case(run, dataset, output)
    except (TransportStop, requests.RequestException) as exc:
        trace = json.loads((output / 'trace.json').read_text())
        result = {'status': 'transport_or_provider_stopped', 'dataset': dataset,
            'reason': str(exc), 'strict_monitor_completed': False, 'predicted_id': None,
            'successful_tools': trace['used_tools'], 'ancillary_slots': trace['used_ancillary'],
            'primary_decisions': len(trace['attempts']), 'inherited_primary_decisions': 0,
            'completed_nodes': [n['id'] for n in (trace['monitor_plan'] or {}).get('nodes', []) if n['status'] == 'completed']}
    result.update(joint_episode_success=False, evidence_audit='not_completed',
        prediction_correctness='not_evaluated', formal_result=False)
    save(output / 'result.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['prepare', 'run'])
    parser.add_argument('--version', required=True)
    parser.add_argument('--dataset', choices=['nr3d', 'scanrefer'], default='nr3d')
    parser.add_argument('--laya-url', default='http://127.0.0.1:18180')
    args = parser.parse_args()
    if args.operation == 'prepare':
        print(json.dumps(prepare(args.version), ensure_ascii=False, indent=2))
    else:
        run_episode(args.version, args.dataset, args.laya_url)
