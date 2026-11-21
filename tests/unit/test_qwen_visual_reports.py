"""Exercise the real dispatcher at its HTTP boundary using recorded schemas."""
import base64
from copy import deepcopy
from hashlib import sha256
import io
import json
from pathlib import Path

from PIL import Image
import pytest
from jsonschema import Draft202012Validator

from tests.unit.test_qwen_evidence_selector import fresh, mock_response
from qwen_evidence_selector import EvidenceSelectionError, image_parts
from qwen_step2_context import decode_tables
from qwen_visual_reports import COORDINATOR, WORKER


def picture():
    data = io.BytesIO()
    Image.new('RGB', (1000, 1000), (10, 90, 190)).save(data, format='PNG')
    raw = data.getvalue()
    return raw, {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(raw).decode()}}


def controller_input():
    fixture = json.loads((Path(__file__).parents[1] / 'fixtures/qwen_step2_exact_quotes.json').read_text())
    raw, image = picture()
    state = {'description': fixture['description'], 'allowed_target_ids': fixture['allowed_target_ids'],
             'active_target_ids': fixture['allowed_target_ids'],
             'objects': [{'id': oid, 'category': 'lamp'} for oid in fixture['known_object_ids']],
             'views': [{'kind': 'predicted_instance_overview', 'marked_ids': [12, 33]}],
             'previous_actions': [], 'observations': [],
             'supporting_evidence_ids': ['initial'], 'supporting_evidence_objects': {'initial': [12, 33]},
             'monitor_requirements': fixture['monitor_requirements'], 'task_dag': None}
    messages = [{'role': 'system', 'content': 'Native state and tool contract.'},
                {'role': 'user', 'content': [{'type': 'text', 'text': json.dumps(state)}, image]}]
    from eviground.grounding.dag_monitor import DagMonitor
    monitor = DagMonitor(state['description'], {}, fixture['known_object_ids'], fixture['allowed_target_ids'],
                         ['measure', 'inspect', 'finish'])
    monitor.requirements = fixture['monitor_requirements']
    action = fixture['positive_real_action']['action']
    native = {'anyOf': [{'type': 'object', 'properties': {'monitor': monitor.schema(compact=True)}}]}
    return raw, state, messages, monitor, native, action


def child_reply(role, context):
    if role == WORKER:
        item = {'object_ids': [12], 'status': 'unknown', 'observation': 'The object is partly occluded here.'}
        return {'image_id': context['task']['image_id'], 'findings': [item]}
    if 'images' in context:
        return {'tasks': [{'image_id': row['image_id'], 'question': context['question'], 'object_ids': [12]}
                          for row in context['images']]}
    return {'groups': [{'assessment': 'insufficient', 'finding_ids': [row['finding_id'] for row in context['findings']]}]}


def read_context(payload):
    value = payload['messages'][1]['content']
    return json.loads(value if isinstance(value, str) else value[0]['text'])


@pytest.mark.parametrize('levels', [2, 3])
def test_concrete_recheck_routes_original_pixels_only_to_worker(tmp_path, monkeypatch, levels):
    raw, state, messages, monitor, native, action = controller_input()
    direct = fresh.DirectCalls(tmp_path, visual_reports=True, visual_levels=levels)
    direct.monitor, direct.key = monitor, 'unit-mock'
    roles, main_states = [], []
    question = 'Can you tell whether lamp 12 is next to a bed in this view?'

    def post(_url, *, json, **_kwargs):
        role = json['response_format']['json_schema']['name'].removesuffix('_answer')
        roles.append(role)
        pictures = image_parts(json['messages'])
        if role == WORKER:
            assert len(pictures) == 1
            assert base64.b64decode(pictures[0]['image_url']['url'].split(',', 1)[1]) == raw
        else:
            assert not pictures
        ctx = read_context(json)
        if role == COORDINATOR and 'findings' in ctx:
            groups = json['response_format']['json_schema']['schema']['properties']['groups']
            count = len(ctx['findings'])
            assert groups['maxItems'] == count
            assert groups['items']['properties']['finding_ids']['maxItems'] == count
        if role == 'semantic_consultant':
            view = decode_tables(ctx)['state']
            main_states.append(view)
            assert len(view['visual_reports']) == len(main_states)
            finding = view['visual_reports'][-1]['findings'][0]
            catalog = {row['image_id']: row['sha256'] for row in view['image_catalog']}
            assert catalog[finding['image_id']] == sha256(raw).hexdigest()
            assert view['visual_reports'][-1]['findings'][0]['status'] == 'unknown'
            assert view['supporting_evidence_objects']['initial'] == []
            if len(main_states) == 1:
                answer = {'state_version': fresh.digest(state), 'action': {'evidence_request':
                          {'image_ids': ['img_0001'], 'question': question}}}
                fresh.validate(answer, json['response_format']['json_schema']['schema'])
                return mock_response(answer, prompt=100)
            return mock_response({'state_version': fresh.digest(state), 'action': action}, prompt=100)
        return mock_response(child_reply(role, ctx), prompt=100)

    monkeypatch.setattr(direct.session, 'post', post)
    before = deepcopy(messages)
    result = direct.choose(messages, response_schema={'schema': native})
    assert json.loads(result['_content']) == action
    step_roles = [COORDINATOR, WORKER, COORDINATOR] if levels == 3 else [WORKER]
    assert roles == [*step_roles, 'semantic_consultant'] * 2
    assert main_states[-1]['visual_reports'][-1]['question'] == question
    assert main_states[0]['allowed_target_ids'] == state['allowed_target_ids']
    assert messages == before and monitor.plan is None
    assert direct.r_used == len(roles) and direct.t_used == 1
    assert all(row['preflight']['maximum'] < 32768 for row in direct.ledger)
    assert sum(bool(row['input_images']) for row in direct.ledger) == 2


@pytest.mark.parametrize('bad_finding', [None, 'bad_status', 'unmarked_object'])
@pytest.mark.parametrize('wrap_report', [False, True])
def test_recorded_worker_array_preserves_findings_and_validation(tmp_path, monkeypatch, bad_finding, wrap_report):
    _, state, messages, monitor, native, action = controller_input()
    state['views'][0]['marked_ids'] = [12, 33, 34, 42]
    messages[1]['content'][0]['text'] = json.dumps(state)
    findings = json.loads((Path(__file__).parents[1] / 'fixtures/qwen_step2_worker_findings_array.json').read_text())
    if bad_finding == 'bad_status':
        findings[0]['status'] = 'certain'
    if bad_finding == 'unmarked_object':
        state['views'][0]['marked_ids'].remove(33)
        messages[1]['content'][0]['text'] = json.dumps(state)
    direct = fresh.DirectCalls(tmp_path, visual_reports=True)
    direct.monitor, direct.key = monitor, 'unit-mock'
    provider_answer = [{'image_id': 'img_0001', 'findings': findings}] if wrap_report else findings

    def post(_url, *, json, **_kwargs):
        role = json['response_format']['json_schema']['name'].removesuffix('_answer')
        ctx = read_context(json)
        if role == WORKER:
            return mock_response(provider_answer, prompt=1650)
        if role == 'semantic_consultant':
            assert not image_parts(json['messages'])
            delivered = decode_tables(ctx)['state']['visual_reports'][0]['findings']
            assert all(row['image_id'] == 'img_0001' for row in delivered)
            assert [{key: row[key] for key in original} for row, original in zip(delivered, findings, strict=True)] == findings
            return mock_response({'state_version': fresh.digest(state), 'action': action}, prompt=100)
        return mock_response(child_reply(role, ctx), prompt=100)

    monkeypatch.setattr(direct.session, 'post', post)
    if bad_finding:
        with pytest.raises(EvidenceSelectionError):
            direct.choose(messages, response_schema={'schema': native})
        assert not any(row['role'] == 'semantic_consultant' and row['http_sent'] for row in direct.ledger)
        return

    result = direct.choose(messages, response_schema={'schema': native})
    assert json.loads(result['_content']) == action
    worker, = tmp_path.glob('calls/*-visual_worker')
    raw = json.loads((worker / 'provider-response.json').read_text())['choices'][0]['message']['content']
    assert json.loads(raw) == provider_answer
    assert json.loads((worker / 'response.json').read_text()) == {'image_id': 'img_0001', 'findings': findings}
    normalization = ('unwrapped_single_assigned_image_report' if wrap_report
                     else 'wrapped_worker_findings_with_assigned_image_id')
    assert json.loads((worker / 'receipt.json').read_text())['response_normalization'] == normalization
    assert all(row['preflight']['maximum'] < 32768 for row in direct.ledger)


def test_report_treats_unknown_as_insufficient_and_rejects_bad_object_provenance():
    _, state, messages, _monitor, _native, _action = controller_input()
    from qwen_visual_reports import VisualReports
    for bad_object in (True, False):
        def call(_tag, role, messages, _schema, _reserve):
            answer = child_reply(role, read_context({'messages': messages}))
            if bad_object and role == WORKER:
                answer['findings'][0].update(object_ids=[999], status='observed')
            if not bad_object and role == COORDINATOR and 'groups' in answer:
                answer['groups'][0]['assessment'] = 'consistent'
            return answer
        reports = VisualReports(call, lambda *_: None)
        if bad_object:
            with pytest.raises(EvidenceSelectionError, match='unknown or duplicate'):
                reports.collect_new(state, messages)
            assert not reports.reports
        else:
            reports.collect_new(state, messages)
            report, = reports.reports.values()
            assert report['groups'][0]['assessment'] == 'insufficient'
            assert report['findings'][0]['status'] == 'unknown'


@pytest.mark.parametrize('unknown', [False, True])
def test_planner_duplicate_target_reference_ids_only_normalized_at_dispatch(unknown):
    from qwen_visual_reports import VisualReports

    _, state, messages, _, _, _ = controller_input()
    raw_ids = [12, 33, 12, 999 if unknown else 33]
    saved = {}
    dispatched = []

    def call(_tag, role, messages, _schema, _reserve):
        request = read_context({'messages': messages})
        answer = child_reply(role, request)
        if 'tasks' in answer:
            answer['tasks'][0]['object_ids'] = raw_ids
        if role == WORKER:
            dispatched.append(request['task']['object_ids'])
        return answer

    reports = VisualReports(call, lambda run, name, value: saved.update({(run, name): value}))
    if unknown:
        with pytest.raises(EvidenceSelectionError, match='unknown or duplicate'):
            reports.collect_new(state, messages)
        assert not dispatched
    else:
        reports.collect_new(state, messages)
        assert dispatched == [[12, 33]]
        assert len(reports.reports) == 1
    assert saved['visual_0001', 'plan.json']['plan']['tasks'][0]['object_ids'] == raw_ids


def test_mask_and_text_capacity_still_apply_to_own_calls(tmp_path):
    _, _state, messages, _monitor, _native, _action = controller_input()
    direct = fresh.DirectCalls(tmp_path, visual_reports=True)
    from unittest.mock import patch
    with patch.object(direct.session, 'post') as send:
        with pytest.raises(EvidenceSelectionError, match='Only the visual worker'):
            direct._qwen('bad', COORDINATOR, messages, {'type': 'object'})
        long_request = deepcopy(messages)
        long_request[1]['content'][0]['text'] = 'a' * 32768
        with pytest.raises(fresh.ContextCapacityExceeded, match='no request sent'):
            direct._qwen('long', WORKER, long_request, {'type': 'object'})
        send.assert_not_called()
    assert all(not row['http_sent'] for row in direct.ledger)


def test_generation_enforces_tool_context_and_pending_delta_fields():
    _, state, _, _, _, action = controller_input()
    state['task_dag'] = action['monitor']
    pending = state['task_dag']['nodes'][0]
    update = {name: pending[name] for name in (
        'id', 'status', 'bound_object_ids', 'evidence_ids', 'conclusion', 'excluded_target_ids')}
    action = {**action, 'monitor': {
        'node_updates': [update], 'current_node_id': update['id'],
        'reopen_node_ids': [], 'revision_reason': 'Continue taking evidence.'}}
    schema = {'properties': {'state_version': {'const': 'v1'}, 'action': {}}}
    prepared = fresh.prepare_qwen_payload([], schema, 'semantic_consultant', 3072,
                                          {**state, 'views': []}, text_visual_reports=True)
    validator = Draft202012Validator(prepared.payload['response_format']['json_schema']['schema'])
    proposal = {'state_version': 'v1', 'action': action}
    assert validator.is_valid(proposal)
    action.update(action='measure', context='scene')
    assert not validator.is_valid(proposal)
    action.update(context=None, azimuth_deg=None, elevation_deg=None)
    assert validator.is_valid(proposal)
    update['excluded_target_ids'] = [12]
    assert not validator.is_valid(proposal)


def test_tentative_repair_keeps_delivered_reports_and_has_no_pixels(tmp_path, monkeypatch):
    _, state, messages, monitor, _, _ = controller_input()
    direct = fresh.DirectCalls(tmp_path, visual_reports=True)
    direct.key, direct.monitor = 'unit-mock', monitor
    attempts = []
    answer = {'state_version': fresh.digest(state), 'predicted_id': None,
              'reason': 'The relative position remains unknown.', 'evidence': [], 'main_rivals': [],
              'unresolved': ['Occlusion left the relation unknown.']}

    def post(_url, *, json, **_kwargs):
        role = json['response_format']['json_schema']['name'].removesuffix('_answer')
        if role != 'semantic_consultant':
            return mock_response(child_reply(role, read_context(json)), prompt=100)
        assert not image_parts(json['messages'])
        ctx = decode_tables(read_context(json))
        attempts.append(ctx['state']['visual_reports'])
        if len(attempts) == 1:
            return mock_response({**answer, 'reason': 'Unsupported source img_9999.'}, prompt=100)
        assert 'img_9999' in ctx['context_and_exact_feedback'][-1]['previous_raw_answer']
        return mock_response(answer, prompt=100)

    monkeypatch.setattr(direct.session, 'post', post)
    with pytest.raises(fresh.TentativeFinal) as exc:
        direct.tentative_final(messages, state, 'offline_test')
    assert exc.value.answer == answer and len(attempts) == 2
    assert attempts[0] and attempts[0] == attempts[1] and monitor.plan is None
    assert all(row['preflight']['maximum'] < 32768 for row in direct.ledger)
