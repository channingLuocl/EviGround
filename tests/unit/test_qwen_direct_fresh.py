"""One-model integration checks against original native controller contracts."""
import json
from pathlib import Path
import sys
import tempfile
import base64
import io
import pytest
from PIL import Image
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import qwen_direct_fresh as fresh
from tests.unit import test_dag_delta as fixtures
from tests.unit.test_dag_monitor import action


@pytest.fixture(autouse=True)
def native_fixture_image_adapter(monkeypatch):
    """Native fixture uses image-name stubs; supply explicit synthetic test pixels."""
    original = fresh.compact_call
    stream = io.BytesIO(); Image.new('RGB', (512, 512)).save(stream, format='PNG')
    url = 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()
    def adapt(messages, state=None, **kwargs):
        messages = fresh.deepcopy(messages)
        for message in messages:
            if isinstance(message.get('content'), list):
                for i, part in enumerate(message['content']):
                    if set(part) == {'image'}:
                        message['content'][i] = {'type': 'image_url', 'image_url': {'url': url}}
        return original(messages, state, **kwargs)
    monkeypatch.setattr(fresh, 'compact_call', adapt)


def test_bigmodel_visual_request_uses_bearer_and_keeps_local_schema_validation(tmp_path, monkeypatch):
    for name in ('MODEL', 'BASE_URL', 'OUTPUT_LIMIT'):
        monkeypatch.setattr(fresh, name, getattr(fresh, name))
    fresh.configure_bigmodel()
    monkeypatch.setattr(fresh, 'API_KEY', 'mock-bigmodel-key')
    pixels = io.BytesIO()
    Image.new('RGB', (8, 8)).save(pixels, format='PNG')
    encoded = base64.b64encode(pixels.getvalue()).decode()
    messages = [{'role': 'system', 'content': 'Look at the image.'},
                {'role': 'user', 'content': [
                    {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + encoded}},
                ]}]
    schema = fresh.obj({'answer': {'type': 'string'}})
    direct = fresh.DirectCalls(tmp_path)

    def post(url, *, json, headers, **kwargs):
        assert url == 'https://open.bigmodel.cn/api/paas/v4/chat/completions'
        assert headers == {'Authorization': 'Bearer mock-bigmodel-key'}
        assert kwargs['allow_redirects'] is False
        assert json['model'] == 'glm-4.5v' and json['thinking'] == {'type': 'disabled'}
        assert 'response_format' not in json and 'chat_template_kwargs' not in json
        assert '"required":["answer"]' in json['messages'][0]['content']
        assert json['messages'][1]['content'][0]['image_url']['url'] == encoded
        value = {'model': 'glm-4.5v', 'usage': {'prompt_tokens': 150, 'completion_tokens': 15},
                 'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}]}
        return SimpleNamespace(status_code=200, headers={}, json=lambda: value, text=fresh.packed(value))

    monkeypatch.setattr(direct.session, 'post', post)
    content = '<think>reasoning</think><|begin_of_box|>{"answer":"blue"}<|end_of_box|>'
    result = direct._qwen('accepted', 'visual_worker', messages, schema)
    assert json.loads(result['_content']) == {'answer': 'blue'}
    call = tmp_path / 'calls/001-accepted-visual_worker'
    assert (call / 'input-image-00.png').read_bytes() == pixels.getvalue()
    recorded = (call / 'request-redacted.json').read_text()
    assert 'mock-bigmodel-key' not in recorded and '[IMAGE_REDACTED]' in recorded
    content = '{"answer":1}'
    with pytest.raises(fresh.ValidationError):
        direct._qwen('rejected', 'visual_worker', messages, schema)


def test_bigmodel_accepts_only_matching_misplaced_action_version(tmp_path, monkeypatch):
    monkeypatch.setattr(fresh, 'MODEL', fresh.BIGMODEL_MODEL)
    direct = fresh.DirectCalls(tmp_path)
    schema = fresh.obj({'state_version': {'const': 'current'},
                        'action': fresh.obj({'action': {'const': 'inspect'}})})
    messages = [{'role': 'system', 'content': 'Choose an action.'}]

    def post(url, *, json, **kwargs):
        assert 'exactly two top-level keys' in json['messages'][-1]['content']
        value = {'model': fresh.MODEL, 'usage': {'prompt_tokens': 150, 'completion_tokens': 15},
                 'choices': [{'message': {'content': fresh.packed(answer)}, 'finish_reason': 'stop'}]}
        return SimpleNamespace(status_code=200, headers={}, json=lambda: value, text=fresh.packed(value))

    monkeypatch.setattr(direct.session, 'post', post)
    answer = {'action': {'action': 'inspect', 'state_version': 'current'}}
    result = direct._qwen('accepted', 'semantic_consultant', messages, schema)
    assert json.loads(result['_content']) == {'state_version': 'current', 'action': {'action': 'inspect'}}
    assert direct.ledger[-1]['response_normalization'] == 'lifted_matching_action_state_version'
    assert answer['action']['state_version'] == 'current'
    for rejected in ({'action': {'action': 'inspect', 'state_version': 'stale'}},
                     {'action': {'action': 'inspect'}},
                     {'action': {'action': 'invalid', 'state_version': 'current'}}):
        answer = rejected
        with pytest.raises(fresh.ValidationError):
            direct._qwen('rejected', 'semantic_consultant', messages, schema)


def test_bigmodel_visual_format_repair_preserves_task_image_and_audit(tmp_path, monkeypatch):
    from collections import Counter
    from audit_qwen_nr3d_context import audit_visual_batches
    from tests.unit.test_qwen_visual_reports import controller_input, child_reply, read_context

    monkeypatch.setattr(fresh, 'MODEL', fresh.BIGMODEL_MODEL)
    raw, state, messages, *_ = controller_input()
    state['views'][0]['marked_ids'] = [12]
    direct = fresh.DirectCalls(tmp_path, visual_reports=True)
    attempts = Counter()

    def post(_url, *, json, **_kwargs):
        ctx = read_context(json)
        operation = 'plan' if 'images' in ctx else 'review' if 'findings' in ctx else 'worker'
        attempts[operation] += 1
        role = fresh.WORKER if operation == 'worker' else fresh.COORDINATOR
        reply = child_reply(role, ctx)
        text = fresh.packed(reply)
        if operation == 'plan' and attempts[operation] == 1:
            text = text.replace('"object_ids":[12]', '"object_ids":[33],"object_ids":[12]')
        if operation == 'worker':
            assert base64.b64decode(json['messages'][1]['content'][1]['image_url']['url']) == raw
            if attempts[operation] == 1:
                reply['findings'][0].update(object_ids=[33], status='observed')
                text = fresh.packed(reply)
        if attempts[operation] == 2:
            assert 'original' in json['messages'][-1]['content']
            assert 'Duplicate JSON field' in json['messages'][-1]['content'] if operation == 'plan' else (
                'without a label' in json['messages'][-1]['content'])
        value = {'model': fresh.MODEL, 'usage': {'prompt_tokens': 100, 'completion_tokens': 20},
                 'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}]}
        return SimpleNamespace(status_code=200, headers={}, json=lambda: value, text=fresh.packed(value))

    monkeypatch.setattr(direct.session, 'post', post)
    direct.visual.collect_new(state, messages)
    direct._save_visual_report('registry', 'images.json', direct.visual.catalog())
    assert attempts == {'plan': 2, 'worker': 2, 'review': 1}
    assert [row['status'] for row in direct.ledger] == ['failed', 'completed', 'failed', 'completed', 'completed']
    assert [row['index'] for row in direct.ledger if row['tag'].endswith('-format-repair')] == [2, 4]
    stats = Counter()
    audit_visual_batches(tmp_path, 3, stats)
    assert stats['visual_batches_verified'] == 1


@pytest.mark.parametrize('failure', ['recoverable', 'still_invalid', 'timeout'])
def test_bigmodel_parser_format_repair_is_bounded_to_known_response(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(fresh, 'MODEL', fresh.BIGMODEL_MODEL)
    direct = fresh.DirectCalls(tmp_path)
    schema = fresh.obj({'relation': {'enum': ['near', 'back']}})
    messages = [{'role': 'system', 'content': 'Parse relation.'}, {'role': 'user', 'content': 'Behind a chair.'}]
    requests = []

    def post(_url, *, json, **_kwargs):
        requests.append(json)
        if failure == 'timeout':
            raise fresh.requests.exceptions.ReadTimeout('HTTP outcome unknown')
        answer = {'relation': 'back' if failure == 'recoverable' and len(requests) == 2 else 'behind'}
        value = {'model': fresh.MODEL, 'usage': {'prompt_tokens': 100, 'completion_tokens': 20},
                 'choices': [{'message': {'content': fresh.packed(answer)}, 'finish_reason': 'stop'}]}
        return SimpleNamespace(status_code=200, headers={}, json=lambda: value, text=fresh.packed(value))

    monkeypatch.setattr(direct.session, 'post', post)
    if failure == 'recoverable':
        result = direct.parser_chat(messages, response_schema={'schema': schema})
        assert json.loads(result['_content']) == {'relation': 'back'}
        assert [row['role'] for row in direct.ledger] == ['parser', 'parser_repair']
        assert 'behind' in requests[1]['messages'][-1]['content']
    else:
        expected = fresh.requests.exceptions.ReadTimeout if failure == 'timeout' else fresh.ValidationError
        with pytest.raises(expected):
            direct.parser_chat(messages, response_schema={'schema': schema})
    assert len(requests) == (1 if failure == 'timeout' else 2)


def test_repair_diagnostic_does_not_pin_an_undelivered_measurement(tmp_path, monkeypatch):
    direct = fresh.DirectCalls(tmp_path, visual_reports=True)
    measurements = [{'evidence_id': f'observation_{i}', 'tool': 'measure'} for i in range(1, 4)]
    direct.context_state = {'observations': measurements}
    direct.visual.current_readback = 'observation_2'
    units = [{'id': 'initial', 'raw': {'tool': 'render'}, 'object_group': []}]
    units.extend({'id': row['evidence_id'], 'raw': row, 'object_group': [1]} for row in measurements)
    monkeypatch.setattr(direct.visual, 'collect_new', lambda *_: units)
    feedback = {'role': 'user', 'content': json.dumps({
        'repair': [[[], "Claim cites archived/unseen evidence: ['observation_1']"]]
    })}

    def prepare(messages, schema, role, reserve, state, *args, **kwargs):
        assert feedback in messages
        bound = fresh.OUTPUT_LIMIT if 'observation_1' in state['supporting_evidence_ids'] else 1000
        return SimpleNamespace(preflight={'maximum': bound}, packing={})

    monkeypatch.setattr(fresh, 'prepare_qwen_payload', prepare)
    prepared = direct._reported_payload([feedback], {}, fresh.OUTPUT_RESERVE)
    assert prepared.preflight['maximum'] < fresh.OUTPUT_LIMIT
    assert prepared.packing['archived_evidence_ids'] == ['initial', 'observation_1']
    assert prepared.packing['visible_native_evidence_ids'] == ['observation_2', 'observation_3']


def test_single_qwen_explicit_native_action_roundtrip_no_laya():
    h = fixtures.DagDeltaTests(); h.setUp()
    native_action = action() | {'monitor': h.monitor.encode(h.start)}
    _, calls = h.run_ground([native_action,
        action('finish', 1) | {'monitor': fixtures.delta(h.finished(), {'n1', 'n11'})}])
    request = calls[0]
    state = fresh.extract_state(request.args[0])
    answer = {'state_version': fresh.digest(state), 'action': native_action}
    import requests
    response = requests.Response(); response.status_code = 200
    response._content = json.dumps({'model': fresh.MODEL, 'choices': [{'finish_reason': 'stop',
        'message': {'content': json.dumps(answer)}}], 'usage': {'prompt_tokens': 1234, 'completion_tokens': 400}}).encode()
    with tempfile.TemporaryDirectory() as root:
        direct = fresh.DirectCalls(Path(root)); direct.monitor = h.new_monitor(); direct.key = 'unit-mock'
        direct.r_used = 8; direct.t_used = 20
        with patch.object(direct.session, 'post', return_value=response) as post:
            result = direct.choose(request.args[0], **request.kwargs)
        assert post.call_count == 1
        assert fresh.BASE_URL in post.call_args.args[0]
        assert json.loads(result['_content']) == native_action
        assert direct.r_used == 9 and direct.t_used == 21
        assert direct.ledger[0]['preflight']['enforced'] is True
        assert not list(Path(root).glob('*laya*'))
        wire = post.call_args.kwargs['json']
        assert 'options' not in wire['response_format']['json_schema']['schema']['properties']
        assert 'action' in wire['response_format']['json_schema']['schema']['properties']
        assert json.loads((Path(root)/'controller-contracts/t21/selected-model-action.json').read_text()) == native_action


def test_latest_native_feedback_visible_and_exact_repeat_schedules_tentative_without_budget_cap():
    h = fixtures.DagDeltaTests(); h.setUp()
    native_action = action() | {'monitor': h.monitor.encode(h.start)}
    _, calls = h.run_ground([native_action,
        action('finish', 1) | {'monitor': fixtures.delta(h.finished(), {'n1', 'n11'})}])
    request = calls[0]
    messages = fresh.deepcopy(request.args[0])
    messages[-1]['content'].append({'type': 'text', 'text': json.dumps({'observations': [
        {'tool': 'monitor_check', 'error': 'Original native rejection: active rivals remain', 'no_evidence_acquired': True}]})})
    state = fresh.extract_state(messages)
    answer = {'state_version': fresh.digest(state), 'action': native_action}
    import requests
    response = requests.Response(); response.status_code = 200
    response._content = json.dumps({'model': fresh.MODEL, 'choices': [{'finish_reason': 'stop',
        'message': {'content': json.dumps(answer)}}], 'usage': {'prompt_tokens': 1234, 'completion_tokens': 400}}).encode()
    with tempfile.TemporaryDirectory() as root:
        direct = fresh.DirectCalls(Path(root)); direct.monitor = h.new_monitor(); direct.key = 'unit-mock'
        with patch.object(direct.session, 'post', return_value=response) as post:
            direct.choose(messages, **request.kwargs)
            direct.choose(messages, **request.kwargs)
        assert post.call_count == 2
        payload = post.call_args.kwargs['json']
        assert 'Original native rejection: active rivals remain' in json.dumps(payload['messages'])
        assert direct.r_used == 3 and direct.t_used == 2
        assert (Path(root)/'controller-contracts/t02/no-progress-stop.json').exists()
        assert direct.pending_tentative_progress == direct.last_progress_identity
        with patch.object(direct, 'tentative_final', return_value={'tentative': True}) as final:
            assert direct.choose(messages, **request.kwargs) == {'tentative': True}
            assert final.call_args.args[2] == 'exact_repeated_native_rejection'


def test_progress_and_domain_use_accepted_state_not_candidate_count_alone():
    h = fixtures.DagDeltaTests(); h.setUp()
    with tempfile.TemporaryDirectory() as root:
        direct = fresh.DirectCalls(Path(root)); direct.monitor = h.new_monitor()
        state = {'allowed_target_ids': [1, 2, 3], 'active_target_ids': [1, 2], 'observations': []}
        assert not direct.observe_progress(state)
        assert not direct.observe_progress(state)
        assert not direct.observe_progress(state)
        assert direct.observe_progress(state)
        state['observations'].append({'tool': 'measure', 'evidence_id': 'observation_1', 'value': 5})
        assert not direct.observe_progress(state)
        assert direct.stagnant_transitions == 0
        direct.monitor.plan = {'nodes': [{'id': 'n0', 'status': 'completed', 'bound_object_ids': [2]}]}
        assert not direct.observe_progress(state)
        contract = fresh.candidate_contract(state, direct.monitor.plan, 2)
        assert contract['fixed_allowed_targets']['ids'] == [1, 2, 3]
        assert contract['current_active_targets']['ids'] == [1, 2]
        assert contract['legal_current_prediction_ids'] == [1, 2]
        assert 'do NOT' in contract['current_active_targets']['meaning']


def test_repeated_structural_repair_pattern_transitions_without_repeating_forever(tmp_path):
    harness = fixtures.DagDeltaTests()
    harness.setUp()
    native_action = action() | {'monitor': harness.monitor.encode(harness.start)}
    _, calls = harness.run_ground([native_action,
        action('finish', 1) | {'monitor': fixtures.delta(harness.finished(), {'n1', 'n11'})}])
    request = calls[0]
    direct = fresh.DirectCalls(tmp_path, visual_reports=True)
    direct.monitor = harness.new_monitor()

    def invalid_answer(tag, role, *_args, **_kwargs):
        index = len(direct.ledger) + 1
        answer = {'state_version': f'wrong-version-{index}', 'action': native_action}
        raw = json.dumps(answer)
        direct.ledger.append({'index': index, 'tag': tag, 'role': role, 'http_sent': True})
        fresh.save(tmp_path/'calls'/f'{index:03d}-{tag}-{role}'/'provider-response.json',
                   {'choices': [{'message': {'content': raw}}]})
        return {'_content': raw}

    with patch.object(direct, '_qwen', side_effect=invalid_answer) as call, \
         patch.object(direct, 'tentative_final', return_value={'tentative': True}) as final:
        assert direct.choose(request.args[0], **request.kwargs) == {'tentative': True}
    assert call.call_count == 3
    assert final.call_args.args[2] == 'repeated_structural_rejection'
    assert (tmp_path/'controller-contracts/t01/structural-stagnation.json').is_file()


def test_tentative_final_is_model_owned_once_and_monitor_never_completed():
    from types import SimpleNamespace
    with tempfile.TemporaryDirectory() as root:
        direct = fresh.DirectCalls(Path(root)); direct.key = 'unit-mock'; direct.t_used = 6
        native_plan = {'nodes': [{'id': 'n0', 'status': 'pending', 'bound_object_ids': [], 'excluded_target_ids': []}]}
        direct.monitor = SimpleNamespace(plan=native_plan, history=[], active_ids=lambda: [1, 2])
        state = {'description': 'the chair left of the desk', 'allowed_target_ids': [1, 2, 3],
                 'active_target_ids': [1, 2], 'objects': [{'id': 1}, {'id': 2}, {'id': 12}],
                 'observations': [{'tool': 'measure', 'evidence_id': 'observation_1'}]}
        answer = {'state_version': fresh.digest(state), 'predicted_id': 2, 'reason': 'Model chose target2',
                  'evidence': [{'source_quote': state['description'], 'object_ids': [2, 12],
                                'evidence_ids': ['observation_1'], 'observation': 'Raw measured left relation'}],
                  'main_rivals': [{'object_id': 1, 'evidence_ids': ['observation_1'], 'comparison': 'Model comparison'}],
                  'unresolved': ['Frame is assumed, category evidence remains uncertain']}
        with patch.object(direct, '_qwen', return_value={'_content': json.dumps(answer)}) as call:
            direct.ledger.append({'index': 1})
            try:
                direct.tentative_final([{'role': 'system', 'content': 'native'}], state, 'test')
                raise AssertionError('No tentative exception')
            except fresh.TentativeFinal as exc:
                assert exc.answer == answer
                assert exc.monitor_snapshot['plan'] == native_plan
            assert call.call_count == 1
            schema = call.call_args.args[3]
            assert schema['properties']['predicted_id']['enum'] == [1, 2, None]
            assert 'monitor' not in schema['properties']
        assert direct.monitor.plan == native_plan
        saved = json.loads((Path(root)/'tentative-final/frozen-model-prediction.json').read_text())
        assert saved['accepted'] is False and saved['strict_monitor_accepted'] is False
        try:
            direct.tentative_final([], state, 'again')
            raise AssertionError('Allowed repeated final evaluation')
        except RuntimeError:
            pass


def test_empty_category_pool_tentative_finish_limits_predictions_to_inspected_targets(tmp_path):
    direct = fresh.DirectCalls(tmp_path)
    direct.monitor = SimpleNamespace(plan=None, history=[], active_ids=lambda: [1, 2])
    direct.ledger.append({'index': 1})
    state = {'description': 'the red shoes', 'allowed_target_ids': [1, 2],
             'objects': [{'id': 1}, {'id': 2}], 'expanded_target_pool': True,
             'observations': [{'tool': 'render', 'evidence_id': 'observation_1',
                               'views': [{'marked_ids': [1]}]}]}
    answer = {'state_version': fresh.digest(state), 'predicted_id': None, 'reason': 'Color uncertain',
              'evidence': [], 'main_rivals': [], 'unresolved': ['Could not establish red color']}
    with patch.object(direct, '_qwen', return_value={'_content': json.dumps(answer)}) as call:
        with pytest.raises(fresh.TentativeFinal):
            direct.tentative_final([{'role': 'system', 'content': 'native'}], state, 'test')
    fields = call.call_args.args[3]['properties']
    assert fields['predicted_id']['enum'] == [1, None]
    assert fields['main_rivals']['items']['properties']['object_id']['enum'] == [1, 2]


def test_tentative_has_only_one_exact_structural_repair_and_preserves_originals():
    from types import SimpleNamespace
    import requests
    with tempfile.TemporaryDirectory() as root:
        direct = fresh.DirectCalls(Path(root)); direct.key = 'unit-mock'
        direct.monitor = SimpleNamespace(plan=None, history=[], active_ids=lambda: [1])
        state = {'description': 'the chair', 'allowed_target_ids': [1], 'active_target_ids': [1],
                 'objects': [{'id': 1}], 'observations': []}
        response = requests.Response(); response.status_code = 200
        response._content = json.dumps({'model': fresh.MODEL, 'choices': [{'finish_reason': 'stop',
            'message': {'content': '{}'}}], 'usage': {'prompt_tokens': 500, 'completion_tokens': 2}}).encode()
        with patch.object(direct.session, 'post', return_value=response) as post:
            try:
                direct.tentative_final([{'role': 'system', 'content': 'native'},
                                        {'role': 'user', 'content': json.dumps(state)}], state, 'fixture')
                raise AssertionError('Invalid terminal not rejected')
            except RuntimeError as exc:
                assert 'one model repair' in str(exc)
        assert post.call_count == 2 and direct.r_used == 4
        assert direct.monitor.plan is None and direct.monitor.history == []
        assert len(list(Path(root).glob('calls/*/provider-response.json'))) == 2
        assert not (Path(root)/'tentative-final/frozen-model-prediction.json').exists()


@pytest.mark.parametrize('duplicate_citation', [False, True])
def test_tentative_rival_repair_constrains_the_model_chosen_target(tmp_path, duplicate_citation):
    targets = list(range(1, 10))
    state = {'description': 'the chair', 'allowed_target_ids': targets, 'active_target_ids': targets,
             'objects': [{'id': oid} for oid in targets], 'observations': []}
    direct = fresh.DirectCalls(tmp_path)
    direct.monitor = SimpleNamespace(plan=None, history=[], active_ids=lambda: targets)
    answer = {'state_version': fresh.digest(state), 'predicted_id': 2, 'reason': 'Chair 2 matches.',
              'evidence': [{'source_quote': 'the chair', 'object_ids': [2], 'evidence_ids': ['initial'],
                            'observation': 'The overview shows chair 2.'}],
              'main_rivals': [{'object_id': 2, 'evidence_ids': ['initial'], 'comparison': 'Check rival.'}],
              'unresolved': []}
    if duplicate_citation:
        answer['evidence'][0]['evidence_ids'].append('initial')

    def respond(tag, role, messages, schema, **_kwargs):
        index = len(direct.ledger) + 1
        if index == 2:
            prepared = fresh.prepare_qwen_payload(messages, schema, role, 1024, state)
            wire = prepared.payload['response_format']['json_schema']['schema']
            with pytest.raises(fresh.ValidationError):
                fresh.validate(answer, wire)
            answer['main_rivals'][0]['object_id'] = 1
            answer['evidence'][0]['evidence_ids'] = ['initial']
            fresh.validate(answer, wire)
        raw = json.dumps(answer)
        direct.ledger.append({'index': index, 'tag': tag, 'role': role})
        fresh.save(tmp_path/'calls'/f'{index:03d}-{tag}-{role}'/'provider-response.json',
                   {'choices': [{'message': {'content': raw}}]})
        return {'_content': raw}

    messages = [{'role': 'system', 'content': 'native'}, {'role': 'user', 'content': json.dumps(state)}]
    with patch.object(direct, '_qwen', side_effect=respond) as calls:
        with pytest.raises(fresh.TentativeFinal) as done:
            direct.tentative_final(messages, state, 'fixture')
    assert calls.call_count == 2 and done.value.answer['predicted_id'] == 2
    assert done.value.answer['main_rivals'][0]['object_id'] == 1
    errors = json.loads((tmp_path/'tentative-final/structural-rejection-0.json').read_text())['errors']
    assert any('Rival list contains predicted target' in row['error'] for row in errors)
    if duplicate_citation:
        assert any(row.get('validator') == 'uniqueItems' for row in errors)


def view_row(azimuth=315, elevation=10, focus=(1, 2), context='isolated', pixels=None):
    return {'tool': 'render', 'focus_ids': list(focus), 'evidence_id': 'observation_1',
            'monitor_node_id': 'n0', 'views': [{'focus_ids': list(focus), 'context': context,
            'azimuth_deg': azimuth, 'elevation_deg': elevation,
            'visible_instance_pixels': pixels or {'1': 1000, '2': 2000}, 'visible_pixels': 3000,
            'points': 4196, 'marked_ids': list(focus)}]}


def test_historical_coverage_ignores_small_angles_pixels_order_and_ab_replay(tmp_path):
    direct = fresh.DirectCalls(tmp_path); direct.monitor = SimpleNamespace(plan=None)
    state = {'active_target_ids': [1, 2], 'observations': []}
    for i, angle in enumerate([359, 0, 1, 2]):
        row = view_row(angle, focus=(2, 1) if i % 2 else (1, 2),
                       pixels={'1': 1000 + i * 10, '2': 2000 - i})
        row.update(evidence_id=f'observation_{i}', monitor_node_id=f'n{i}')
        row['views'][0]['points'] += i
        state['observations'].append(row)
        assert direct.observe_progress(state) is (i == 3)
    # Crossing a bin is finite historical coverage, not a perpetual progress reset.
    for angle, expected in [(8, 0), (7, 1), (8, 2), (7, 3), (8, 4)]:
        state['observations'].append(view_row(angle))
        direct.observe_progress(state)
        assert direct.stagnant_transitions == expected
    # New views, contexts, comparison groups and newly visible IDs reset progress.
    for row in [view_row(90), view_row(90, context='scene'), view_row(90, focus=(1, 3)),
                view_row(90, focus=(1, 3), pixels={'1': 1000, '2': 2000, '3': 500})]:
        state['observations'].append(row)
        assert not direct.observe_progress(state)
        assert direct.stagnant_transitions == 0
    # Replaying two already seen groups in either object order cannot keep it alive.
    for focus in [(1, 2), (3, 1), (2, 1), (1, 3)]:
        state['observations'].append(view_row(90, focus=focus))
        direct.observe_progress(state)
    assert direct.stagnant_transitions == 4


def test_measurement_order_receipts_and_frames_use_historical_raw_coverage(tmp_path):
    direct = fresh.DirectCalls(tmp_path); direct.monitor = SimpleNamespace(plan=None)
    state = {'active_target_ids': [1, 2], 'observations': []}
    measurement = {'tool': 'measure', 'focus_ids': [1, 2], 'evidence_id': 'observation_1',
        'objects': [{'id': 1, 'center_xyz_m': [0, 1, 2]}, {'id': 2, 'center_xyz_m': [2, 1, 0]}],
        'directed_pair_measurements': [{'object_id': 1, 'reference_id': 2, 'xy_center_m': 2}],
        'selected_observer_frame': None}
    state['observations'].append(measurement)
    assert not direct.observe_progress(state)
    repeat = fresh.deepcopy(measurement)
    repeat.update(evidence_id='observation_2', monitor_node_id='n2', focus_ids=[2, 1])
    repeat['objects'].reverse()
    state['observations'].append(repeat)
    direct.observe_progress(state)
    assert direct.stagnant_transitions == 1
    changed = fresh.deepcopy(measurement)
    changed['selected_observer_frame'] = {'origin_ids': [], 'facing_ids': [1, 2],
                                          'rows_left_to_right': [[1, -1], [2, 1]]}
    state['observations'].append(changed)
    assert not direct.observe_progress(state)
    state['observations'].append(measurement)
    direct.observe_progress(state)
    assert direct.stagnant_transitions == 1
    changed = fresh.deepcopy(measurement)
    changed['directed_pair_measurements'].append({'object_id': 1, 'reference_id': 3, 'xy_center_m': 3})
    state['observations'].append(changed)
    assert not direct.observe_progress(state)
    direct.monitor.plan = {'nodes': [{'id': 'n0', 'status': 'completed', 'bound_object_ids': [1]}]}
    assert not direct.observe_progress(state)


def test_q00171_actual_acquisitions_trigger_without_gt_or_model_calls(tmp_path):
    fixture = json.loads((Path(__file__).parents[1]/'fixtures/qwen_direct_q00171_progress.json').read_text())
    direct = fresh.DirectCalls(tmp_path); direct.monitor = SimpleNamespace(plan=None)
    state = {'active_target_ids': fixture['active_target_ids'], 'observations': []}
    triggers = []
    for record in fixture['decisions']:
        direct.t_used = record['T']
        if 'accepted_plan' in record:
            direct.monitor.plan = record['accepted_plan']
        if record.get('last_observation'):
            state['observations'].append(record['last_observation'])
        if direct.observe_progress(state):
            triggers.append(record['T'])
    assert triggers[0] == 26
    assert direct.ledger == []
    assert len(fixture['decisions']) == 66


def test_glm_repeat_rejected_before_tool_runs(tmp_path):
    direct = fresh.DirectCalls(tmp_path, visual_reports=True)
    acquired = {'action': 'inspect', 'focus_ids': [1, 2], 'context': 'isolated',
                'azimuth_deg': 315, 'elevation_deg': 30}
    proposal = {**acquired, 'focus_ids': [2, 1], 'azimuth_deg': None, 'elevation_deg': None}
    state = {'unavailable_evidence_actions': [acquired]}
    with pytest.raises(ValueError, match='tool request already ran'):
        direct.check_glm_evidence_action(proposal, state, {})
    direct.check_glm_evidence_action({**proposal, 'context': 'scene'}, state, {})


def test_glm_abstention_gets_one_followup_when_evidence_tools_are_available(tmp_path):
    direct = fresh.DirectCalls(tmp_path, visual_reports=True)
    abstain = {'action': 'finish', 'predicted_id': None}
    state = {'active_target_ids': [1, 2]}
    tools = {'anyOf': [{'properties': {'action': {'enum': ['measure']}}}]}
    direct.check_glm_evidence_action(abstain, state, {})
    direct.check_glm_evidence_action(abstain, {'active_target_ids': [1]}, tools)
    with pytest.raises(ValueError, match='one focused evidence attempt'):
        direct.check_glm_evidence_action(abstain, state, tools)
    direct.check_glm_evidence_action(abstain, state, tools)


@pytest.mark.parametrize('final_binding', ([], [1]))
def test_glm_repairs_null_finish_only_after_verified_unique_target(final_binding):
    h = fixtures.DagDeltaTests()
    h.setUp()
    plan = h.finished()
    plan['nodes'][-1]['bound_object_ids'] = final_binding
    proposal = action('finish', None) | {
        'evidence': [{'object_ids': [1]}],
        'unresolved': ['The task is complete.'],
        'monitor': fixtures.delta(plan, {'n1', 'n11'}),
    }
    state = {'observations': h.observations()}
    before = fresh.deepcopy(h.monitor.plan)
    with pytest.raises(ValueError, match='only object 1, but predicted_id is null'):
        fresh.check_glm_abstention(proposal, h.monitor, state)
    assert h.monitor.plan == before

    plan['nodes'][1]['excluded_target_ids'] = []
    proposal['monitor'] = fixtures.delta(plan, {'n1', 'n11'})
    fresh.check_glm_abstention(proposal, h.monitor, state)


def test_explicit_model_replanning_adds_measure_stage_without_program_completion(tmp_path):
    from jsonschema import validate, ValidationError
    h = fixtures.DagDeltaTests(); h.setUp()
    initial = fresh.deepcopy(h.start)
    initial['nodes'] = [initial['nodes'][0], initial['nodes'][-1]]
    initial['nodes'][0].update(status='pending', bound_object_ids=[], evidence_ids=[], conclusion='', allowed_actions=['inspect'])
    initial['nodes'][-1]['depends_on'] = ['n0']
    initial['current_node_id'] = 'n0'
    inspect = action('inspect') | {'context': 'isolated', 'monitor': h.monitor.encode(initial)}
    proposed = fresh.deepcopy(h.start)
    proposed['nodes'][0].update(evidence_ids=['observation_1'], conclusion='The returned view supports these bindings')
    measure = action() | {'monitor': {'replace_plan': h.monitor.encode(proposed)}}
    _, requests = h.run_ground([inspect, measure,
        action('finish', 1) | {'monitor': fixtures.delta(h.finished(), {'n1', 'n11'})}])
    request = requests[1]
    direct = fresh.DirectCalls(tmp_path); direct.monitor = h.new_monitor()
    direct.monitor.update(initial, inspect, [])
    state = fresh.extract_state(request.args[0])
    answer = {'state_version': fresh.digest(state), 'action': measure}
    with patch.object(direct, '_qwen', return_value={'_content': json.dumps(answer)}) as call:
        result = direct.choose(request.args[0], **request.kwargs)
    # The program serialized exactly the model's explicit replacement.
    assert json.loads(result['_content']) == measure
    assert direct.monitor.plan == initial
    visible_schema = call.call_args.args[3]
    provider_wire = fresh.provider_schema(fresh.factor_schema(visible_schema))
    fresh.Draft202012Validator.check_schema(provider_wire)
    validate(answer, provider_wire)
    monitor_schema = visible_schema['properties']['action']['anyOf'][0]['properties']['monitor']
    validate(measure['monitor'], monitor_schema)
    incomplete = fresh.deepcopy(measure['monitor'])
    del incomplete['replace_plan']['verification']
    try:
        validate(incomplete, monitor_schema)
        raise AssertionError('Incomplete replacement was accepted')
    except ValidationError:
        pass
    visible_contract = json.loads(call.call_args.args[2][-1]['content'])['candidate_contract']['monitor_progression']
    assert 'NEXT tool' in visible_contract['rule'] and 'Delta node_updates cannot change' in visible_contract['replanning']
    direct.monitor.update(direct.monitor.decode(measure['monitor']), measure,
                          [{'tool': 'render', 'focus_ids': [1, 2, 3], 'evidence_id': 'observation_1'}])
    assert direct.monitor.plan['current_node_id'] == 'n1'
    assert direct.monitor.plan['nodes'][0]['status'] == 'completed'
    assert direct.monitor.plan['nodes'][1]['status'] == 'pending'
