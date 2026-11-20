"""One-model integration checks against original native controller contracts."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import qwen_direct_fresh as fresh
from tests.unit import test_dag_delta as fixtures
from tests.unit.test_dag_monitor import action


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
        assert direct.ledger[0]['preflight']['enforced'] is False
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
        assert 'Original native rejection: active rivals remain' in payload['messages'][-1]['content']
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
