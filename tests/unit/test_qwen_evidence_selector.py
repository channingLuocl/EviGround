"""ID-only selectors: original pixels, native quote domains and receipt boundaries."""
import base64
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import requests
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
import qwen_direct_fresh as fresh
import qwen_step2_context as context
import qwen_evidence_selector as selector
from eviground.grounding.dag_monitor import DagMonitor, exact_source_quotes, usable_evidence_observation


def image(index):
    stream = io.BytesIO(); Image.new('RGB', (1800, 1200), (index * 11 % 256, 95, 155)).save(stream, format='PNG')
    return {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()}}


@pytest.fixture(scope='module')
def synthetic_history():
    query = "'the lamp near the desk'"
    def node(name, kind, deps, actions):
        return {'id': name, 'goal': 'check original conditions', 'kind': kind, 'source_quotes': [query],
                'depends_on': deps, 'allowed_actions': actions, 'status': 'pending', 'bound_object_ids': [],
                'evidence_ids': [], 'conclusion': '', 'excluded_target_ids': []}
    observations, views = [], [{'kind': 'target_preview', 'marked_ids': [1, 2]}, {'kind': 'scene_preview', 'marked_ids': [1, 2, 9]}]
    for i in range(8):
        rendered = [{'focus_ids': [1, 2, 9], 'context': 'scene', 'azimuth_deg': i * 20 + j, 'elevation_deg': 30,
                     'marked_ids': [1, 2, 9]} for j in range(2)]
        observations.append({'tool': 'render', 'focus_ids': [1, 2, 9], 'views': deepcopy(rendered),
                             'evidence_id': f'observation_{i+1}', 'monitor_node_id': 'n0'})
        views.extend(deepcopy(rendered))
    # The last view deliberately aliases an earlier image: main dedup must retain source mapping.
    images = [image(i) for i in range(len(views))]; images[-1] = deepcopy(images[2])
    state = {'description': query, 'allowed_target_ids': [1, 2], 'active_target_ids': [1, 2],
        'objects': [{'id': i, 'category': category, 'category_hypotheses': [{'category': category, 'rank': 1, 'gap_from_top1': 0}],
                     'center_x_m': i * .123456789123, 'center_y_m': -i, 'height_m': 1}
                    for i, category in [(1, 'lamp'), (2, 'lamp'), (9, 'desk')]],
        'parsed_reference_categories': ['desk'], 'reference_category_bindings': [{'category': 'desk', 'candidates': [{'id': 9}]}],
        'monitor_requirements': [{'id': 'q0', 'source_quote': query}], 'previous_actions': [],
        'observations': observations, 'views': views,
        'task_dag': {'nodes': [node('n0', 'binding', [], ['inspect', 'measure'])],
            'verification': node('n11', 'verification', ['n0'], ['finish']),
            'requirement_nodes': {'q0': ['n11']}, 'current_node_id': 'n0', 'reopen_node_ids': [], 'revision_reason': 'start'},
        'supporting_evidence_ids': ['initial'] + [o['evidence_id'] for o in observations],
        'supporting_evidence_objects': {e: [1, 2, 9] for e in ['initial'] + [o['evidence_id'] for o in observations]}}
    messages = [{'role': 'system', 'content': 'Return original native JSON; no semantic program repair.'},
                {'role': 'user', 'content': [{'type': 'text', 'text': json.dumps(state)}, *images]}]
    return state, messages


def mock_response(content, prompt=4000):
    response = requests.Response(); response.status_code = 200
    response._content = json.dumps({'model': fresh.MODEL, 'choices': [{'finish_reason': 'stop',
        'message': {'content': json.dumps(content)}}], 'usage': {'prompt_tokens': prompt, 'completion_tokens': 20}}).encode()
    return response


def selector_reply(payload):
    data = context.decode_tables(json.loads(payload['messages'][1]['content'][0]['text']))
    return {'batch_id': data['batch_id'], 'selected_evidence_ids': data['required_input_dependencies']}


def test_exact_quotes_real_failures_rejected_without_mutation_and_real_legal_monitor_passes():
    fixture = json.loads((Path(__file__).parents[1]/'fixtures/qwen_step2_exact_quotes.json').read_text())
    desc, requirements = fixture['description'], fixture['monitor_requirements']
    quotes = exact_source_quotes(desc, requirements)
    assert quotes == ["'The lamp not next to the beds.'", 'not', 'not next to the beds']
    monitor = DagMonitor(desc, {}, fixture['known_object_ids'], fixture['allowed_target_ids'], ['inspect', 'measure', 'finish'])
    monitor.requirements = requirements
    native = monitor.schema(compact=True)
    state = {'description': desc, 'monitor_requirements': requirements, 'task_dag': None}
    wire = context.compact_wire_schema({'properties': {'state_version': {'const': 'test'}, 'action': {}}}, state)
    action_schema = wire['properties']['action']
    original = deepcopy(fixture['bad_real_actions'])
    for row in original:
        with pytest.raises(fresh.ValidationError): fresh.validate(row['action'], action_schema)
        with pytest.raises(fresh.ValidationError): fresh.validate(row['action']['monitor'], native)
    good = fixture['positive_real_action']['action']
    fresh.validate(good, action_schema); fresh.validate(good['monitor'], native)
    monitor.update(monitor.decode(good['monitor']), good, [])
    assert monitor.plan is not None and fixture['bad_real_actions'] == original
    replacement = deepcopy(good); replacement['monitor'] = {'replace_plan': good['monitor']}
    later = context.compact_wire_schema({'properties': {'state_version': {'const': 'test'}, 'action': {}}},
                                        {**state, 'task_dag': {'nodes': []}})
    fresh.validate(replacement, later['properties']['action'])
    bad = deepcopy(replacement); bad['monitor']['replace_plan']['verification']['source_quotes'] = ['The lamp not next to the beds.']
    with pytest.raises(fresh.ValidationError): fresh.validate(bad, later['properties']['action'])


def test_native_feedback_predicate_and_view_hash_position_mapping(synthetic_history):
    state, messages = deepcopy(synthetic_history)
    state['observations'].extend([{'tool': 'finish_check', 'missing': 'Need appearance'},
        {'tool': 'monitor_check', 'error': 'rejected'}, {'tool': 'measure', 'evidence_id': 'observation_99', 'error': 'failed'}])
    full, units = selector.registry(state, messages)
    assert {row['id'] for row in units} == DagMonitor.evidence_ids(state['observations'])
    assert units[0]['source_view_indices'] == [0, 1]
    assert units[1]['source_view_indices'] == [2, 3]
    assert units[-1]['source_view_indices'] == [16, 17]
    images = selector.image_parts(messages)
    for row in units:
        for index, sha in zip(row['source_view_indices'], row['image_sha256']):
            assert selector.digest({'not_the_image': index}) != sha
            raw = base64.b64decode(images[index]['image_url']['url'].split(',', 1)[1])
            assert __import__('hashlib').sha256(raw).hexdigest() == sha
    projected, projected_messages = selector.project(full, messages, units, {'initial', 'observation_1'})
    assert projected['observations'][-3:] == state['observations'][-3:]
    assert projected['task_dag'] == state['task_dag'] and projected['active_target_ids'] == [1, 2]
    assert len(selector.image_parts(projected_messages)) == 4
    assert all(not usable_evidence_observation(row) for row in projected['observations'][-3:])


def test_original_multibatch_selector_preview_no_resize_and_no_native_state_changes(tmp_path, synthetic_history):
    state, messages = deepcopy(synthetic_history); before = deepcopy(state)
    direct = fresh.DirectCalls(tmp_path); direct.context_state = state; direct.preview_only = True
    direct.selector_preview = lambda meta: [x for x in meta['allowed_ids'] if x in meta['mandatory_ids']]
    with patch.object(context, 'resize_overflow_images', side_effect=AssertionError('No resize in production')), pytest.raises(fresh.PreparedCall) as raised:
        direct._qwen('t01-action-0', 'semantic_consultant', messages, {'type': 'object'})
    call = raised.value; audit = call.packing['evidence_selection']
    assert len(audit['batches']) >= 2 and call.preflight['maximum'] < 32768
    assert all(b['preflight']['maximum'] < 32768 for b in audit['batches'])
    assert audit['selected_ids'] == ['initial'] and len(audit['omitted_ids']) == 8
    assert call.packing['all_active_target_ids'] == [1, 2] and state == before
    assert direct.ledger == [] and call.packing['image_derivations'] == []
    assert all(b['provenance']['simulated_selection'] for b in audit['batches'])
    assert call.preflight['images'][0]['width'] == 1800


def test_selector_union_overflow_not_silently_trimmed(tmp_path, synthetic_history):
    state, messages = deepcopy(synthetic_history)
    direct = fresh.DirectCalls(tmp_path); direct.context_state = state; direct.preview_only = True
    direct.selector_preview = lambda meta: meta['allowed_ids']
    with pytest.raises(selector.EvidenceSelectionError, match='union cannot fit') as failure:
        direct._qwen('t01-action-0', 'semantic_consultant', messages, {'type': 'object'})
    assert len(failure.value.audit['selected_ids']) == 9
    assert failure.value.audit['selected_main_bound'] >= 32768
    assert not any(row.get('http_sent') for row in direct.ledger)


@pytest.mark.parametrize('answer', [{'action': 'finish'}, {'batch_id': 'wrong', 'selected_evidence_ids': ['initial']}])
def test_selector_illegal_shape_or_registry_rejected_without_default(tmp_path, synthetic_history, answer):
    state, messages = deepcopy(synthetic_history)
    direct = fresh.DirectCalls(tmp_path); direct.context_state = state; direct.key = 'unit-mock'
    with patch.object(direct.session, 'post', return_value=mock_response(answer)) as post, pytest.raises(selector.EvidenceSelectionError):
        direct._qwen('t01-action-0', 'semantic_consultant', messages, {'type': 'object'})
    assert post.call_count == 1 and direct.ledger[0]['role'] == 'evidence_selector'
    assert direct.ledger[0]['http_sent'] and not direct.ledger[-1]['http_sent']


def test_actual_dispatch_mock_counts_selector_costs_and_blocks_unseen_main_citations(tmp_path, synthetic_history):
    state, messages = deepcopy(synthetic_history)
    direct = fresh.DirectCalls(tmp_path); direct.context_state = state; direct.key = 'unit-mock'
    roles = []
    def post(*args, **kwargs):
        payload = kwargs['json']; role = payload['response_format']['json_schema']['name']; roles.append(role)
        if role == 'evidence_selector_answer': return mock_response(selector_reply(payload))
        return mock_response({'evidence': [{'evidence_ids': ['observation_8']}]})
    with patch.object(direct.session, 'post', side_effect=post), pytest.raises(ValueError, match='not actually delivered'):
        direct._qwen('t01-action-0', 'semantic_consultant', messages, {'type': 'object'})
    assert roles[-1] == 'semantic_consultant_answer' and roles.count('evidence_selector_answer') >= 2
    assert direct.r_used == len(roles) and len(direct.ledger) == len(roles)
    assert state == synthetic_history[0]
    assert all(row.get('usage') == {'prompt_tokens': 4000, 'completion_tokens': 20} for row in direct.ledger)


def test_selector_provider_capacity_violation_keeps_type_and_never_sends_main(tmp_path, synthetic_history):
    state, messages = deepcopy(synthetic_history)
    direct = fresh.DirectCalls(tmp_path); direct.context_state = state; direct.key = 'unit-mock'
    def post(*args, **kwargs): return mock_response(selector_reply(kwargs['json']), prompt=32760)
    with patch.object(direct.session, 'post', side_effect=post) as send, pytest.raises(fresh.ProviderCapacityViolation):
        direct._qwen('t01-action-0', 'semantic_consultant', messages, {'type': 'object'})
    assert send.call_count == 1 and direct.ledger[0]['error_type'] == 'ProviderCapacityViolation'
    assert direct.ledger[0]['role'] == 'evidence_selector' and not direct.ledger[-1]['http_sent']


def test_visible_reference_guard_covers_delta_replacement_and_tentative_rivals():
    with pytest.raises(ValueError): selector.validate_visible_references({'action': {'monitor': {'node_updates': [
        {'id': 'n0', 'evidence_ids': ['observation_7']}]}}}, ['initial'])
    with pytest.raises(ValueError): selector.validate_visible_references({'action': {'monitor': {'replace_plan': {
        'nodes': [{'evidence_ids': ['observation_7']}]}}}}, ['initial'])
    with pytest.raises(ValueError): selector.validate_visible_references({'main_rivals': [{'evidence_ids': ['observation_7']}]}, ['initial'])
    selector.validate_visible_references({'evidence': [{'evidence_ids': ['initial']}]}, ['initial'])


def test_accepted_node_evidence_is_required_even_if_selector_omits_it(tmp_path, synthetic_history):
    state, messages = deepcopy(synthetic_history)
    state['task_dag']['nodes'][0].update(status='completed', evidence_ids=['observation_1'], bound_object_ids=[1], conclusion='Prior binding')
    messages[1]['content'][0]['text'] = json.dumps(state)
    direct = fresh.DirectCalls(tmp_path); direct.context_state = state; direct.preview_only = True
    direct.selector_preview = lambda meta: []
    with pytest.raises(fresh.PreparedCall) as raised:
        direct._qwen('t03-action-1', 'semantic_consultant', messages, {'type': 'object'})
    audit = raised.value.packing['evidence_selection']
    assert 'observation_1' in audit['mandatory_ids'] and 'observation_1' in audit['selected_ids']
    assert state['task_dag']['nodes'][0]['status'] == 'completed'
