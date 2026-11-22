"""Local hard boundaries and exact evidence packing; no real model requests."""
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
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import qwen_direct_fresh as fresh
import qwen_step2_context as context


def pixels():
    data = io.BytesIO(); Image.new('RGB', (512, 512)).save(data, format='PNG')
    return {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(data.getvalue()).decode()}}


def response(prompt=100, completion=1):
    result = requests.Response(); result.status_code = 200
    result._content = json.dumps({'model': fresh.MODEL, 'choices': [{'finish_reason': 'stop', 'message': {'content': '{"ok":true}'}}],
                                 'usage': {'prompt_tokens': prompt, 'completion_tokens': completion}}).encode()
    return result


@pytest.mark.parametrize('role', ['parser', 'parser_repair', 'semantic_consultant', 'image_selector', 'new_subcall'])
@pytest.mark.parametrize('maximum', [32767, 32768, 32769])
def test_every_call_role_strict_boundary_before_http(tmp_path, role, maximum):
    direct = fresh.DirectCalls(tmp_path)
    preflight = fresh.qwen_preflight
    def controlled(*args):
        result = preflight(*args); result['maximum'] = maximum
        return result
    with patch.object(fresh, 'qwen_preflight', side_effect=controlled), patch.object(direct.session, 'post', return_value=response()) as post:
        if maximum >= 32768:
            with pytest.raises(RuntimeError, match='no request sent'):
                direct._qwen('test', role, [{'role': 'system', 'content': 'Return JSON'}], {'type': 'object'})
            assert post.call_count == 0
            assert direct.ledger[-1]['status'] == 'rejected_preflight'
            assert direct.ledger[-1]['http_sent'] is False
        else:
            direct._qwen('test', role, [{'role': 'system', 'content': 'Return JSON'}], {'type': 'object'})
            assert post.call_count == 1
        assert direct.record_only


def test_response_reserved_actual_and_count_bounds_fail_without_retry():
    bound = {'maximum': 32767}
    with pytest.raises(RuntimeError, match='reserved context'):
        fresh.read_response(response(28672), fresh.MODEL, bound, 4096)
    with pytest.raises(RuntimeError, match='completion_tokens'):
        fresh.read_response(response(100, 4097), fresh.MODEL, bound, 4096)
    with pytest.raises(RuntimeError, match='input upper bound'):
        fresh.read_response(response(500), fresh.MODEL, {'maximum': 4200}, 4096)
    assert fresh.read_response(response(100, 100), fresh.MODEL, bound, 4096)['usage']['prompt_tokens'] == 100


@pytest.fixture
def glm_backend(monkeypatch):
    for name in ('MODEL', 'BASE_URL', 'OUTPUT_LIMIT'):
        monkeypatch.setattr(fresh, name, getattr(fresh, name))
    fresh.configure_glm('http://127.0.0.1:18045/v1')


def test_glm_text_preflight_blocks_completion_at_context_limit(tmp_path, glm_backend):
    direct = fresh.DirectCalls(tmp_path, visual_reports=True)
    tokenizer = requests.Response()
    tokenizer.status_code = 200
    tokenizer._content = json.dumps({'count': 65536 - 768 - 10, 'max_model_len': 65536}).encode()
    with patch.object(direct.session, 'post', return_value=tokenizer) as post:
        with pytest.raises(fresh.ContextCapacityExceeded, match='GLM tokenizer count'):
            direct._qwen('test', 'parser', [{'role': 'user', 'content': 'return JSON'}], {'type': 'object'}, reserve=768)
    post.assert_called_once()
    assert post.call_args.args[0].endswith('/tokenize')
    assert not direct.ledger[-1]['http_sent']


def test_glm_visual_worker_preserves_image_without_tokenizer_call(tmp_path, glm_backend):
    direct = fresh.DirectCalls(tmp_path, visual_reports=True)
    part = pixels()
    with patch.object(direct.session, 'post', return_value=response()) as post:
        direct._qwen('worker', fresh.WORKER, [{'role': 'user', 'content': [part]}], {'type': 'object'})
    post.assert_called_once()
    assert post.call_args.args[0].endswith('/v1/chat/completions')
    wire = post.call_args.kwargs['json']
    assert wire['model'] == 'GLM-4.5V'
    assert wire['messages'][0]['content'] == [part]
    assert wire['chat_template_kwargs'] == {'enable_thinking': False}
    assert 'vl_high_resolution_images' not in wire


def test_compact_inspect_keeps_native_angle_bounds():
    elevation = {'anyOf': [{'type': 'number', 'minimum': 10, 'maximum': 75}, {'type': 'null'}]}
    native = {'properties': {'state_version': {'const': 'v'}, 'action': {'anyOf': [
        {'properties': {'action': {'enum': ['inspect']}, 'elevation_deg': elevation}}
    ]}}}
    wire = context.compact_wire_schema(native, {'description': 'A lamp'}, text_visual_reports=True)
    inspect = wire['properties']['action']['anyOf'][0]['properties']
    assert inspect['elevation_deg'] == elevation
    with pytest.raises(fresh.ValidationError):
        fresh.validate(0, inspect['elevation_deg'])


def test_glm_tentative_schema_requires_evidence_and_legal_rivals_or_explained_abstention(glm_backend):
    candidates = [12, 33, 34, 35, 36, 37]
    fields = {'predicted_id': {'enum': candidates + [None]},
              'evidence': {'type': 'array'},
              'main_rivals': {'type': 'array', 'items': {'type': 'object', 'properties': {
                  'object_id': {'enum': candidates}}}},
              'unresolved': {'type': 'array'}}
    schema = {'type': 'object', 'properties': fields, 'required': list(fields)}
    prepared = fresh.prepare_qwen_payload([], schema, 'semantic_consultant', 1024)
    wire = prepared.payload['response_format']['json_schema']['schema']
    answer = {'predicted_id': 12, 'evidence': [], 'main_rivals': [], 'unresolved': ['missing geometry']}
    with pytest.raises(fresh.ValidationError):
        fresh.validate(answer, wire)
    fresh.validate({**answer, 'predicted_id': None}, wire)
    fresh.validate({**answer, 'evidence': ['image'], 'main_rivals': [{'object_id': 33}]}, wire)
    with pytest.raises(fresh.ValidationError):
        fresh.validate({**answer, 'evidence': ['image'], 'main_rivals': [{'object_id': 7}]}, wire)


def test_glm_planner_wire_keeps_both_native_finish_contracts():
    native = {'properties': {'state_version': {'const': 'v'}, 'action': {'anyOf': [
        {'properties': {'action': {'enum': ['finish']}, 'predicted_id': {'type': 'null'},
                        'unresolved': {'type': 'array', 'minItems': 1}, 'evidence': {'type': 'array'}}},
        {'properties': {'action': {'enum': ['finish']}, 'predicted_id': {'enum': [11, 12]},
                        'unresolved': {'type': 'array', 'maxItems': 0},
                        'evidence': {'type': 'array', 'minItems': 1}}}
    ]}}}
    wire = context.compact_wire_schema(native, {'description': 'A lamp'}, text_visual_reports=True,
                                       native_finish=True)
    finishes = [branch['properties'] for branch in wire['properties']['action']['anyOf']
                if branch['properties']['action']['const'] == 'finish']
    assert [f['predicted_id'] for f in finishes] == [{'type': 'null'}, {'enum': [11, 12]}]
    with pytest.raises(fresh.ValidationError):
        fresh.validate([], finishes[0]['unresolved'])
    with pytest.raises(fresh.ValidationError):
        fresh.validate([], finishes[1]['evidence'])


def test_glm_tentative_rejects_visual_evidence_assigned_to_wrong_object():
    scopes = {'observation_1': [0], 'observation_2': [7]}
    answer = {'evidence': [{'evidence_ids': ['observation_1'], 'object_ids': [7]}]}
    with pytest.raises(ValueError, match='does not cover object IDs'):
        fresh.validate_tentative_coverage(answer, scopes)
    fresh.validate_tentative_coverage(
        {'evidence': [{'evidence_ids': ['observation_1', 'observation_2'], 'object_ids': [0, 7]}]}, scopes)


def measurement():
    return {'tool': 'measure', 'evidence_id': 'observation_1', 'objects': [{'id': i, 'center_xyz_m': [i, -i, 1.234567890123]} for i in range(8)],
        'directed_pair_measurements': [{'object_id': i, 'reference_id': j, 'xy_center_m': abs(i-j) + .1234567890123}
                                       for i in range(8) for j in range(8) if i != j],
        'selected_observer_frame': {'origin_ids': [], 'facing_ids': [1, 2], 'rows_left_to_right': [[2, -.5], [1, 1.5]]},
        'conditional_observer_frames': [{'reference_id': 7, 'rows_left_to_right': [[i, -i*.123456789] for i in range(8)]}],
        'monitor_node_id': 'n1'}


def test_pair_frame_group_roundtrip_no_rounding_or_dangling_refs():
    first = measurement(); second = deepcopy(first)
    second.update(evidence_id='observation_2', monitor_node_id='n2')
    second['selected_observer_frame']['rows_left_to_right'].reverse()
    state = {'observations': [first, second]}
    before = deepcopy(state)
    encoded = context.encode_tables(context.pack_measurements(state))
    decoded = context.decode_tables(encoded)
    assert len(decoded['measurement_pairs']) == 56
    assert len(decoded['observations'][0]['pair_refs']) == 56
    assert decoded['observations'][0]['comparison_group'] == list(range(8))
    assert context.unpack_measurements(decoded) == state == before


def test_full_id_catalog_references_top5_and_explicit_archive_preserved():
    state = {'description': 'lamp beside the beds', 'active_target_ids': [1], 'objects': [
        {'id': i, 'category': name, 'category_hypotheses': [{'category': name, 'rank': 1, 'gap_from_top1': 0}],
         'center_x_m': .123456789123} for i, name in [(1, 'lamp'), (2, 'bed'), (3, 'cabinet'), (4, 'door')]],
        'reference_category_bindings': [{'category': 'cabinet', 'candidates': [{'id': 3}]}],
        'previous_actions': [], 'observations': []}
    before = deepcopy(state); visible, detailed, archived = context.compact_catalog(state)
    assert [o['id'] for o in visible['objects']] == [1, 2, 3, 4]
    assert detailed == [1, 2, 3]
    assert archived == {'4': ['center_x_m']}
    assert visible['objects'][2]['center_x_m'] == .123456789123
    assert 'top5_category_indices' in visible['objects'][3]
    assert state == before


def test_compaction_retains_tentative_contract_actual_pixels_rollback_and_native_state(tmp_path):
    state = {'description': 'the lamp', 'allowed_target_ids': [1, 2], 'active_target_ids': [1, 2],
        'objects': [{'id': 1, 'category': 'lamp'}, {'id': 2, 'category': 'lamp'}],
        'views': [{'marked_ids': [1, 2]}, {'marked_ids': [1, 2]}], 'task_dag': None,
        'previous_actions': [], 'observations': [{'tool': 'monitor_check', 'no_evidence_acquired': True,
            'error': 'Rollback accepted; old exclusions reset'}, {'tool': 'monitor_check', 'no_evidence_acquired': True, 'error': 'latest rejection'}]}
    original = deepcopy(state)
    messages = [{'role': 'system', 'content': 'performing ONE final tentative evaluation; legal allowed intersect active, null, rivals; no Monitor'},
        {'role': 'user', 'content': [{'type': 'text', 'text': json.dumps(state)}, pixels(), pixels()]},
        {'role': 'user', 'content': json.dumps({'candidate_contract': {'latest_native_rollback_verbatim': state['observations'][0]},
                                             'terminal_status': 'tentative; strict_monitor_accepted=false'})}]
    direct = fresh.DirectCalls(tmp_path); direct.context_state = state; direct.preview_only = True
    with pytest.raises(fresh.PreparedCall) as caught:
        direct._qwen('t03-tentative-final-0', 'semantic_consultant', messages, {'type': 'object'})
    call = caught.value
    assert state == original and not direct.ledger
    assert 'performing ONE final tentative' in call.payload['messages'][0]['content']
    assert call.packing['original_images'] == 2 and call.packing['delivered_images'] == 1
    part = call.payload['messages'][1]['content'][1]
    assert part['max_pixels'] == fresh.IMAGE_MAX_PIXELS and 'max_pixels' not in part['image_url']
    decoded = context.decode_tables(json.loads(call.payload['messages'][1]['content'][0]['text']))
    assert 'Rollback accepted;' in json.dumps(decoded)
    assert call.payload['max_tokens'] == 3072


def test_invalid_packaging_records_failure_and_never_sends(tmp_path):
    direct = fresh.DirectCalls(tmp_path)
    direct.context_state = {'objects': [], 'views': [], 'observations': []}
    messages = [{'role': 'user', 'content': [pixels()]}]
    with patch.object(direct.session, 'post') as post, pytest.raises(RuntimeError, match='packaging failed before HTTP'):
        direct._qwen('failure', 'semantic_consultant', messages, {'type': 'object'})
    assert post.call_count == 0 and direct.r_used == 1
    assert direct.ledger[-1]['status'] == 'rejected_context_packaging' and not direct.ledger[-1]['http_sent']


@pytest.mark.parametrize('width,height', [(1, 1), (31, 65), (1800, 1200), (10000, 8000)])
def test_vision_bound_nonzero_and_capped(width, height):
    assert 0 < fresh.image_token_bound(width, height) <= fresh.IMAGE_MAX_PIXELS // 1024 + 2


def test_unknown_media_cannot_sneak_past_text_image_accounting():
    with pytest.raises(ValueError, match='Unsupported uncounted'):
        fresh.qwen_preflight([{'role': 'user', 'content': [{'type': 'input_audio', 'data': 'uncounted'}]}], {}, 100)
    with pytest.raises(ValueError, match='framing'):
        fresh.qwen_preflight([{'role': 'user', 'name': 'uncounted', 'content': 'x'}], {}, 100)


def large_pixels(size=(1000, 1000)):
    data = io.BytesIO(); Image.new('RGB', size, (35, 105, 170)).save(data, format='PNG')
    return {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(data.getvalue()).decode()}}


def test_adaptive_first_fit_is_byte_identical_and_does_not_try_other_stages():
    messages = [{'role': 'system', 'content': 'Return JSON'}, {'role': 'user', 'content': [large_pixels()]}]
    original = deepcopy(messages); preflight = fresh.qwen_preflight(messages, {}, 4096)
    with patch.object(context, 'tighten_overflow_metadata', side_effect=AssertionError('first fit must stay unchanged')):
        sent, audit, final = context.adapt_overflow(messages, {}, 4096, {}, preflight, fresh.qwen_preflight, 32768)
    assert sent == original and sent is messages and final is preflight
    assert audit['adaptive_selected_stage'] == 'initial' and len(audit['adaptive_attempts']) == 1


def test_adaptive_reversible_metadata_can_rescue_without_resizing_or_altering_measurement():
    state = {'description': 'the lamp', 'allowed_target_ids': [1, 2], 'active_target_ids': [1, 2],
        'objects': [{'id': i, 'category_score_type': 'uncalibrated_prior_not_probability'} for i in range(100)],
        'observations': [measurement()]}
    proposal = {'action': {'focus_ids': [1, 2, 3], 'monitor': {'node_updates': []}}}
    failure = {'previous_raw_answer': json.dumps(proposal), 'errors': [{'path': ['action', 'focus_ids'],
        'validator': 'maxItems', 'error': repr(proposal['action']['focus_ids']) + ' is expected to be empty'}]}
    root = {'state': context.pack_measurements(state), 'context_and_exact_feedback': [failure]}
    packet = context.encode_tables(root)
    messages = [{'role': 'system', 'content': 'Return JSON'}, {'role': 'user', 'content': [
        {'type': 'text', 'text': context.packed(packet)}]}]
    original = deepcopy(messages)
    tight, metadata = context.tighten_overflow_metadata(messages)
    before = fresh.qwen_preflight(messages, {}, 4096); after = fresh.qwen_preflight(tight, {}, 4096)
    assert after['maximum'] < before['maximum']
    checks = []
    def check(*args):
        value = fresh.qwen_preflight(*args); checks.append(value['maximum']); return value
    sent, audit, bound = context.adapt_overflow(messages, {}, 4096, {}, before, check, before['maximum'])
    assert audit['adaptive_selected_stage'] == 'metadata' and len(checks) == 1
    assert bound['maximum'] < before['maximum'] and not audit['image_derivations']
    decoded = context.decode_tables(json.loads(sent[1]['content'][0]['text']))
    assert context.unpack_measurements(decoded['state'])['observations'] == state['observations']
    error = decoded['context_and_exact_feedback'][0]['errors'][0]
    assert error['error_value_path'] == ['action', 'focus_ids'] and error['rule_suffix'] == ' is expected to be empty'
    assert messages == original and metadata['factored_errors'] == 1


def test_adaptive_image_derivatives_recounted_from_original_and_labeled_lossy():
    messages = [{'role': 'system', 'content': 'Return JSON'}, {'role': 'user', 'content': [large_pixels() for _ in range(4)]}]
    original = deepcopy(messages); before = fresh.qwen_preflight(messages, {}, 4096)
    checks = []
    def check(*args):
        value = fresh.qwen_preflight(*args); checks.append(value); return value
    sent, audit, final = context.adapt_overflow(messages, {}, 4096, {}, before, check, before['maximum'])
    assert final['maximum'] < before['maximum'] and len(checks) >= 2
    assert audit['adaptive_selected_stage'].startswith('images_max_side_')
    assert all(row['original_size'] == [1000, 1000] for row in audit['image_derivations'])
    for part, row in zip(sent[1]['content'][:4], audit['image_derivations']):
        raw = base64.b64decode(part['image_url']['url'].split(',', 1)[1])
        with Image.open(io.BytesIO(raw)) as image: assert list(image.size) == row['sent_size']
        assert row['lossy_downsampling'] and row['original_sha256'] != row['sent_sha256']
        assert row['scale_x'] == row['sent_size'][0] / 1000
    assert 'ONLY sent-resolution pixels' in sent[-1]['content'][-1]['text']
    assert messages == original


def test_all_adaptive_attempts_over_limit_never_send_and_archive_original(tmp_path):
    source = large_pixels()
    messages = [{'role': 'system', 'content': 'Return JSON'}, {'role': 'user', 'content': [
        {'type': 'text', 'text': 'x' * 40000}, source]}]
    before = deepcopy(messages); direct = fresh.DirectCalls(tmp_path)
    with patch.object(direct.session, 'post') as post, pytest.raises(fresh.ContextCapacityExceeded):
        direct._qwen('overflow', 'semantic_consultant', messages, {'type': 'object'})
    assert post.call_count == 0 and direct.r_used == 1 and len(direct.ledger) == 1
    record = direct.ledger[-1]; folder = next((tmp_path/'calls').iterdir())
    audit = json.loads((folder/'context-packing.json').read_text())
    assert len(audit['adaptive_attempts']) == 2
    assert audit['production_resize_fallback'] is False
    assert all(row['maximum'] >= 32768 for row in audit['adaptive_attempts'])
    assert not record['http_sent'] and record['status'] == 'rejected_preflight'
    raw = base64.b64decode(source['image_url']['url'].split(',', 1)[1])
    assert (folder/record['original_input_images'][0]['path']).read_bytes() == raw
    assert fresh.sha256(folder/record['input_images'][0]['path']) == record['input_images'][0]['sha256']
    assert messages == before


@pytest.mark.parametrize('last_bound', [32767, 32768])
def test_selector_role_final_boundary_keeps_original_pixels_and_never_recurses(tmp_path, last_bound):
    direct = fresh.DirectCalls(tmp_path)
    original = fresh.qwen_preflight
    def controlled(messages, schema, reserve):
        value = original(messages, schema, reserve)
        value['maximum'] = last_bound
        return value
    messages = [{'role': 'system', 'content': 'Return JSON'}, {'role': 'user', 'content': [large_pixels()]}]
    with patch.object(fresh, 'qwen_preflight', side_effect=controlled), patch.object(direct.session, 'post', return_value=response()) as post:
        if last_bound == 32768:
            with pytest.raises(fresh.ContextCapacityExceeded): direct._qwen('edge', 'evidence_selector', messages, {'type': 'object'})
            assert post.call_count == 0
        else:
            direct._qwen('edge', 'evidence_selector', messages, {'type': 'object'})
            assert post.call_count == 1
        folder = next((tmp_path/'calls').iterdir())
        audit = json.loads((folder/'context-packing.json').read_text())
        assert audit['adaptive_selected_stage'] == 'initial'
        assert audit['production_resize_fallback'] is False
        assert not (tmp_path/'evidence-selections').exists()
        assert audit['adaptive_attempts'][-1]['maximum'] == last_bound


def test_overflow_tentative_contract_and_raw_state_remain_unchanged():
    state = {'description': 'the lamp', 'allowed_target_ids': [1, 2], 'active_target_ids': [1, 2],
             'objects': [], 'observations': [measurement()]}
    packet = context.encode_tables({'state': context.pack_measurements(state), 'context_and_exact_feedback': []})
    system = 'performing ONE final tentative evaluation; allowed intersect active; null allowed; cite evidence and main_rivals; no Monitor changes'
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': [
        {'type': 'text', 'text': context.packed(packet)}, *[large_pixels() for _ in range(4)]]}]
    before = deepcopy(messages); first = fresh.qwen_preflight(messages, {}, 3072)
    sent, audit, _ = context.adapt_overflow(messages, {}, 3072, {}, first, fresh.qwen_preflight, first['maximum'])
    assert system in sent[0]['content'] and messages == before
    decoded = context.decode_tables(json.loads(sent[1]['content'][0]['text']))
    assert context.unpack_measurements(decoded['state']) == state
    assert not audit['native_monitor_mutated_by_adaptation']


def test_real_initial_wrapper_is_rejected_and_prior_valid_bare_shape_passes():
    fixture = json.loads((Path(__file__).parents[1]/'fixtures/qwen_step2_initial_monitor_wrapper.json').read_text())
    state = {'task_dag': None, 'monitor_requirements': fixture['monitor_requirements']}
    schema = context.compact_wire_schema({'properties': {'state_version': {'const': 'test'}, 'action': {}}}, state)
    monitor = schema['properties']['action']['anyOf'][0]['properties']['monitor']
    fresh.Draft202012Validator.check_schema(schema)
    original = deepcopy(fixture['actual_model_monitor'])
    with pytest.raises(fresh.ValidationError): fresh.validate(original, monitor)
    fresh.validate(fixture['valid_monitor_from_step1_v3'], monitor)
    repeated_tools = deepcopy(fixture['valid_monitor_from_step1_v3'])
    repeated_tools['nodes'][0]['allowed_actions'] = ['inspect', 'measure'] * 500
    with pytest.raises(fresh.ValidationError):
        fresh.validate(repeated_tools, monitor)
    incomplete = deepcopy(fixture['valid_monitor_from_step1_v3'])
    del incomplete['nodes'][0]['goal']
    with pytest.raises(fresh.ValidationError):
        fresh.validate(incomplete, monitor)
    missing_reason = {**fixture['valid_monitor_from_step1_v3'], 'revision_reason': ''}
    with pytest.raises(fresh.ValidationError):
        fresh.validate(missing_reason, monitor)
    # The real wrapped proposal also contains a finish-only non-verification node.
    with pytest.raises(fresh.ValidationError): fresh.validate(original['replace_plan'], monitor)
    assert fixture['actual_model_monitor'] == original
    later = context.compact_wire_schema({'properties': {'state_version': {'const': 'test'}, 'action': {}}},
                                        {**state, 'task_dag': {'nodes': []}})
    fresh.validate({'replace_plan': fixture['valid_monitor_from_step1_v3']},
                   later['properties']['action']['anyOf'][0]['properties']['monitor'])


def test_visual_wire_keeps_native_focus_limit():
    native = {'properties': {'state_version': {'const': 'test'}, 'action': {'anyOf': [
        {'properties': {'action': {'enum': ['inspect']}, 'focus_ids': {'maxItems': 8}}},
        {'properties': {'action': {'enum': ['measure']}, 'focus_ids': {'maxItems': 6}}},
    ]}}}
    schema = context.compact_wire_schema(native, {}, text_visual_reports=True)
    inspect, measure, _ = schema['properties']['action']['anyOf']
    fresh.validate(list(range(8)), inspect['properties']['focus_ids'])
    with pytest.raises(fresh.ValidationError):
        fresh.validate(list(range(9)), inspect['properties']['focus_ids'])
    assert measure['properties']['focus_ids']['maxItems'] == 6


def test_repair_feedback_uses_the_selected_monitor_node_branch():
    branches = [
        {'properties': {'current_node_id': {'enum': [node]}, 'status': {'enum': ['pending']}}}
        for node in ('n0', 'n1', 'n2')
    ]
    validator = fresh.Draft202012Validator({'anyOf': branches})
    failure = next(validator.iter_errors({'current_node_id': 'n1', 'status': 'completed'}))
    rows = fresh.complete_validation_feedback(failure, concise=True)
    assert [(row['path'], row['schema_path'][1]) for row in rows] == [(['status'], 1)]


def test_overflow_replacement_keeps_required_top_level_monitor_fields():
    schema = {'properties': {'state_version': {'const': 'test'}, 'action': {}}}
    state = {'task_dag': {'nodes': [{'id': 'n0'}]}, 'monitor_requirements': [{'id': 'r0'}]}
    full = context.compact_wire_schema(schema, state, text_visual_reports=True)
    compact = context.compact_wire_schema(schema, state, text_visual_reports=True, compact_replacement=True)
    def replacement(wire):
        return (wire['properties']['action']['anyOf'][0]['properties']['monitor']['anyOf'][1]
                ['properties']['replace_plan'])
    assert replacement(full)['required'] == replacement(compact)['required']
    assert len(context.packed(compact)) < len(context.packed(full))
    with pytest.raises(fresh.ValidationError):
        fresh.validate({'nodes': [{}]}, replacement(compact))
