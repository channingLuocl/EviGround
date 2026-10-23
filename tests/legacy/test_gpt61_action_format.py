"""The relay's equivalent action envelope must retain the model's explicit choices."""
import json
import unittest
from contextlib import nullcontext
from unittest.mock import patch

from jsonschema import Draft202012Validator

from seeground.api.qwen_api import (QwenAPIError, _chat_once, _chat_schema_retry, effective_timeout,
                      normalize_schema_payload, parse_r18_action_content,
                      parse_scene_grounding_content,
                      repair_missing_nullable_fields)


ACTION_SCHEMA = {
    'type': 'object',
    'properties': {
        'action': {'type': 'string', 'enum': ['inspect', 'measure', 'finish']},
        'reason': {'type': 'string', 'minLength': 1},
        'focus_ids': {'type': 'array', 'items': {'type': 'integer'}},
        'context': {'anyOf': [{'type': 'string', 'enum': ['scene', 'isolated']}, {'type': 'null'}]},
        'azimuth_deg': {'anyOf': [{'type': 'number'}, {'type': 'null'}]},
        'elevation_deg': {'anyOf': [{'type': 'number'}, {'type': 'null'}]},
        'frame': {'anyOf': [{'type': 'object'}, {'type': 'null'}]},
        'predicted_id': {'anyOf': [{'type': 'integer'}, {'type': 'null'}]},
        'evidence': {'type': 'array'},
        'unresolved': {'type': 'array'},
    },
    'required': ['action', 'reason', 'focus_ids', 'context', 'azimuth_deg',
                 'elevation_deg', 'frame', 'predicted_id', 'evidence', 'unresolved'],
    'additionalProperties': False,
}


class RelayActionFormatTests(unittest.TestCase):
    def test_scene_prose_without_explicit_focus_is_not_an_action(self):
        for content in ['I will inspect the scene first.', 'Let me measure the objects.',
                        'inspect()', 'inspect: focus_ids=[]']:
            with self.subTest(content=content):
                self.assertIsNone(parse_r18_action_content(content))

    def test_empty_focus_is_a_format_error_and_retried_before_any_tool(self):
        def response(focus):
            data = {'choices': [{'message': {'content': json.dumps({
                'action': 'inspect', 'focus_ids': focus, 'context': 'isolated'})},
                'finish_reason': 'stop'}], 'usage': {}}
            result = unittest.mock.Mock(ok=True, status_code=200, headers={}, text=json.dumps(data))
            result.json.return_value = data
            return result
        schema = {'name': 'r18_react_action', 'strict': True, 'schema': ACTION_SCHEMA}
        with patch('seeground.api.qwen_api.api_provider', return_value='openai'), \
                patch('seeground.api.qwen_api._settings', return_value=('https://example.invalid/v1', 'test-model', 'test-key')), \
                patch('seeground.api.qwen_api.api_request_slot', return_value=nullcontext(0.0)), \
                patch('seeground.api.qwen_api.requests.post', side_effect=[response([]), response([1, 2])]) as post:
            result = _chat_schema_retry([], json_mode=True, response_schema=schema)
        self.assertEqual(json.loads(result['_content'])['focus_ids'], [1, 2])
        self.assertEqual(post.call_count, 2)
        self.assertEqual(result['_baseline']['schema_retry'], 1)

    def test_quota_exhaustion_is_not_retried_as_a_temporary_rate_limit(self):
        failure = QwenAPIError('quota', diagnostic={'failure_kind': 'http_4xx',
            'http_status': 429, 'response_body': '{"code":"insufficient_quota"}'})
        with patch('seeground.api.qwen_api.api_provider', return_value='openai'), \
                patch('seeground.api.qwen_api._chat_once', side_effect=failure) as request, \
                patch('seeground.api.qwen_api.time.sleep') as sleep:
            with self.assertRaises(QwenAPIError):
                _chat_schema_retry([])
        self.assertEqual(request.call_count, 1)
        sleep.assert_not_called()

    def test_openai_read_timeout_is_raised_and_qwen_default_is_unchanged(self):
        with patch('seeground.api.qwen_api.api_provider', return_value='openai'):
            self.assertEqual(effective_timeout((15, 120)), (15, 240))
            with patch.dict('os.environ', {'SEEGROUND_OPENAI_READ_TIMEOUT_SECONDS': '300'}):
                self.assertEqual(effective_timeout((15, 120)), (15, 300))
        with patch('seeground.api.qwen_api.api_provider', return_value='qwen'):
            self.assertEqual(effective_timeout((15, 120)), (15, 120))

    def test_openai_transient_failure_gets_one_retry_without_using_format_budget(self):
        failure = QwenAPIError('relay timeout', diagnostic={'failure_kind': 'timeout'})
        success = {'usage': {}, '_baseline': {}}
        with patch('seeground.api.qwen_api.api_provider', return_value='openai'), \
                patch('seeground.api.qwen_api._chat_once', side_effect=[failure, success]) as request:
            result = _chat_schema_retry([], json_mode=True)

        self.assertEqual(request.call_count, 2)
        self.assertEqual(result['_baseline']['transport_retries'], 1)
        self.assertEqual(result['_baseline']['schema_retry'], 0)
        self.assertEqual(result['_baseline']['extra_attempts'], 1)

    def test_response_schema_is_included_in_the_first_request(self):
        schema = {'name': 'r18_react_action', 'strict': True, 'schema': ACTION_SCHEMA}
        success = {'usage': {}, '_baseline': {}}
        with patch('seeground.api.qwen_api.api_provider', return_value='openai'), \
                patch('seeground.api.qwen_api._chat_once', return_value=success) as request:
            _chat_schema_retry([{'role': 'user', 'content': 'choose'}],
                               json_mode=True, response_schema=schema)

        request_messages = request.call_args.args[0]
        self.assertIn('JSON Schema', request_messages[-1]['content'])
        self.assertIn('Include every required field', request_messages[-1]['content'])

    def test_invalid_schema_response_gets_four_total_format_attempts(self):
        schema = {'name': 'r18_react_action', 'strict': True, 'schema': ACTION_SCHEMA}
        failure = QwenAPIError('invalid response', diagnostic={
            'failure_kind': 'response_schema', 'validation_error': 'Expected a JSON object'})
        with patch('seeground.api.qwen_api.api_provider', return_value='openai'), \
                patch('seeground.api.qwen_api._chat_once', side_effect=[failure] * 4) as request:
            with self.assertRaises(QwenAPIError):
                _chat_schema_retry([], json_mode=True, response_schema=schema)

        self.assertEqual(request.call_count, 4)

    def test_qwen_transport_failure_is_not_retried(self):
        failure = QwenAPIError('timeout', diagnostic={'failure_kind': 'timeout'})
        with patch('seeground.api.qwen_api.api_provider', return_value='qwen'), \
                patch('seeground.api.qwen_api._chat_once', side_effect=failure) as request:
            with self.assertRaises(QwenAPIError):
                _chat_schema_retry([])
        self.assertEqual(request.call_count, 1)

    def test_chat_validates_and_records_normalized_action_envelope(self):
        response_data = {'choices': [{'message': {'content': json.dumps({
            'action_type': 'inspect', 'parameters': {
                'focus_ids': [10, 11], 'isolated': False,
                'azimuth_deg': 45, 'elevation_deg': 30}})},
            'finish_reason': 'stop'}], 'usage': {'total_tokens': 12}}

        class Response:
            ok = True
            status_code = 200
            headers = {}
            text = json.dumps(response_data)

            @staticmethod
            def json():
                return response_data

        schema = {'name': 'r18_react_action', 'strict': True, 'schema': ACTION_SCHEMA}
        with patch('seeground.api.qwen_api.api_provider', return_value='openai'), \
                patch('seeground.api.qwen_api._settings', return_value=('https://example.invalid/v1', 'test-model', 'test-key')), \
                patch('seeground.api.qwen_api.api_request_slot', return_value=nullcontext(0.0)), \
                patch('seeground.api.qwen_api.requests.post', return_value=Response()) as post:
            result = _chat_once([], json_mode=True, response_schema=schema)

        normalized = json.loads(result['_content'])
        self.assertEqual(normalized['action'], 'inspect')
        self.assertEqual(normalized['focus_ids'], [10, 11])
        self.assertEqual(normalized['context'], 'scene')
        self.assertIn('action_type->action', result['_baseline']['response_normalization'])
        self.assertEqual(post.call_args.kwargs['timeout'], (15, 240))
        self.assertEqual(post.call_count, 1)

    def test_action_type_parameters_envelope_is_normalized_without_changing_choice(self):
        payload = {'action_type': 'inspect', 'parameters': {
            'focus_ids': [10, 11, 17], 'isolated': True,
            'azimuth_deg': 30, 'elevation_deg': 25}}

        normalized, repairs = normalize_schema_payload('r18_react_action', payload)
        normalized, _ = repair_missing_nullable_fields(normalized, ACTION_SCHEMA)

        Draft202012Validator(ACTION_SCHEMA).validate(normalized)
        self.assertEqual(normalized['action'], 'inspect')
        self.assertEqual(normalized['focus_ids'], [10, 11, 17])
        self.assertEqual(normalized['context'], 'isolated')
        self.assertEqual(normalized['azimuth_deg'], 30)
        self.assertEqual(normalized['elevation_deg'], 25)
        self.assertNotIn('action_type', normalized)
        self.assertNotIn('parameters', normalized)
        self.assertIn('action_type->action', repairs)

    def test_finish_repair_does_not_invent_target_or_evidence(self):
        payload = {'action_type': 'finish', 'parameters': {'predicted_id': 14}}

        normalized, _ = normalize_schema_payload('r18_react_action', payload)
        normalized, _ = repair_missing_nullable_fields(normalized, ACTION_SCHEMA)

        self.assertEqual(normalized['action'], 'finish')
        self.assertEqual(normalized['predicted_id'], 14)
        self.assertEqual(normalized['evidence'], [])
        self.assertEqual(normalized['focus_ids'], [])

    def test_top_level_isolated_flag_maps_to_scene_context(self):
        payload = {'action_type': 'inspect', 'focus_ids': [12, 13], 'isolated': False}

        normalized, repairs = normalize_schema_payload('r18_react_action', payload)

        self.assertEqual(normalized['action'], 'inspect')
        self.assertEqual(normalized['context'], 'scene')
        self.assertNotIn('isolated', normalized)
        self.assertIn('isolated->context', repairs)

    def test_view_context_alias_is_preserved(self):
        payload = {'action_type': 'inspect', 'focus_ids': [1, 2], 'view': 'scene'}

        normalized, repairs = normalize_schema_payload('r18_react_action', payload)

        self.assertEqual(normalized['action'], 'inspect')
        self.assertEqual(normalized['context'], 'scene')
        self.assertNotIn('view', normalized)
        self.assertIn('view->context', repairs)

    def test_mode_alias_replaces_null_context_and_validates(self):
        payload = {'action': 'inspect', 'focus_ids': [1, 4], 'mode': 'scene',
                   'context': None, 'azimuth_deg': 0, 'elevation_deg': 10,
                   'frame': None, 'predicted_id': None, 'evidence': [], 'unresolved': []}

        normalized, repairs = normalize_schema_payload('r18_react_action', payload)
        normalized, _ = repair_missing_nullable_fields(normalized, ACTION_SCHEMA)

        Draft202012Validator(ACTION_SCHEMA).validate(normalized)
        self.assertEqual(normalized['context'], 'scene')
        self.assertNotIn('mode', normalized)
        self.assertIn('mode->context', repairs)

    def test_params_and_render_mode_aliases_are_normalized(self):
        payload = {'action_type': 'inspect', 'params': {
            'focus_ids': [5, 8, 12], 'isolated': True,
            'azimuth_deg': 0, 'elevation_deg': 15, 'render_mode': 'isolated'}}

        normalized, repairs = normalize_schema_payload('r18_react_action', payload)
        normalized, _ = repair_missing_nullable_fields(normalized, ACTION_SCHEMA)

        Draft202012Validator(ACTION_SCHEMA).validate(normalized)
        self.assertEqual(normalized['action'], 'inspect')
        self.assertEqual(normalized['focus_ids'], [5, 8, 12])
        self.assertEqual(normalized['context'], 'isolated')
        self.assertIn('params->parameters', repairs)
        self.assertIn('parameters.render_mode->context', repairs)

    def test_function_style_inspect_is_recovered(self):
        payload = parse_r18_action_content(
            'inspect(6, 12, 34, azimuth_deg=0, elevation_deg=30, mode="scene")')
        self.assertEqual(payload, {'action': 'inspect', 'focus_ids': [6, 12, 34],
                                   'context': 'scene', 'azimuth_deg': 0,
                                   'elevation_deg': 30})

    def test_colon_style_inspect_is_recovered(self):
        payload = parse_r18_action_content(
            'inspect: focus_ids=[16,35,37], mode=scene, azimuth_deg=45, elevation_deg=30')
        self.assertEqual(payload, {'action': 'inspect', 'focus_ids': [16, 35, 37],
                                   'context': 'scene', 'azimuth_deg': 45,
                                   'elevation_deg': 30})

    def test_embedded_json_inside_prose_is_recovered(self):
        payload = parse_r18_action_content(
            'I will inspect. `inspect({"focus_ids":[0,1],"render_mode":"scene"})`')
        self.assertEqual(payload['action'], 'inspect')
        self.assertEqual(payload['focus_ids'], [0, 1])
        self.assertEqual(payload['render_mode'], 'scene')

    def test_finish_text_recovers_explicit_id_without_inventing_evidence(self):
        payload = parse_r18_action_content('Answer: id 7')
        self.assertEqual(payload, {'action': 'finish', 'predicted_id': 7})

    def test_scene_aliases_are_normalized_without_overwriting_explicit_null(self):
        normalized, repairs = normalize_schema_payload('scene_grounding', {
            'target_id': 35, 'reasoning': 'The target is closest.',
            'predicted_id': None, 'evidence': [], 'unresolved': []})
        self.assertIsNone(normalized['predicted_id'])
        self.assertEqual(normalized['reason'], 'The target is closest.')
        self.assertNotIn('target_id', normalized)
        self.assertNotIn('reasoning', normalized)
        self.assertIn('reasoning->reason', repairs)

    def test_scene_plain_text_recovery_is_conservative(self):
        payload = parse_scene_grounding_content('Based on the scene, the answer is object id 44.')
        self.assertEqual(payload['predicted_id'], 44)
        self.assertEqual(payload['evidence'], [])
        self.assertEqual(parse_scene_grounding_content('**Target: 23**')['predicted_id'], 23)
        self.assertEqual(parse_scene_grounding_content('57')['predicted_id'], 57)
        self.assertIsNone(parse_scene_grounding_content('No target can be distinguished.'))

    def test_scene_scalar_response_is_recovered_before_schema_validation(self):
        response_data = {'choices': [{'message': {'content': '57'}, 'finish_reason': 'stop'}],
                         'usage': {}}

        class Response:
            ok = True
            status_code = 200
            headers = {}
            text = '57'

            @staticmethod
            def json():
                return response_data

        schema = {'name': 'scene_grounding', 'strict': True, 'schema': {
            'type': 'object',
            'properties': {
                'predicted_id': {'type': ['integer', 'null']},
                'reason': {'type': 'string'}, 'evidence': {'type': 'array'},
                'unresolved': {'type': 'array'},
            },
            'required': ['predicted_id', 'reason', 'evidence', 'unresolved'],
            'additionalProperties': False,
        }}
        with patch('seeground.api.qwen_api.api_provider', return_value='openai'), \
                patch('seeground.api.qwen_api._settings', return_value=('https://example.invalid/v1', 'test-model', 'test-key')), \
                patch('seeground.api.qwen_api.api_request_slot', return_value=nullcontext(0.0)), \
                patch('seeground.api.qwen_api.requests.post', return_value=Response()):
            result = _chat_once([], json_mode=True, response_schema=schema)

        self.assertEqual(json.loads(result['_content'])['predicted_id'], 57)

    def test_other_schemas_are_untouched(self):
        payload = {'action_type': 'inspect', 'parameters': {'focus_ids': [1]}}
        normalized, repairs = normalize_schema_payload('parser', payload)
        self.assertEqual(normalized, payload)
        self.assertEqual(repairs, [])


if __name__ == '__main__':
    unittest.main()
