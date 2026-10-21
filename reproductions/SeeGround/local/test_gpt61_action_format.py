"""The relay's equivalent action envelope must retain the model's explicit choices."""
import json
import unittest
from contextlib import nullcontext
from unittest.mock import patch

from jsonschema import Draft202012Validator

from qwen_api import (QwenAPIError, _chat_once, _chat_schema_retry, effective_timeout,
                      normalize_schema_payload, repair_missing_nullable_fields)


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
    def test_openai_read_timeout_is_raised_and_qwen_default_is_unchanged(self):
        with patch('qwen_api.api_provider', return_value='openai'):
            self.assertEqual(effective_timeout((15, 120)), (15, 240))
            with patch.dict('os.environ', {'SEEGROUND_OPENAI_READ_TIMEOUT_SECONDS': '300'}):
                self.assertEqual(effective_timeout((15, 120)), (15, 300))
        with patch('qwen_api.api_provider', return_value='qwen'):
            self.assertEqual(effective_timeout((15, 120)), (15, 120))

    def test_openai_transient_failure_gets_one_retry_without_using_format_budget(self):
        failure = QwenAPIError('relay timeout', diagnostic={'failure_kind': 'timeout'})
        success = {'usage': {}, '_baseline': {}}
        with patch('qwen_api.api_provider', return_value='openai'), \
                patch('qwen_api._chat_once', side_effect=[failure, success]) as request:
            result = _chat_schema_retry([], json_mode=True)

        self.assertEqual(request.call_count, 2)
        self.assertEqual(result['_baseline']['transport_retries'], 1)
        self.assertEqual(result['_baseline']['schema_retry'], 0)
        self.assertEqual(result['_baseline']['extra_attempts'], 1)

    def test_response_schema_is_included_in_the_first_request(self):
        schema = {'name': 'r18_react_action', 'strict': True, 'schema': ACTION_SCHEMA}
        success = {'usage': {}, '_baseline': {}}
        with patch('qwen_api.api_provider', return_value='openai'), \
                patch('qwen_api._chat_once', return_value=success) as request:
            _chat_schema_retry([{'role': 'user', 'content': 'choose'}],
                               json_mode=True, response_schema=schema)

        request_messages = request.call_args.args[0]
        self.assertIn('JSON Schema', request_messages[-1]['content'])
        self.assertIn('Include every required field', request_messages[-1]['content'])

    def test_invalid_schema_response_gets_four_total_format_attempts(self):
        schema = {'name': 'r18_react_action', 'strict': True, 'schema': ACTION_SCHEMA}
        failure = QwenAPIError('invalid response', diagnostic={
            'failure_kind': 'response_schema', 'validation_error': 'Expected a JSON object'})
        with patch('qwen_api.api_provider', return_value='openai'), \
                patch('qwen_api._chat_once', side_effect=[failure] * 4) as request:
            with self.assertRaises(QwenAPIError):
                _chat_schema_retry([], json_mode=True, response_schema=schema)

        self.assertEqual(request.call_count, 4)

    def test_qwen_transport_failure_is_not_retried(self):
        failure = QwenAPIError('timeout', diagnostic={'failure_kind': 'timeout'})
        with patch('qwen_api.api_provider', return_value='qwen'), \
                patch('qwen_api._chat_once', side_effect=failure) as request:
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
        with patch('qwen_api.api_provider', return_value='openai'), \
                patch('qwen_api._settings', return_value=('https://example.invalid/v1', 'test-model', 'test-key')), \
                patch('qwen_api.api_request_slot', return_value=nullcontext(0.0)), \
                patch('qwen_api.requests.post', return_value=Response()) as post:
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

    def test_other_schemas_are_untouched(self):
        payload = {'action_type': 'inspect', 'parameters': {'focus_ids': [1]}}
        normalized, repairs = normalize_schema_payload('parser', payload)
        self.assertEqual(normalized, payload)
        self.assertEqual(repairs, [])


if __name__ == '__main__':
    unittest.main()
