"""Thinking configuration must reach requests, provenance and checkpoint identity."""
from copy import deepcopy
import os
import unittest
from unittest.mock import patch

from qwen_api import QwenAPIError, api_config_summary, chat, request_options


class ThinkingAPITests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {
            'SEEGROUND_API_PROVIDER': 'openai', 'SEEGROUND_ENABLE_THINKING': '0',
            'SEEGROUND_THINKING_TOKEN_FLOOR': '8192',
            'SEEGROUND_OPENAI_TOKEN_PARAMETER': 'max_completion_tokens',
            'SEEGROUND_OPENAI_TEMPERATURE': '0',
            'SEEGROUND_BASE_URL': 'https://example.invalid/v1',
            'SEEGROUND_API_KEY': 'test-only', 'SEEGROUND_MODEL': 'qwen3.7-plus',
        })
        env.start(); self.addCleanup(env.stop)

    def test_disabled_preserves_legacy_token_budget(self):
        options = request_options(1600, None, True)
        self.assertIs(options['enable_thinking'], False)
        self.assertEqual(options['max_completion_tokens'], 1600)
        self.assertNotIn('thinking_token_floor', api_config_summary())

    def test_thinking_preserves_strict_json_with_room_for_reasoning(self):
        schema = {'name': 'probe', 'strict': True, 'schema': {
            'type': 'object', 'properties': {'ok': {'type': 'boolean'}},
            'required': ['ok'], 'additionalProperties': False}}
        with patch.dict(os.environ, {'SEEGROUND_ENABLE_THINKING': '1'}):
            options = request_options(1600, None, True, schema)
            config = api_config_summary()
        self.assertIs(options['enable_thinking'], True)
        self.assertEqual(options['max_completion_tokens'], 8192)
        self.assertEqual(options['temperature'], 0)
        self.assertEqual(options['response_format']['json_schema'], schema)
        self.assertEqual(config['thinking_token_floor'], 8192)
        self.assertIs(config['request_options']['enable_thinking'], True)

    def test_requested_larger_budget_is_not_reduced(self):
        with patch.dict(os.environ, {'SEEGROUND_ENABLE_THINKING': '1'}):
            self.assertEqual(request_options(10000, None, True)['max_completion_tokens'], 10000)

    def test_qwen_token_parameter_uses_the_same_thinking_floor(self):
        with patch.dict(os.environ, {'SEEGROUND_API_PROVIDER': 'qwen',
                                    'SEEGROUND_ENABLE_THINKING': '1'}):
            options = request_options(1600, None, True)
        self.assertIs(options['enable_thinking'], True)
        self.assertEqual(options['max_tokens'], 8192)

    def test_non_json_calls_preserve_existing_behavior(self):
        with patch.dict(os.environ, {'SEEGROUND_ENABLE_THINKING': '1'}):
            options = request_options(1600, None, False)
        self.assertNotIn('enable_thinking', options)
        self.assertEqual(options['max_completion_tokens'], 1600)

    def test_invalid_thinking_settings_fail_before_request(self):
        for values in ({'SEEGROUND_ENABLE_THINKING': 'yes'},
                       *({'SEEGROUND_ENABLE_THINKING': '1',
                          'SEEGROUND_THINKING_TOKEN_FLOOR': value}
                         for value in ('0', '-1', 'invalid'))):
            with self.subTest(values=values), patch.dict(os.environ, values):
                with self.assertRaises(QwenAPIError):
                    request_options(1600, None, True)

    def test_checkpoint_cannot_reuse_another_thinking_budget_or_model(self):
        from v5_runtime import sample_recovery
        budget = {'used': False, 'attempts': [], 'api_checkpoint': True,
                  'resuming': True, 'completed_requests': {}}
        token = sample_recovery.set(budget)
        self.addCleanup(sample_recovery.reset, token)
        response = {'_content': '{"ok":true}', 'usage': {}, '_baseline': {
            'api_attempts': [{'status': 'success'}]}}
        with patch('v4_policy.v5_enabled', return_value=True), \
                patch('v5_runtime.persist_api_recovery'), \
                patch('qwen_api._chat_schema_retry', side_effect=lambda *a, **k: deepcopy(response)) as api:
            chat([], max_tokens=1600, json_mode=True)
            cached = chat([], max_tokens=1600, json_mode=True)
            self.assertTrue(cached['_baseline']['cache_hit'])
            self.assertEqual(api.call_count, 1)
            with patch.dict(os.environ, {'SEEGROUND_ENABLE_THINKING': '1'}):
                chat([], max_tokens=1600, json_mode=True)
                self.assertEqual(api.call_count, 2)
                with patch.dict(os.environ, {'SEEGROUND_THINKING_TOKEN_FLOOR': '4096'}):
                    chat([], max_tokens=1600, json_mode=True)
                    self.assertEqual(api.call_count, 3)
                with patch.dict(os.environ, {'SEEGROUND_MODEL': 'another-model'}):
                    chat([], max_tokens=1600, json_mode=True)
                    self.assertEqual(api.call_count, 4)


if __name__ == '__main__':
    unittest.main()
