"""Mock-only relay provider regression tests; never contacts a model service."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import requests
from PIL import Image

from qwen_api import QwenAPIError, _settings, chat, image_part, _redact
from react_decision import validate_react_config
from real_scene import parse_query, run_query
import test_real_scene


class OpenAIAPITests(unittest.TestCase):
    def setUp(self):
        env = patch.dict('os.environ', {
            'SEEGROUND_API_PROVIDER': 'openai', 'SEEGROUND_REACT_BACKEND': 'openai',
            'SEEGROUND_BASE_URL': 'http://relay.invalid:3000/v1',
            'SEEGROUND_ALLOW_HTTP': '1', 'SEEGROUND_MODEL': 'gpt-6.1-sol',
            'SEEGROUND_API_KEY': 'mock-relay-key',
            'SEEGROUND_OPENAI_TOKEN_PARAMETER': 'max_completion_tokens',
            'SEEGROUND_OPENAI_TEMPERATURE': '',
        })
        env.start()
        self.addCleanup(env.stop)

    def success(self):
        return SimpleNamespace(ok=True, status_code=200, text='{}', headers={},
            json=lambda: {'choices': [{'message': {'content': '{"ok":true}'},
                                      'finish_reason': 'stop'}]})

    def test_gpt_text_json_uses_completion_limit_without_qwen_parameters(self):
        with patch('qwen_api.requests.post', return_value=self.success()) as post:
            result = chat([{'role': 'user', 'content': 'Return JSON {"ok":true}'}],
                          max_tokens=900, json_mode=True)
        payload = post.call_args.kwargs['json']
        self.assertEqual(payload['model'], 'gpt-6.1-sol')
        self.assertEqual(payload['max_completion_tokens'], 900)
        self.assertEqual(payload['response_format'], {'type': 'json_object'})
        self.assertNotIn('enable_thinking', payload)
        self.assertNotIn('temperature', payload)
        self.assertNotIn('max_tokens', payload)
        self.assertEqual(result['_baseline']['provider'], 'openai')
        self.assertEqual(validate_react_config(), 'openai')

    def test_gpt_image_json_preserves_multimodal_content(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'probe.png'; Image.new('RGB', (8, 8), 'red').save(path)
            messages = [{'role': 'user', 'content': [image_part(path),
                         {'type': 'text', 'text': 'Describe the color in JSON.'}]}]
            with patch('qwen_api.requests.post', return_value=self.success()) as post:
                result = chat(messages, json_mode=True)
        self.assertEqual(post.call_args.kwargs['json']['messages'], messages)
        self.assertTrue(result['_baseline']['request_summary']['contains_images'])
        self.assertNotIn('base64', json.dumps(result['_baseline']))

    def test_explicit_legacy_token_parameter_and_temperature_do_not_auto_retry(self):
        with patch.dict('os.environ', {'SEEGROUND_OPENAI_TOKEN_PARAMETER': 'max_tokens',
                                       'SEEGROUND_OPENAI_TEMPERATURE': '0.2'}), \
             patch('qwen_api.requests.post', return_value=self.success()) as post:
            chat([], max_tokens=768)
        self.assertEqual(post.call_count, 1)
        self.assertEqual(post.call_args.kwargs['json']['max_tokens'], 768)
        self.assertEqual(post.call_args.kwargs['json']['temperature'], .2)

    def test_invalid_parameter_configuration_fails_before_network(self):
        for variable, value in (('SEEGROUND_OPENAI_TOKEN_PARAMETER', 'invented'),
                                ('SEEGROUND_OPENAI_TEMPERATURE', 'nan'),
                                ('SEEGROUND_OPENAI_TEMPERATURE', 'invalid')):
            with self.subTest(variable=variable, value=value), \
                 patch.dict('os.environ', {variable: value}), patch('qwen_api.requests.post') as post:
                with self.assertRaises(QwenAPIError):
                    chat([])
                post.assert_not_called()

    def test_http_requires_explicit_configuration_and_webpage_is_rejected(self):
        with patch.dict('os.environ', {'SEEGROUND_ALLOW_HTTP': ''}):
            with self.assertRaisesRegex(QwenAPIError, 'ALLOW_HTTP'):
                _settings()
        for address in ('http://relay.invalid:3000/pricing',
                        'https://user:password@relay.invalid/v1',
                        'https://relay.invalid/v1?api_key=secret'):
            with self.subTest(address=address), patch.dict('os.environ', {'SEEGROUND_BASE_URL': address}):
                with self.assertRaises(QwenAPIError):
                    _settings()

    def test_qwen_compatible_mode_base_and_parameters_are_preserved(self):
        with patch.dict('os.environ', {'SEEGROUND_API_PROVIDER': 'qwen',
                'SEEGROUND_BASE_URL': 'https://dashscope.invalid/compatible-mode/v1'}), \
             patch('qwen_api.requests.post', return_value=self.success()) as post:
            chat([], max_tokens=600, json_mode=True)
        self.assertEqual(post.call_args.args[0],
                         'https://dashscope.invalid/compatible-mode/v1/chat/completions')
        payload = post.call_args.kwargs['json']
        self.assertEqual(payload['max_tokens'], 600)
        self.assertEqual(payload['temperature'], 0.0)
        self.assertIs(payload['enable_thinking'], False)
        self.assertNotIn('max_completion_tokens', payload)

    def test_parser_records_actual_gpt_request_options_and_backend_conflict_fails(self):
        parsed = {'Target': ['shelf'], 'Anchor': [], 'Relations': [],
                  'Observer': {}, 'Attributes': [], 'Unhandled': []}
        response = SimpleNamespace(ok=True, status_code=200, text='{}', headers={},
            json=lambda: {'choices': [{'message': {'content': json.dumps(parsed)},
                                      'finish_reason': 'stop'}]})
        calls = []
        with patch('qwen_api.requests.post', return_value=response):
            parse_query('the shelf', mode='active_diagnostic', calls=calls)
        self.assertEqual(calls[0]['max_completion_tokens'], 1600)
        self.assertEqual(calls[0]['request_summary']['provider'], 'openai')
        self.assertNotIn('enable_thinking', calls[0])
        with patch.dict('os.environ', {'SEEGROUND_REACT_BACKEND': 'qwen'}):
            with self.assertRaisesRegex(ValueError, 'match'):
                validate_react_config()

    def test_gpt_invalid_json_content_fails_without_repair_or_prediction(self):
        for content in ('invalid JSON', '[{"ok":true}]', '', '{"Target":'):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as td:
                root = Path(td); test_real_scene.RealSceneTests().scene(root); out = root / 'failed'
                response = SimpleNamespace(ok=True, status_code=200,
                    text=json.dumps({'choices': [{'message': {'content': content},
                                                  'finish_reason': 'stop'}]}),
                    headers={'x-request-id': 'json-test-id'},
                    json=lambda: {'choices': [{'message': {'content': content},
                                              'finish_reason': 'stop'}]})
                with patch('qwen_api.requests.post', return_value=response) as post:
                    with self.assertRaises(QwenAPIError):
                        run_query(root, 'shelf', out, mode='active_diagnostic')
                self.assertEqual(post.call_count, 1)
                record = json.loads((out / 'run.json').read_text())
                diagnostic = record['error_diagnostic']
                self.assertEqual(diagnostic['failure_kind'], 'response_parse')
                self.assertEqual(diagnostic['client_stage'], 'response_json_parse')
                self.assertEqual(diagnostic['request_id'], 'json-test-id')
                self.assertEqual(diagnostic['http_status'], 200)
                self.assertFalse((out / 'prediction.json').exists())

    def test_gpt_errors_are_diagnostic_single_attempt_without_prediction(self):
        cases = [(400, 'http_4xx'), (404, 'http_4xx'), (503, 'http_5xx'),
                 ('timeout', 'timeout'), ('parse', 'response_parse')]
        for code, kind in cases:
            with self.subTest(code=code), tempfile.TemporaryDirectory() as td:
                root = Path(td); test_real_scene.RealSceneTests().scene(root); out = root / 'failed'
                response = SimpleNamespace(ok=False, status_code=code if type(code) is int else 200,
                    text='invalid: api_key=mock-relay-key data:image/png;base64,QUJDRA==',
                    headers={'x-request-id': 'relay-test-id'}, json=lambda: {})
                if code == 'parse':
                    response.ok = True
                with patch('qwen_api.requests.post', return_value=response,
                           side_effect=requests.Timeout() if code == 'timeout' else None) as post:
                    with self.assertRaises(QwenAPIError):
                        run_query(root, 'water cooler', out, mode='active_diagnostic')
                self.assertEqual(post.call_count, 1)
                record = json.loads((out / 'run.json').read_text())
                diagnostic = record['error_diagnostic']
                self.assertEqual(diagnostic['failure_kind'], kind)
                self.assertEqual(diagnostic['model'], 'gpt-6.1-sol')
                self.assertEqual(diagnostic['request_summary']['provider'], 'openai')
                self.assertEqual(record['error_stage'], 'parse_query')
                self.assertFalse((out / 'prediction.json').exists())
                saved = json.dumps(record)
                self.assertNotIn('mock-relay-key', saved)
                self.assertNotIn('QUJDRA==', saved)

    def test_safe_long_raw_response_keeps_format_and_image_echo_is_redacted(self):
        raw = '{"detail":"' + ('word ' * 1200) + '"}'
        self.assertEqual(_redact(raw, limit=None), raw)
        sanitized = _redact('{"base64":"QUJDRA==","api_key":"mock-relay-key"}', limit=None)
        self.assertNotIn('QUJDRA==', sanitized)
        self.assertNotIn('mock-relay-key', sanitized)


if __name__ == '__main__':
    unittest.main()
