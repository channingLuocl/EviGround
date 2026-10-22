import json
import unittest
import requests
from types import SimpleNamespace
from unittest.mock import patch
from qwen_api import chat, QwenAPIError, _redact, api_config_summary
from response_schemas import obj, enum, array, schema_kwargs


class QwenAPITests(unittest.TestCase):
    def setUp(self):
        env = patch.dict('os.environ', {'SEEGROUND_API_PROVIDER': 'qwen', 'SEEGROUND_V5': '0'})
        env.start()
        self.addCleanup(env.stop)

    def test_request_and_finish_reason(self):
        response=SimpleNamespace(ok=True,json=lambda:{'choices':[{'finish_reason':'length',
                                  'message':{'content':'Predicted ID: 1'}}],'usage':{'total_tokens':12}})
        with patch.dict('os.environ',{'SEEGROUND_MODEL':'mock','SEEGROUND_BASE_URL':'https://example.invalid/v1',
                                    'SEEGROUND_API_KEY':'test-only'}), \
             patch('qwen_api.requests.post',return_value=response) as post:
            result=chat([{'role':'user','content':'test'}],max_tokens=768)
        self.assertEqual(post.call_args.args[0],'https://example.invalid/v1/chat/completions')
        self.assertEqual(post.call_args.kwargs['json']['max_tokens'],768)
        self.assertNotIn('response_format',post.call_args.kwargs['json'])
        self.assertNotIn('enable_thinking',post.call_args.kwargs['json'])
        self.assertEqual(result['_finish_reason'],'length')
        self.assertEqual(result['_content'],'Predicted ID: 1')

    def test_json_mode_sets_response_format_and_disables_thinking(self):
        response=SimpleNamespace(ok=True,json=lambda:{'choices':[{'finish_reason':'stop',
                                  'message':{'content':'{"Target":[]}'}}]})
        with patch.dict('os.environ',{'SEEGROUND_MODEL':'qwen3-vl-flash','SEEGROUND_BASE_URL':'https://example.invalid/v1',
                                    'SEEGROUND_API_KEY':'test-only'}), \
             patch('qwen_api.requests.post',return_value=response) as post:
            chat([{'role':'user','content':'Return JSON'}],json_mode=True)
        body=post.call_args.kwargs['json']
        self.assertEqual(body['response_format'],{'type':'json_object'})
        self.assertIs(body['enable_thinking'],False)

    def test_bad_response_is_value_error(self):
        with patch.dict('os.environ',{'SEEGROUND_MODEL':'mock','SEEGROUND_BASE_URL':'https://example.invalid/v1',
                                    'SEEGROUND_API_KEY':'test-only'}), \
             patch('qwen_api.requests.post',return_value=SimpleNamespace(ok=True,json=lambda:{})):
            with self.assertRaises(QwenAPIError):
                chat([])

    def test_http_400_records_redacted_diagnostic_without_retry(self):
        response = SimpleNamespace(
            ok=False,
            status_code=400,
            text='bad request: token=secret-token data:image/png;base64,QUJDRA==',
            headers={'x-request-id': 'req-400'},
        )
        messages = [{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,QUJDRA=='}},
            {'type': 'text', 'text': 'inspect'},
        ]}]
        with patch.dict('os.environ', {'SEEGROUND_MODEL': 'qwen-test',
                                       'SEEGROUND_BASE_URL': 'https://example.invalid/v1',
                                       'SEEGROUND_API_KEY': 'secret-api-key'}), \
             patch('qwen_api.requests.post', return_value=response) as post:
            with self.assertRaises(QwenAPIError) as raised:
                chat(messages, json_mode=True)
        self.assertEqual(post.call_count, 1)
        diagnostic = raised.exception.diagnostic
        self.assertEqual(diagnostic['failure_kind'], 'http_4xx')
        self.assertEqual(diagnostic['http_status'], 400)
        self.assertEqual(diagnostic['request_id'], 'req-400')
        self.assertEqual(diagnostic['model'], 'qwen-test')
        self.assertIn('bad request', diagnostic['response_body'])
        rendered = json.dumps(diagnostic, ensure_ascii=False)
        self.assertNotIn('secret-api-key', rendered)
        self.assertNotIn('secret-token', rendered)
        self.assertNotIn('data:image/png;base64,QUJDRA==', rendered)
        self.assertNotIn('Authorization', rendered)
        self.assertEqual(diagnostic['request_summary']['contains_images'], True)

    def test_failure_kinds_distinguish_5xx_timeout_and_response_parse(self):
        cases = [
            (SimpleNamespace(ok=False, status_code=503, text='unavailable', headers={}), None, 'http_5xx'),
            (None, requests.Timeout('timeout'), 'timeout'),
            (SimpleNamespace(ok=True, status_code=200, text='invalid body', headers={},
                             json=lambda: {}), None, 'response_parse'),
            (SimpleNamespace(ok=True, status_code=200, text='invalid message', headers={},
                             json=lambda: {'choices': [{'message': None}]}), None, 'response_parse'),
        ]
        for response, error, kind in cases:
            with self.subTest(kind=kind), patch.dict('os.environ', {
                    'SEEGROUND_MODEL': 'mock', 'SEEGROUND_BASE_URL': 'https://example.invalid/v1',
                    'SEEGROUND_API_KEY': 'test-only'}), patch('qwen_api.requests.post',
                    return_value=response, side_effect=error) as post:
                with self.assertRaises(QwenAPIError) as raised:
                    chat([])
            self.assertEqual(post.call_count, 1)
            self.assertEqual(raised.exception.diagnostic['failure_kind'], kind)

    def test_json_error_body_redacts_credentials_and_raw_image_base64(self):
        body={'error':'invalid image', 'api_key':'secret-api-key',
              'access_token':'secret-token', 'b64_json':'QUJDRA==',
              'request_id':'body-400'}
        response=SimpleNamespace(ok=False,status_code=400,text=json.dumps(body),headers={},json=lambda:body)
        with patch.dict('os.environ',{'SEEGROUND_MODEL':'mock',
                'SEEGROUND_BASE_URL':'https://example.invalid/v1',
                'SEEGROUND_API_KEY':'secret-api-key'}), \
             patch('qwen_api.requests.post',return_value=response):
            with self.assertRaises(QwenAPIError) as error:
                chat([])
        diagnostic=error.exception.diagnostic
        self.assertEqual(diagnostic['request_id'],'body-400')
        saved=json.dumps(diagnostic)
        for secret in ('secret-api-key','secret-token','QUJDRA=='):
            self.assertNotIn(secret,saved)

    def test_http_400_redacts_nested_base64_and_extended_credential_fields(self):
        body = {'error': 'invalid image', 'request_id': 'nested-400', 'echo': {
            'base64': 'QUJDRA==', 'b64': 'RUZHSA==', 'imageData': 'SUprTA==',
            'X-API-Key': 'echo-api-key', 'refresh_token': 'echo-refresh-token',
            'api_token': 'echo-api-token', 'API-Secret': 'echo-api-secret',
            'client_secret': 'echo-client-secret', 'password': 'echo-password',
            'Proxy-Authorization': 'echo-proxy-auth', 'Ocp-Apim-Subscription-Key': 'echo-subscription-key',
        }}
        response = SimpleNamespace(ok=False, status_code=400, text=json.dumps(body),
                                   headers={}, json=lambda: body)
        with patch.dict('os.environ', {'SEEGROUND_MODEL': 'mock',
                'SEEGROUND_BASE_URL': 'https://example.invalid/v1',
                'SEEGROUND_API_KEY': 'test-only'}), \
             patch('qwen_api.requests.post', return_value=response) as post:
            with self.assertRaises(QwenAPIError) as raised:
                chat([])
        self.assertEqual(post.call_count, 1)
        diagnostic = raised.exception.diagnostic
        self.assertEqual(diagnostic['request_id'], 'nested-400')
        saved = json.dumps(diagnostic)
        for sensitive in body['echo'].values():
            self.assertNotIn(sensitive, saved)
        self.assertIn('invalid image', diagnostic['response_body'])

    def test_redact_general_image_data_uri_and_long_base64_body(self):
        encoded = 'QUJD' * 80
        folded = '\n'.join(encoded[start:start + 80] for start in range(0, len(encoded), 80))
        short_tail = '\n'.join(('QUJD' * 70)[start:start + 80]
                               for start in range(0, 280, 80))
        cases = [
            'invalid image: data:image/png;charset=utf-8;base64,QUJD\nRA==',
            'invalid image body: ' + encoded,
            'invalid folded image body:\n' + folded,
            'invalid image body with short final line:\n' + short_tail,
        ]
        for body in cases:
            with self.subTest(body=body[:32]):
                saved = _redact(body, limit=64)
                self.assertIn('invalid', saved)
                self.assertIn('[IMAGE_REDACTED]', saved)
                self.assertNotIn('QUJD', saved)
                self.assertNotIn('RA==', saved)
                self.assertNotIn('data:image', saved)

    def test_redact_extended_credential_fields_in_plain_error_body(self):
        body = ('invalid credentials: X-API-Key="echo-api-key" '
                'refresh_token=echo-refresh-token client_secret=echo-client-secret '
                'api_token=echo-api-token api_secret=echo-api-secret '
                'password=echo-password Proxy-Authorization=echo-proxy-auth '
                'subscription_key=echo-subscription-key')
        saved = _redact(body)
        self.assertIn('invalid credentials', saved)
        for sensitive in ('echo-api-key', 'echo-refresh-token', 'echo-client-secret',
                          'echo-api-token', 'echo-api-secret', 'echo-password',
                          'echo-proxy-auth', 'echo-subscription-key'):
            self.assertNotIn(sensitive, saved)

    def test_qwen_react_schema_retries_invalid_action_without_inventing_evidence(self):
        schema = obj({'action': enum('inspect', 'measure', 'finish'), 'reason': {'type': 'string'},
            'focus_ids': array({'type': 'integer', 'enum': [1, 2]}),
            'predicted_id': {'type': ['integer', 'null'], 'enum': [1, 2, None]},
            'evidence': array(obj({'object_ids': array({'type': 'integer'})}))})
        valid = {'action': 'inspect', 'reason': 'Compare the chairs',
                 'focus_ids': [1, 2], 'predicted_id': None, 'evidence': []}
        invalid = [dict(valid, focus_ids=[]), dict(valid, focus_ids=[99]),
                   dict(valid, action='finish', focus_ids=[], predicted_id=1)]
        def response(value):
            return SimpleNamespace(ok=True, status_code=200, text=json.dumps(value), headers={},
                json=lambda: {'choices': [{'finish_reason': 'stop',
                    'message': {'content': json.dumps(value)}}]})
        for bad in invalid:
            with self.subTest(action=bad), patch.dict('os.environ', {
                    'SEEGROUND_MODEL': 'qwen3-vl-plus', 'SEEGROUND_V4': '1',
                    'SEEGROUND_V7': '1', 'SEEGROUND_ENABLE_THINKING': '0',
                    'SEEGROUND_BASE_URL': 'https://example.invalid/v1',
                    'SEEGROUND_API_KEY': 'test-only'}), patch('qwen_api.requests.post',
                    side_effect=[response(bad), response(valid)]) as post:
                contract = schema_kwargs('r18_react_action', schema)
                self.assertIn('response_schema', contract)
                result = chat([{'role': 'user', 'content': 'Compare chair instances'}],
                              json_mode=True, **contract)
                self.assertEqual(json.loads(result['_content']), valid)
                self.assertEqual(post.call_count, 2)
                self.assertEqual(result['_baseline']['schema_retry'], 1)
                body = post.call_args.kwargs['json']
                self.assertEqual(body['response_format'], {'type': 'json_object'})
                self.assertNotIn('reasoning_effort', body)
                self.assertIn('previous response failed validation', body['messages'][-1]['content'])
                self.assertTrue(api_config_summary()['local_schema_validation'])

    def test_qwen_schema_failure_stops_after_four_attempts(self):
        value = {'action': 'unavailable'}
        response = SimpleNamespace(ok=True, status_code=200, text=json.dumps(value), headers={},
            json=lambda: {'choices': [{'finish_reason': 'stop',
                                      'message': {'content': json.dumps(value)}}]})
        with patch.dict('os.environ', {'SEEGROUND_MODEL': 'qwen3-vl-plus',
                'SEEGROUND_ENABLE_THINKING': '0',
                'SEEGROUND_BASE_URL': 'https://example.invalid/v1',
                'SEEGROUND_API_KEY': 'test-only'}), \
                patch('qwen_api.requests.post', return_value=response) as post:
            with self.assertRaises(QwenAPIError) as raised:
                chat([], json_mode=True, response_schema={'name': 'r18_react_action',
                    'strict': True, 'schema': obj({'action': enum('finish')})})
        self.assertEqual(post.call_count, 4)
        self.assertEqual(raised.exception.diagnostic['failure_kind'], 'response_schema')


if __name__=='__main__':unittest.main()
