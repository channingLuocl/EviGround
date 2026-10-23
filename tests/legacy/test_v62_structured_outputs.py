"""Offline v6.2 regressions for strict transport and existing controller semantics."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from jsonschema import Draft202012Validator, ValidationError
import requests

from seeground.api.qwen_api import QwenAPIError, chat, api_config_summary
from seeground.grounding.real_scene import parse_query, repair_missing_attributes
from seeground.api.response_schemas import (APPEARANCE, ATTRIBUTES, CATEGORY_REVIEW, INSTANCE_REVIEW,
    RGB_REVIEW, VISUAL_REPORT, obj, parser_schema, planner_schema, schema_kwargs)
from seeground.grounding.v4_policy import settings, v6_enabled
from seeground.grounding.v5_runtime import sample_recovery


def response(content, finish='stop', refusal=None):
    body = {'choices': [{'message': {'content': content, 'refusal': refusal},
                         'finish_reason': finish}]}
    return SimpleNamespace(ok=True, status_code=200, headers={'x-request-id': 'strict-test'},
                           text=json.dumps(body), json=lambda: body)


def parser_answer():
    return {'Target': ['chair'], 'Anchor': [], 'Relations': [], 'Viewpoint': '',
        'Observer': {'origin': {'kind': 'unspecified', 'category': '', 'phrase': ''},
                     'facing': {'category': '', 'group': False, 'phrase': ''}},
        'Attributes': [], 'Unhandled': []}


def planner_answer(**values):
    answer = {key: None for key in planner_schema([1, 2], [1, 2])['properties']}
    answer.update(action='stop', reason='Evidence does not distinguish candidates.')
    answer.update(values)
    return answer


class StructuredOutputsTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict('os.environ', {'SEEGROUND_V4': '1', 'SEEGROUND_V6': '0',
            'SEEGROUND_V62': '1', 'SEEGROUND_API_PROVIDER': 'openai',
            'SEEGROUND_REACT_BACKEND': 'openai', 'SEEGROUND_MODEL': 'mock',
            'SEEGROUND_BASE_URL': 'https://relay.invalid/v1', 'SEEGROUND_API_KEY': 'mock-key',
            'SEEGROUND_OPENAI_TOKEN_PARAMETER': 'max_completion_tokens',
            'SEEGROUND_OPENAI_TEMPERATURE': '', 'SEEGROUND_SYNTAX_BACKEND': 'none'})
        env.start()
        self.addCleanup(env.stop)
        token = sample_recovery.set({'used': False, 'attempts': []})
        self.addCleanup(sample_recovery.reset, token)

    def test_v62_opt_in_enables_v6_and_records_version(self):
        self.assertTrue(v6_enabled())
        self.assertEqual(settings()['version'], 'relative-size-v6.2')
        self.assertTrue(api_config_summary()['structured_outputs'])
        self.assertEqual(api_config_summary()['request_options']['response_format'], {'type': 'json_schema'})
        with patch.dict('os.environ', {'SEEGROUND_V62': '0', 'SEEGROUND_V6': '1'}):
            self.assertEqual(settings()['version'], 'relative-size-v6')
            self.assertEqual(schema_kwargs('parser', parser_schema('diagnostic')), {})
        with patch.dict('os.environ', {'SEEGROUND_API_PROVIDER': 'qwen'}):
            self.assertEqual(schema_kwargs('parser', parser_schema('diagnostic')), {})

    def test_all_contracts_close_objects_and_require_every_field(self):
        def inspect(node):
            if isinstance(node, dict):
                if node.get('type') == 'object':
                    self.assertFalse(node['additionalProperties'])
                    self.assertEqual(set(node['required']), set(node['properties']))
                for value in node.values():
                    inspect(value)
            elif isinstance(node, list):
                for value in node:
                    inspect(value)
        for schema in (parser_schema('baseline'), parser_schema('diagnostic'), ATTRIBUTES,
                       planner_schema([1, 2], [1]), APPEARANCE, VISUAL_REPORT,
                       RGB_REVIEW, CATEGORY_REVIEW, INSTANCE_REVIEW):
            Draft202012Validator.check_schema(schema)
            inspect(schema)

    def test_real_parser_sends_strict_schema_and_records_actual_format(self):
        calls = []
        with patch('seeground.api.qwen_api.requests.post', return_value=response(json.dumps(parser_answer()))) as post:
            parsed = parse_query('the chair', mode='active_diagnostic', calls=calls)
        payload = post.call_args.kwargs['json']
        self.assertEqual(payload['response_format']['type'], 'json_schema')
        self.assertTrue(payload['response_format']['json_schema']['strict'])
        self.assertEqual(payload['response_format']['json_schema']['schema'], parser_schema('diagnostic'))
        self.assertEqual(payload['max_completion_tokens'], 1600)
        self.assertEqual(parsed['Target'], ['chair'])
        self.assertEqual(calls[0]['response_format'], payload['response_format'])

    def test_invalid_schema_output_is_rejected_once_without_fallback(self):
        good = parser_answer()
        missing = deepcopy(good); missing.pop('Target')
        wrong_type = deepcopy(good); wrong_type['Target'] = 'chair'
        extra = deepcopy(good); extra['invented'] = True
        bad_enum = deepcopy(good); bad_enum['Observer']['origin']['kind'] = 'guessed'
        for content in ('not JSON', "'''json\n" + json.dumps(good) + "\n'''",
                        json.dumps(missing), json.dumps(wrong_type), json.dumps(extra), json.dumps(bad_enum)):
            with self.subTest(content=content), patch('seeground.api.qwen_api.requests.post',
                    return_value=response(content)) as post:
                with self.assertRaises(QwenAPIError) as error:
                    chat([], json_mode=True, **schema_kwargs('query_parser', parser_schema('diagnostic')))
                self.assertEqual(post.call_count, 1)
                self.assertEqual(error.exception.diagnostic['failure_kind'], 'response_schema')
                self.assertEqual(error.exception.diagnostic['schema_name'], 'query_parser')
                self.assertEqual(error.exception.diagnostic['request_id'], 'strict-test')
                self.assertNotIn('mock-key', json.dumps(error.exception.diagnostic))

    def test_refusal_and_truncation_fail_before_a_partial_action_can_execute(self):
        for finish, refusal in (('length', None), ('stop', 'cannot answer')):
            with self.subTest(finish=finish), patch('seeground.api.qwen_api.requests.post', return_value=response(
                    json.dumps(parser_answer()), finish=finish, refusal=refusal)):
                with self.assertRaises(QwenAPIError) as error:
                    chat([], json_mode=True, **schema_kwargs('query_parser', parser_schema('diagnostic')))
                self.assertEqual(error.exception.diagnostic['failure_kind'], 'response_schema')

    def test_unsupported_schema_is_not_silently_downgraded(self):
        rejected = SimpleNamespace(ok=False, status_code=400, text='schema unsupported', headers={})
        with patch('seeground.api.qwen_api.requests.post', return_value=rejected) as post:
            with self.assertRaises(QwenAPIError):
                chat([], json_mode=True, **schema_kwargs('query_parser', parser_schema('diagnostic')))
        self.assertEqual(post.call_count, 1)
        self.assertEqual(post.call_args.kwargs['json']['response_format']['type'], 'json_schema')

    def test_transient_retry_preserves_the_strict_schema(self):
        with patch('seeground.api.qwen_api.requests.post', side_effect=[requests.Timeout(),
                response(json.dumps(parser_answer()))]) as post:
            chat([], json_mode=True, **schema_kwargs('query_parser', parser_schema('diagnostic')))
        self.assertEqual(post.call_count, 2)
        formats = [call.kwargs['json']['response_format'] for call in post.call_args_list]
        self.assertEqual(formats[0], formats[1])
        self.assertEqual(formats[0]['type'], 'json_schema')

    def test_recovery_cache_identity_includes_the_schema(self):
        budget = sample_recovery.get()
        with tempfile.TemporaryDirectory() as td:
            budget.update(api_checkpoint=str(Path(td) / 'api.json'))
            with patch('seeground.api.qwen_api.requests.post', return_value=response('{"ok":true}')) as post:
                chat([], json_mode=True, **schema_kwargs('first', obj({'ok': {'type': 'boolean'}})))
                budget['resuming'] = True
                chat([], json_mode=True, **schema_kwargs('second', obj({'ok': {'type': 'boolean'}})))
            self.assertEqual(post.call_count, 2)
            self.assertEqual(len(budget['completed_requests']), 2)

    def test_planner_rejects_unknown_ids_invalid_angles_and_blank_reason(self):
        validator = Draft202012Validator(planner_schema([1, 2, 3], [1, 2]))
        validator.validate(planner_answer(action='evidence', dimension='appearance'))
        for change in ({'reason': ''}, {'reason': '   '}, {'predicted_id': 3},
                       {'focus_ids': [99]}, {'azimuth_deg': 360}, {'action': 'guess'}):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                validator.validate(planner_answer(**change))

    def test_null_planner_options_preserve_existing_appearance_defaults(self):
        from tests.legacy.test_active_diagnostic import ActiveDiagnosticTests
        from seeground.grounding.active_diagnostic import compute_observation
        from seeground.grounding.diagnostic import validate_structure
        seen = []
        def compute(dimension, *args, **kwargs):
            seen.append(dimension)
            return compute_observation(dimension, *args, **kwargs)
        actions = [planner_answer(action='evidence', dimension='appearance'), planner_answer()]
        checks = {'checks': [{'id': oid, 'category': 'window', 'role': 'target',
            'category_verdict': 'yes', 'verdict': 'yes', 'visibility': 'visible',
            'reason': 'Window visible.', 'attribute_checks': []} for oid in (1, 2)]}
        parsed = parser_answer(); parsed['Target'] = ['window']
        with patch('test_active_diagnostic.parsed', return_value=validate_structure('window', parsed)):
            result, _ = ActiveDiagnosticTests().run_actions(actions, visual=[checks], compute=compute)
        self.assertEqual(seen, ['appearance'])
        self.assertEqual(result['predicted_id'], None)
        self.assertEqual(result['implementation_version'], 'relative-size-v6.2')

    def test_attribute_repair_uses_its_own_contract(self):
        parsed = parser_answer()
        parsed['diagnostic_missing'] = ['appearance condition not represented: red']
        calls = []
        answer = {'Attributes': [{'category': 'chair', 'role': 'target', 'phrase': 'red'}]}
        with patch('seeground.api.qwen_api.requests.post', return_value=response(json.dumps(answer))) as post:
            result = repair_missing_attributes('the red chair', parsed, calls)
        self.assertEqual(post.call_args.kwargs['json']['response_format']['json_schema']['name'], 'query_attributes')
        self.assertEqual(result['Attributes'][0]['phrase'], 'red')

    def test_visual_report_uses_strict_contract_without_certifying_unknown(self):
        from seeground.rendering.v4_visual import visual_report
        answer = {key: [] for key in ('observations', 'relations', 'metric_choices', 'contradictions', 'findings')}
        answer.update(status='unknown', reason='Insufficient visible evidence.')
        calls = []
        with patch('seeground.rendering.v4_visual.image_part', return_value={'type': 'text', 'text': 'image'}), \
                patch('seeground.api.qwen_api.requests.post', return_value=response(json.dumps(answer))) as post:
            result = visual_report('final_prediction_review', 'chair',
                {'image': 'mock.png', 'focus_ids': [1], 'visible_instance_pixels': {'1': 100}}, {}, calls)
        self.assertEqual(post.call_args.kwargs['json']['response_format']['json_schema']['schema'], VISUAL_REPORT)
        self.assertEqual(result['status'], 'unknown')
        self.assertFalse(result['supported'])
        self.assertEqual(calls[0]['response_format']['type'], 'json_schema')
