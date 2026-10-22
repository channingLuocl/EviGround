import json
from pathlib import Path
import tempfile
import unittest

from audit_single_pass import audit_run


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = [{'csv_row': 1, 'target_id': 10}, {'csv_row': 2, 'target_id': 20}]
        manifest = {'rows': self.rows, 'code_sha256': {}, 'models': {'SEEGROUND_MODEL': 'gpt-6.1-sol'},
            'r8_grounding_revision': 'r18', 'categories': {'gate': {'top_k': 5}},
            'evaluation_policy': 'fixed_predictions_no_historical_correct_reuse',
            'api': {'provider': 'openai', 'thinking_token_floor': 8192, 'local_schema_validation': True,
                    'request_options': {'response_format': {'type': 'json_object'}}}}
        (self.root / 'manifest.json').write_text(json.dumps(manifest))

    def case(self, row, oid, failure=False):
        directory = self.root / f"q{row['csv_row']:05d}"
        directory.mkdir()
        result = {**row, 'predicted_id': oid, 'correct': oid == row['target_id']}
        if failure:
            result.update(error='worker format failure', predicted_id=None, correct=False)
        else:
            prediction = {'predicted_id': oid, 'candidate_target_ids': [10, 20],
                'gt_categories_supplied_to_model': False, 'query_target_id_supplied_to_model': False,
                'diagnosis_policy': 'predicted_category_multimodal_react',
                'active_trace': {'stop_reason': 'r18_react_finish', 'r18_final_decision': {'predicted_id': oid}}}
            (directory / 'prediction.json').write_text(json.dumps(prediction))
        (directory / 'evaluation.json').write_text(json.dumps(result))

    def audit(self, total=2):
        return audit_run(self.root, self.rows, {}, self.root, expected_total=total)

    def test_successful_subset_cannot_pass_the_complete_gate(self):
        self.case(self.rows[0], 10)
        report = self.audit()
        self.assertFalse(report['verified_complete'])
        self.assertFalse(report['threshold_passed'])
        self.assertEqual(report['full_selected_denominator_accuracy'], .5)

    def test_fatal_error_stays_in_denominator(self):
        self.case(self.rows[0], 10)
        self.case(self.rows[1], None, failure=True)
        report = self.audit()
        self.assertTrue(report['verified_complete'])
        self.assertEqual(report['fatal_errors'], 1)
        self.assertEqual(report['full_selected_denominator_accuracy'], .5)
        self.assertFalse(report['threshold_passed'])

    def test_wrong_reported_correct_flag_is_rejected(self):
        self.case(self.rows[0], 10)
        self.case(self.rows[1], 10)
        p = self.root / 'q00002/evaluation.json'
        value = json.loads(p.read_text()); value['correct'] = True; p.write_text(json.dumps(value))
        self.assertFalse(self.audit()['verified_complete'])

    def test_gt_input_flag_and_wrong_total_prevent_completion(self):
        for row in self.rows:
            self.case(row, row['target_id'])
        self.assertTrue(self.audit()['threshold_passed'])
        self.assertFalse(self.audit(7805)['threshold_passed'])
        p = self.root / 'q00002/prediction.json'
        value = json.loads(p.read_text()); value['gt_categories_supplied_to_model'] = True; p.write_text(json.dumps(value))
        self.assertFalse(self.audit()['threshold_passed'])

    def test_qwen_requires_explicit_matching_configuration_and_same_denominator(self):
        config = {'model': 'qwen3-vl-plus', 'provider': 'qwen',
                  'thinking_token_floor': None, 'token_parameter': 'max_tokens'}
        p = self.root / 'manifest.json'
        value = json.loads(p.read_text())
        value['models']['SEEGROUND_MODEL'] = config['model']
        value['api'].update(provider='qwen', token_parameter='max_tokens')
        value['api'].pop('thinking_token_floor')
        p.write_text(json.dumps(value))
        self.case(self.rows[0], 10)
        self.case(self.rows[1], None, failure=True)
        report = audit_run(self.root, self.rows, {}, self.root, expected_total=2,
                           expected_config=config)
        self.assertTrue(report['verified_complete'])
        self.assertEqual(report['full_selected_denominator_accuracy'], .5)
        self.assertFalse(report['threshold_passed'])
        self.assertFalse(self.audit()['verified_complete'])
        value['api']['token_parameter'] = 'max_completion_tokens'
        p.write_text(json.dumps(value))
        self.assertFalse(audit_run(self.root, self.rows, {}, self.root, expected_total=2,
                                  expected_config=config)['verified_complete'])

    def test_actual_call_model_mismatch_prevents_completion(self):
        for row in self.rows:
            self.case(row, row['target_id'])
        (self.root / 'q00002/run.json').write_text(json.dumps({
            'calls': [{'model': 'qwen3-vl-plus', 'provider': 'qwen'}]}))
        report = self.audit()
        self.assertFalse(report['verified_complete'])
        self.assertTrue(any('Mixed inference' in issue for issue in report['structural_issues']))

    def test_empty_category_pool_is_a_counted_abstention(self):
        self.case(self.rows[0], 10)
        self.case(self.rows[1], None)
        p = self.root / 'q00002/prediction.json'
        value = json.loads(p.read_text())
        value.pop('diagnosis_policy')
        value.update(candidate_target_ids=[], category_target_ids=[], visual_fallback_target_ids=[],
            prediction_source='abstained', bbox_3d=None,
            diagnosis={'candidate_ids': [],
                       'missing': ['no retained category hypothesis matches the parsed target']},
            active_trace={'stop_reason': 'category_gate_abstention'})
        p.write_text(json.dumps(value))
        report = self.audit()
        self.assertTrue(report['verified_complete'])
        self.assertEqual(report['category_gate_abstentions'], 1)
        self.assertEqual(report['full_selected_denominator_accuracy'], .5)
        self.assertFalse(report['threshold_passed'])
        value['candidate_target_ids'] = [10, 20]
        p.write_text(json.dumps(value))
        self.assertFalse(self.audit()['verified_complete'])

    def original_qwen_config(self):
        config = {'model': 'qwen3.7-plus', 'provider': 'openai', 'thinking_token_floor': 8192,
                  'local_schema_validation': None, 'structured_outputs': True,
                  'candidate_pool_field': 'category_target_ids',
                  'request_options': {'enable_thinking': True, 'temperature': 0.,
                                      'response_format': {'type': 'json_schema'}}}
        path = self.root / 'manifest.json'
        manifest = json.loads(path.read_text())
        manifest['models']['SEEGROUND_MODEL'] = config['model']
        manifest['api'].pop('local_schema_validation')
        manifest['api'].update(structured_outputs=True, request_options=config['request_options'])
        path.write_text(json.dumps(manifest))
        return config

    def test_original_native_schema_and_pool_keep_failures_in_denominator(self):
        config = self.original_qwen_config()
        self.case(self.rows[0], 10)
        self.case(self.rows[1], None, failure=True)
        path = self.root / 'q00001/prediction.json'
        prediction = json.loads(path.read_text())
        prediction['category_target_ids'] = prediction.pop('candidate_target_ids')
        path.write_text(json.dumps(prediction))
        report = audit_run(self.root, self.rows, {}, self.root, expected_total=2,
                           expected_config=config)
        self.assertTrue(report['verified_complete'])
        self.assertEqual(report['fatal_errors'], 1)
        self.assertEqual(report['full_selected_denominator_accuracy'], .5)
        self.assertFalse(report['threshold_passed'])
        self.assertFalse(self.audit()['verified_complete'])
        prediction['category_target_ids'] = [20]
        path.write_text(json.dumps(prediction))
        self.assertFalse(audit_run(self.root, self.rows, {}, self.root, expected_total=2,
                                  expected_config=config)['verified_complete'])

    def test_original_empty_category_pool_is_a_counted_abstention(self):
        config = self.original_qwen_config()
        self.case(self.rows[0], None)
        self.case(self.rows[1], None, failure=True)
        path = self.root / 'q00001/prediction.json'
        prediction = json.loads(path.read_text())
        prediction.pop('candidate_target_ids')
        prediction.pop('diagnosis_policy')
        prediction.update(category_target_ids=[], prediction_source='abstained', bbox_3d=None,
            diagnosis={'candidate_ids': [],
                       'missing': ['no retained category hypothesis matches the parsed target']},
            active_trace={'stop_reason': 'category_gate_abstention'})
        path.write_text(json.dumps(prediction))
        report = audit_run(self.root, self.rows, {}, self.root, expected_total=2,
                           expected_config=config)
        self.assertTrue(report['verified_complete'])
        self.assertEqual(report['category_gate_abstentions'], 1)
        self.assertEqual(report['fatal_errors'], 1)
        self.assertEqual(report['full_selected_denominator_accuracy'], 0.)


if __name__ == '__main__':
    unittest.main()
