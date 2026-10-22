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


if __name__ == '__main__':
    unittest.main()
