import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from eviground.experiments.run_module_acceptance import (
    MODULES, bind_output, case_modules, execution_audit, parsed_dimensions,
    final_audit_missing, preflight, report,
)


class ModuleAcceptanceTests(unittest.TestCase):
    def test_case_modules_requires_known_dimension(self):
        self.assertEqual(case_modules({'module': 'size'}), ['size'])
        self.assertEqual(case_modules({'modules': ['position', 'viewpoint']}),
                         ['position', 'viewpoint'])
        with self.assertRaises(ValueError):
            case_modules({'module': 'not-a-module'})

    def test_parsed_dimensions_keeps_size_position_and_viewpoint_separate(self):
        parsed = {
            'Attributes': [{'category': 'cabinet', 'phrase': 'tall'}],
            'Relations': [{'target_category': 'chair', 'anchor_category': 'table',
                           'relation': 'above', 'phrase': 'above the table'},
                          {'target_category': 'chair', 'anchor_category': 'door',
                           'relation': 'nearest', 'phrase': 'nearest the door'},
                          {'target_category': 'chair', 'anchor_category': 'window',
                           'relation': 'left', 'phrase': 'left of the window'}],
            'Viewpoint': 'facing the window',
        }
        self.assertEqual(parsed_dimensions(parsed), ['appearance', 'distance', 'position', 'viewpoint'])

    def test_execution_audit_distinguishes_precompute_from_tool_request(self):
        parsed = {'Relations': [{'relation': 'above'}], 'Attributes': []}
        diagnosis = {'geometry': {'relation_pairs': [
            {'target_id': 2, 'passes': True, 'relation': 'above'}]},
            'missing': [], 'pending_size_conditions': [], 'visual_needs': []}
        trace = {'initial_diagnosis': diagnosis, 'steps': [
            {'action': {'action': 'evidence', 'dimension': 'position'},
             'observation': {'dimension': 'position', 'operation_status': 'evaluated',
                             'coverage': {'checked_ids': [2], 'unknown_ids': [], 'uninspected_ids': []},
                             'candidate_evidence': [], 'missing': []}},
            {'action': {'action': 'finish'}, 'observation': {'tool': 'finish'}},
        ]}
        audit = execution_audit(parsed, trace, diagnosis)
        self.assertTrue(audit['position']['tool_requested'])
        self.assertTrue(audit['position']['tool_executed'])
        self.assertTrue(audit['position']['precomputed_evidence'])
        self.assertFalse(audit['size']['tool_executed'])

    def test_rejected_request_is_not_executed_and_near_is_distance_evidence(self):
        parsed = {'Relations': [{'relation': 'near'}], 'Attributes': []}
        diagnosis = {'geometry': {'proximity_pairs': [{'target_id': 2, 'distance_m': .2}]}}
        trace = {'initial_diagnosis': diagnosis, 'steps': [
            {'action': {'action': 'evidence', 'dimension': 'appearance'},
             'observation': {'no_evidence_acquired': True, 'error': 'invalid focus'}},
        ]}
        audit = execution_audit(parsed, trace, diagnosis)
        self.assertTrue(audit['appearance']['tool_requested'])
        self.assertFalse(audit['appearance']['tool_executed'])
        self.assertTrue(audit['distance']['exercised'])
        self.assertFalse(audit['position']['exercised'])

    def test_preflight_records_missing_optional_structure_without_secret(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene = root / 'scenes/scene0000_00'
            scene.mkdir(parents=True)
            (root / 'local').mkdir()
            (scene / 'scene.npz').write_bytes(b'scene')
            (scene / 'provenance.json').write_text('{}')
            (scene / 'objects_gt.json').write_text('[{"bbox_id": 2}]')
            manifest = {'csv_sha256': 'expected', 'rows': [{'module': 'size',
                'csv_row': 1, 'scene': scene.name, 'query': 'small table', 'target_id': 2,
                'stimulus_candidate_ids': [2]}]}
            with patch('eviground.experiments.run_module_acceptance.ROOT', root), \
                    patch('eviground.experiments.run_module_acceptance._settings', return_value=('https://example.test', 'model', 'secret')), \
                    patch('eviground.experiments.run_module_acceptance.validate_react_config', return_value='qwen'):
                settings = preflight(manifest, root / 'scenes', 5)
            self.assertIsNone(settings['input_sha256'][scene.name]['scene_instances.json'])
            self.assertNotIn('secret', json.dumps(settings))
            bind_output(root / 'out', settings)
            bind_output(root / 'out', settings)
            changed = {**settings, 'review_limit': 6}
            with self.assertRaisesRegex(ValueError, 'different code/settings'):
                bind_output(root / 'out', changed)

    def test_report_counts_acceptance_and_dimension_execution(self):
        row = {'module': 'size', 'csv_row': 1, 'scene': 's', 'query': 'small table',
               'target_id': 2, 'accepted': True, 'correct': False,
               'decision_status': 'resolved', 'api_calls': 3,
               'module_audit': {'size': {'parsed': True, 'tool_executed': True,
                                         'exercised': True}},
               'recorded_api_calls': 3, 'api_stages': {'parse_query': 1}}
        with tempfile.TemporaryDirectory() as directory:
            summary = report(Path(directory), [row], [row])
            saved = json.loads((Path(directory) / 'results.json').read_text())
        self.assertEqual(summary['accepted_wrong'], 1)
        self.assertEqual(saved['modules']['size']['expected_total'], 1)
        self.assertEqual(saved['modules']['size']['tool_executed_cases'], 1)
        self.assertEqual(set(saved['modules']), set(MODULES))

    def test_final_audit_missing_reports_unresolved_ambiguity(self):
        trace = {'final_evidence_state': {
            'status': 'ambiguous', 'missing': [],
            'remaining_candidate_ids': [8, 13]}}
        diagnosis = {'status': 'ambiguous', 'missing': [], 'candidate_ids': [8, 13]}
        self.assertEqual(
            final_audit_missing(trace, diagnosis),
            ['multiple candidates remain under unresolved evidence: [8, 13]'])

    def test_final_audit_missing_reports_conflicting_singleton_hypotheses(self):
        trace = {'final_evidence_state': {
            'status': 'ambiguous', 'missing': [], 'remaining_candidate_ids': [20]}}
        diagnosis = {'status': 'ambiguous', 'missing': [], 'candidate_ids': [20],
                     'hypotheses': [{'candidate_ids': [20]}, {'candidate_ids': []}]}
        self.assertEqual(final_audit_missing(trace, diagnosis),
            ['remaining candidate lacks joint support across legal hypotheses: [20]'])


if __name__ == '__main__':
    unittest.main()
