import json
from pathlib import Path
import tempfile
import unittest

from r8_challenge_results import collect


class CumulativeTests(unittest.TestCase):
    def make_run(self, directory, row, predicted, **manifest_changes):
        directory.mkdir()
        manifest = {'version': 'predicted-category-grounding-v9-r8-r3',
                    'categories': {'source_predictions_sha256': 'same-export'}, 'rows': [row], **manifest_changes}
        (directory/'manifest.json').write_text(json.dumps(manifest))
        case = directory/'q00001'
        case.mkdir()
        (case/'evaluation.json').write_text(json.dumps({**row, 'predicted_id': predicted,
            'correct': predicted == row['target_id'], 'api_attempts': 2}))
        return directory

    def test_retains_correct_history_and_counts_all_attempts_with_full_denominator(self):
        row = {'csv_row': 1, 'scene': 's', 'query': 'chair', 'target_id': 7, 'easy': True, 'view_dep': False}
        selection = {'rows': [row, {**row, 'csv_row': 2}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = [self.make_run(root/str(i), row, oid) for i, oid in enumerate([2, 7, 3])]
            report = collect(selection, runs)
        self.assertEqual(report['summary']['correct'], 1)
        self.assertEqual(report['summary']['accuracy_over_selected'], .5)
        self.assertEqual(report['summary']['total_api_attempts'], 6)
        self.assertFalse(report['summary']['complete'])
        self.assertFalse(report['summary']['deployment_prediction'])
        self.assertEqual(report['results'][0]['predicted_id'], 7)
        self.assertEqual(report['summary']['unreported_token_usage_attempts'], 6)

    def test_failure_usage_is_reported_as_missing_instead_of_free(self):
        row = {'csv_row': 1, 'scene': 's', 'query': 'chair', 'target_id': 7}
        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run(Path(directory)/'run', row, 7)
            path = run/'q00001/evaluation.json'
            value = json.loads(path.read_text())
            value['call_usage'] = [{'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12}, None]
            path.write_text(json.dumps(value))
            report = collect({'rows': [row]}, [run])
        source = report['single_pass_runs'][0]
        self.assertEqual(source['tokens']['total_tokens'], 12)
        self.assertTrue(source['token_totals_are_lower_bounds'])
        self.assertEqual(source['unreported_token_usage_attempts'], 1)

    def test_rejects_gt_runs_answer_mismatch_and_export_changes(self):
        row = {'csv_row': 1, 'scene': 's', 'query': 'chair', 'target_id': 7}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = self.make_run(root/'good', row, 7)
            bad_gt = self.make_run(root/'gt', row, 7, version='relative-size-v7.4')
            bad_answer = self.make_run(root/'answer', {**row, 'target_id': 8}, 7)
            bad_export = self.make_run(root/'export', row, 7, categories={'source_predictions_sha256': 'different'})
            for runs in ([good, bad_gt], [good, bad_answer], [good, bad_export], [good, good]):
                with self.subTest(runs=runs), self.assertRaises(ValueError):
                    collect({'rows': [row]}, runs)


if __name__ == '__main__':
    unittest.main()
