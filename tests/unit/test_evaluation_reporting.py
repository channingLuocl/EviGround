"""Progress groups keep their frozen benchmark denominators, including failures."""
import json
from pathlib import Path
import tempfile
import unittest

from eviground.evaluation.reporting import report


class EvaluationReportingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.output = Path(temp.name)
        self.selection = [dict(csv_row=i, easy=i < 3, view_dep=i % 2 == 0,
                               unique=i < 2) for i in range(6)]
        self.results = [
            dict(self.selection[0], predicted_id=7, correct=True, iou_3d=0.5),
            dict(self.selection[2], predicted_id=None, correct=False, iou_3d=1.0),
            dict(self.selection[3], predicted_id=None, correct=False, iou_3d=1.0,
                 error='worker failed'),
        ]

    def test_partial_groups_use_fixed_counts_and_report_all_statuses(self):
        summary = report(self.output, self.results, 6, selection=self.selection)
        groups = json.loads((self.output / 'results.json').read_text())['groups']
        self.assertEqual(summary['accuracy'], 1 / 6)
        self.assertEqual(summary['completion_ratio'], 0.5)
        self.assertEqual((summary['answered'], summary['abstained'], summary['errors']), (1, 1, 1))
        self.assertEqual(groups['easy']['True']['selected_total'], 3)
        self.assertEqual(groups['easy']['True']['accuracy'], 1 / 3)
        unique, multiple = groups['unique']['True'], groups['unique']['False']
        self.assertEqual((unique['selected_total'], unique['processed'], unique['acc50']), (2, 1, 0.5))
        self.assertEqual((multiple['selected_total'], multiple['processed'], multiple['acc25']), (4, 2, 0.0))
        self.assertEqual((multiple['abstained'], multiple['errors']), (1, 1))
        self.assertFalse(summary['complete'])

    def test_not_started_groups_are_present_with_their_full_denominators(self):
        report(self.output, [], 6, selection=self.selection)
        data = json.loads((self.output / 'results.json').read_text())
        self.assertEqual(data['groups']['unique']['True']['selected_total'], 2)
        self.assertEqual(data['groups']['unique']['False']['selected_total'], 4)
        self.assertEqual(data['summary']['completion_ratio'], 0.0)
        self.assertIsNone(data['summary']['accuracy'])
        self.assertIsNone(data['summary']['acc25'])
        self.assertIsNone(data['summary']['acc50'])
        self.assertIsNone(data['groups']['unique']['True']['accuracy'])

    def test_revision_rates_reach_progress_and_cost_reports(self):
        flags = dict(dag_reopened=True, candidates_restored=True, revision_triggered=True)
        self.results[0]['cost'] = flags
        self.results[1]['cost'] = flags
        summary = report(self.output, self.results, 6, selection=self.selection)
        cost = json.loads((self.output / 'cost_summary.json').read_text())
        self.assertEqual(summary['judgment_revision'], cost['judgment_revision'])
        value = summary['judgment_revision']['revision_triggered']
        self.assertEqual(value['triggered_samples'], 2)
        self.assertEqual(value['missing_history_samples'], 1)
        self.assertEqual(value['correction_success_rate'], .5)

    def test_unselected_results_and_changed_group_labels_are_rejected(self):
        for changed in (dict(self.results[0], csv_row=99),
                        dict(self.results[0], unique=False)):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, 'selection'):
                report(self.output, [changed], 6, selection=self.selection)

    def test_selection_size_and_duplicate_ids_are_rejected(self):
        for selected in (self.selection[:-1], [self.selection[0]] * 6):
            with self.assertRaisesRegex(ValueError, 'selection'):
                report(self.output, [], 6, selection=selected)


if __name__ == '__main__':
    unittest.main()
