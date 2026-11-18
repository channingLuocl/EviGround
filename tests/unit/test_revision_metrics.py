"""Revision rates use actual state transitions and independent evaluator outcomes."""

from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from eviground.evaluation.costs import revision_metrics, sample_cost, summarize_costs
from scripts.summarize_inference_costs import export


def event(statuses, invalidated=(), restored=()):
    return dict(plan=dict(nodes=[dict(id=node_id, status=status) for node_id, status in statuses.items()]),
                invalidated_node_ids=list(invalidated), reactivated_target_ids=list(restored))


class RevisionMetricsTests(unittest.TestCase):
    def trace(self):
        return dict(dag_monitor_history=[
            event(dict(comparison='completed', verification='pending')),
            event(dict(comparison='pending', verification='pending'),
                  invalidated=['comparison', 'verification'], restored=[2]),
        ])

    def test_actual_rollback_counts_even_when_its_action_was_rejected(self):
        trace = self.trace()
        trace['dag_monitor_history'][-1]['action_error'] = 'rollback accepted; action not ready'
        original = deepcopy(trace)
        self.assertEqual(revision_metrics(trace), dict(
            dag_reopened=True, candidates_restored=True, revision_triggered=True))
        self.assertEqual(trace, original)

    def test_pending_requests_and_deleted_nodes_are_not_reopened_tasks(self):
        history = [event(dict(done='completed', pending='pending')),
                   event(dict(pending='pending'), invalidated=['done', 'pending'])]
        self.assertEqual(revision_metrics(dict(dag_monitor_history=history)), dict(
            dag_reopened=False, candidates_restored=False, revision_triggered=False))
        history[-1]['reactivated_target_ids'] = [7]
        flags = revision_metrics(dict(dag_monitor_history=history))
        self.assertFalse(flags['dag_reopened'])
        self.assertTrue(flags['candidates_restored'])
        self.assertTrue(flags['revision_triggered'])

    def test_missing_histories_are_unknown_but_empty_histories_are_observed(self):
        self.assertEqual(set(revision_metrics({}).values()), {None})
        self.assertEqual(set(revision_metrics(dict(dag_monitor_history=[])).values()), {False})
        summary = summarize_costs([dict(cost=sample_cost({})), dict(cost=None)], 4)
        value = summary['judgment_revision']['revision_triggered']
        self.assertEqual(value['missing_history_samples'], 2)
        self.assertIsNone(value['trigger_rate'])
        self.assertIsNone(value['correction_success_rate'])

    def test_replayed_and_previous_events_count_as_one_sample(self):
        first = sample_cost(dict(active_trace=self.trace()))
        resumed = sample_cost(dict(active_trace=self.trace(), previous_execution_costs=[first]))
        value = summarize_costs([dict(cost=resumed, correct=True)], 5)['judgment_revision']['revision_triggered']
        self.assertEqual(value['triggered_samples'], 1)
        self.assertEqual(value['trigger_rate'], 1)
        self.assertEqual(value['correction_success_rate'], 1)
        self.assertTrue(revision_metrics({}, [first])['revision_triggered'])

    def test_repeated_events_and_overlapping_trigger_types_do_not_duplicate_samples(self):
        trace = self.trace()
        trace['dag_monitor_history'] += deepcopy(trace['dag_monitor_history'])
        both = sample_cost(dict(active_trace=trace))
        rows = [dict(cost=both, correct=False),
                dict(cost=dict(dag_reopened=True, candidates_restored=False, revision_triggered=True), correct=True),
                dict(cost=dict(dag_reopened=False, candidates_restored=True, revision_triggered=True), correct=True),
                dict(cost=revision_metrics(dict(dag_monitor_history=[])), correct=True)]
        summary = summarize_costs(rows, 10)['judgment_revision']
        self.assertEqual(summary['dag_reopened']['triggered_samples'], 2)
        self.assertEqual(summary['candidates_restored']['triggered_samples'], 2)
        self.assertEqual(summary['revision_triggered']['triggered_samples'], 3)
        self.assertEqual(summary['revision_triggered']['trigger_rate'], .75)
        self.assertEqual(summary['revision_triggered']['correction_success_rate'], 2 / 3)

    def test_errors_abstentions_and_unknown_outcomes_are_not_successes(self):
        cost = sample_cost(dict(active_trace=self.trace()))
        rows = [dict(cost=cost, correct=True), dict(cost=cost, correct=True, error='worker failed'),
                dict(cost=cost, correct=False, predicted_id=None), dict(cost=cost),
                dict(cost=revision_metrics(dict(dag_monitor_history=[])), correct=True), dict(cost=None)]
        value = summarize_costs(rows, 10)['judgment_revision']['revision_triggered']
        self.assertEqual(value['observed_samples'], 5)
        self.assertEqual(value['missing_history_samples'], 1)
        self.assertEqual(value['trigger_rate'], 4 / 5)
        self.assertEqual(value['evaluated_triggered_samples'], 3)
        self.assertEqual(value['unknown_outcome_samples'], 1)
        self.assertEqual(value['correct_after_trigger'], 1)
        self.assertEqual(value['correction_success_rate'], 1 / 3)

    def test_export_joins_evaluator_outcomes_and_writes_json_csv_markdown(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            selected = [dict(csv_row=i, scene='synthetic', query=f'query {i}') for i in range(4)]
            (root / 'manifest.json').write_text(json.dumps(dict(rows=selected)))
            outcomes = [dict(selected[0], correct=True),
                        dict(selected[1], correct=False, error='worker failed')]
            (root / 'results.json').write_text(json.dumps(dict(results=outcomes)))
            for row in selected[:3]:
                sample = root / f"q{row['csv_row']:05d}"
                sample.mkdir()
                (sample / 'run.json').write_text(json.dumps(dict(query=row['query'], active_trace=self.trace())))
            summary = export(root)
            value = summary['judgment_revision']['revision_triggered']
            self.assertEqual((summary['processed'], summary['pending']), (3, 1))
            self.assertEqual(value['triggered_samples'], 3)
            self.assertEqual(value['correction_success_rate'], .5)
            self.assertEqual(value['unknown_outcome_samples'], 1)
            self.assertEqual(json.loads((root / 'cost_summary.json').read_text()), summary)
            with (root / 'cost_samples.csv').open() as stream:
                exported = list(csv.DictReader(stream))
            self.assertEqual([r['correct'] for r in exported], ['True', 'False', ''])
            self.assertEqual([r['revision_triggered'] for r in exported], ['True'] * 3)
            self.assertIn('1/2 (50.00%)', (root / 'cost_summary.md').read_text())
            for invalid in ([outcomes[0], outcomes[0]], [dict(outcomes[0], query='wrong query')]):
                (root / 'results.json').write_text(json.dumps(dict(results=invalid)))
                with self.assertRaises(ValueError):
                    export(root)
            (root / 'results.json').unlink()
            value = export(root)['judgment_revision']['revision_triggered']
            self.assertEqual(value['unknown_outcome_samples'], 3)
            self.assertIsNone(value['correction_success_rate'])


if __name__ == '__main__':
    unittest.main()
