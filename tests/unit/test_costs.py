"""Failed/replayed requests must not lose usage or become zero-cost samples."""

import unittest
import json
from pathlib import Path
import tempfile

from eviground.evaluation.costs import sample_cost, summarize_costs


class CostTests(unittest.TestCase):
    def test_resume_preserves_partial_journal_tail_before_appending(self):
        from eviground.api.usage_log import log_attempt, read_attempts
        from eviground.grounding.recovery import sample_recovery

        for tail in (b'{"local_attempt_id":', b'{"local_attempt_id":\n', b'{"x":"\xe4'):
            with self.subTest(tail=tail), tempfile.TemporaryDirectory() as root:
                path = Path(root) / 'http_attempts.jsonl'
                prefix = json.dumps(dict(local_attempt_id='a', event='started')).encode() + b'\n'
                path.write_bytes(prefix + tail)
                token = sample_recovery.set(dict(api_checkpoint=str(Path(root) / 'recovery.json')))
                try:
                    log_attempt('b', 'started')
                finally:
                    sample_recovery.reset(token)
                self.assertEqual([a['local_attempt_id'] for a in read_attempts(path)], ['a', 'b'])
                archived = list(Path(root).glob('http_attempts.jsonl.truncated-*'))
                self.assertEqual(len(archived), 1)
                self.assertEqual(archived[0].read_bytes(), tail)

    def test_complete_journal_line_without_newline_is_kept(self):
        from eviground.api.usage_log import log_attempt, read_attempts
        from eviground.grounding.recovery import sample_recovery

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'http_attempts.jsonl'
            path.write_text(json.dumps(dict(local_attempt_id='a', event='started')))
            token = sample_recovery.set(dict(api_checkpoint=str(Path(root) / 'recovery.json')))
            try:
                log_attempt('b', 'started')
            finally:
                sample_recovery.reset(token)
            self.assertEqual(len(read_attempts(path)), 2)

    def test_journal_keeps_started_only_and_truncated_finish_unknown(self):
        from eviground.api.usage_log import read_attempts

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'http_attempts.jsonl'
            events = [dict(local_attempt_id='a', event='started', stage='react_decision'),
                      dict(local_attempt_id='a', event='finished', usage=dict(total_tokens=4)),
                      dict(local_attempt_id='b', event='started', stage='react_decision')]
            path.write_text('\n'.join(map(json.dumps, events)) + '\n{"local_attempt_id":')
            attempts = read_attempts(path)
        cost = sample_cost(dict(http_attempt_records=attempts))
        self.assertEqual(cost['actual_http_attempts'], 2)
        self.assertEqual(cost['decision_http_attempts'], 2)
        self.assertEqual(cost['total_tokens_observed'], 4)
        self.assertIsNone(cost['total_tokens'])
        self.assertEqual(cost['total_tokens_missing_attempts'], 1)

    def test_failed_attempt_usage_is_counted_once_and_missing_stays_unknown(self):
        measured = dict(prompt_tokens=100, completion_tokens=20, total_tokens=120)
        attempts = [dict(local_attempt_id='a', status='failed', usage=measured),
                    dict(local_attempt_id='b', status='success', usage=measured),
                    dict(local_attempt_id='c', status='failed', usage=None)]
        run = dict(calls=[dict(stage='react_decision', api_attempts=attempts, usage=measured),
                          dict(stage='react_decision', api_attempts=[attempts[0]]),
                          dict(stage='parse_query', cache_hit=True, usage=measured)],
                   error='network failed', online_latency_s=8)
        cost = sample_cost(run)
        self.assertEqual(cost['actual_http_attempts'], 3)
        self.assertIsNone(cost['total_tokens'])
        self.assertEqual(cost['total_tokens_observed'], 240)
        self.assertEqual(cost['total_tokens_missing_attempts'], 1)
        self.assertEqual(cost['total_tokens_coverage'], 2 / 3)
        self.assertTrue(cost['error'])

    def test_tool_calls_panels_and_final_submissions_are_separate(self):
        trace = dict(react_actions=[dict(action=a) for a in ['inspect', 'measure', 'finish']],
            steps=[dict(action=dict(action='inspect'), observation=dict(tool='render')),
                   dict(action=dict(action='measure'), observation=dict(tool='monitor_check', error='pending')),
                   dict(action=dict(action='measure'), observation=dict(tool='measure', error='bad frame', no_evidence_acquired=True))],
            internal_usage=dict(render=6), rendered_panel_count=6,
            rendered_image_count=3, rendered_file_count=7,
            termination_cause='repair_exhausted', react_budget=dict(failed_checks=6))
        cost = sample_cost(dict(active_trace=trace))
        self.assertEqual(cost['inspect_successes'], 1)
        self.assertEqual(cost['measure_attempts'], 1)
        self.assertEqual(cost['measure_failures'], 1)
        self.assertEqual(cost['final_submissions'], 1)
        self.assertEqual(cost['rendered_images'], 3)
        self.assertEqual(cost['rendered_files'], 7)
        self.assertEqual(cost['rendered_panels'], 6)
        self.assertTrue(cost['budget_exhausted'])

    def test_legacy_render_work_is_not_mislabeled_as_image_count(self):
        cost = sample_cost(dict(active_trace=dict(internal_usage=dict(render=6))))
        self.assertIsNone(cost['rendered_images'])

    def test_six_corrections_can_follow_seven_rejected_replies(self):
        action = dict(action='finish')
        steps = [dict(action=action, observation=dict(tool='monitor_check', error='invalid'))] * 7
        cost = sample_cost(dict(calls=[dict(stage='react_decision')] * 7,
            active_trace=dict(steps=steps, react_actions=[action] * 7)))
        self.assertEqual(cost['failed_checks'], 7)
        self.assertEqual(cost['repair_calls'], 6)

    def test_restarts_keep_tool_work_and_unknown_interrupted_latency(self):
        trace = dict(rendered_image_count=2, rendered_file_count=2, rendered_panel_count=2,
                     steps=[dict(action=dict(action='measure'), observation=dict(tool='measure'))])
        first = sample_cost(dict(active_trace=trace, online_latency_s=3,
                                 execution_complete=False))
        second = sample_cost(dict(active_trace=trace, online_latency_s=8,
            online_latency_observed_s=8, execution_complete=True, previous_execution_costs=[first]))
        self.assertEqual(second['measure_successes'], 2)
        self.assertEqual(second['rendered_images'], 4)
        self.assertEqual(second['execution_attempts'], 2)
        self.assertIsNone(second['online_latency_s'])
        self.assertFalse(second['online_latency_complete'])
        self.assertEqual(second['online_latency_observed_s'], 8)

    def test_summary_does_not_add_unprocessed_zero_costs(self):
        rows = [dict(cost=dict(online_latency_s=10, total_tokens=100, error=False)),
                dict(cost=dict(online_latency_s=30, total_tokens=None, error=True))]
        value = summarize_costs(rows, 10)
        self.assertEqual(value['processed'], 2)
        self.assertEqual(value['pending'], 8)
        self.assertEqual(value['online_latency_s']['mean'], 20)
        self.assertEqual(value['online_latency_s']['p50'], 20)
        self.assertEqual(value['total_tokens']['missing'], 1)
        self.assertEqual(value['total_tokens']['mean'], 100)
        self.assertEqual(value['failure_rate'], .5)

    def test_resumed_extra_corrections_and_finish_are_counted(self):
        first = dict(repair_calls=2, failed_checks=3, final_submissions=1,
                     stage_timings_s=dict(load_scene=2))
        trace = dict(react_actions=[dict(action='finish')], react_budget=dict(repair_calls=1))
        cost = sample_cost(dict(active_trace=trace, previous_execution_costs=[first],
                               stage_timings_s=dict(load_scene=4)))
        self.assertEqual(cost['repair_calls'], 3)
        self.assertEqual(cost['final_submissions'], 2)
        self.assertEqual(cost['failed_checks'], 3)
        self.assertEqual(cost['cumulative_stage_timings_s']['load_scene'], 6)

    def test_cached_repairs_and_finish_are_not_counted_as_new_model_work(self):
        action = dict(action='finish')
        trace = dict(react_actions=[action, action, action],
            react_budget=dict(repair_calls=2), steps=[
                dict(action=action, observation=dict(tool='monitor_check', error='invalid')),
                dict(action=action, observation=dict(tool='monitor_check', error='invalid'))])
        calls = [dict(stage='react_decision', cache_hit=hit) for hit in (True, True, False)]
        cost = sample_cost(dict(active_trace=trace, calls=calls))
        self.assertEqual(cost['logical_repair_calls'], 2)
        self.assertEqual(cost['repair_calls'], 1)
        self.assertEqual(cost['logical_final_submissions'], 3)
        self.assertEqual(cost['final_submissions'], 1)


if __name__ == '__main__':
    unittest.main()
