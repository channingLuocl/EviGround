"""A model cannot call its own completed task unresolved in the same action."""
from copy import deepcopy
import unittest

from eviground.grounding.dag_monitor import DagMonitor
from tests.helpers import parsed
from tests.unit.test_dag_monitor import QUERY, action, node, plan


class CompletionConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.monitor = DagMonitor(QUERY, parsed(), {1, 2, 3}, {1, 2},
                                  ['inspect', 'measure', 'finish'])
        self.start = plan()
        self.start['nodes'][1]['goal'] = 'verify the middle chair'
        self.monitor.update(self.start, action(), [])

    def proposal(self):
        proposal = deepcopy(self.start)
        proposal['nodes'][1].update(status='completed', evidence_ids=['initial'],
                                    conclusion='The candidates are all chairs.')
        proposal['nodes'].append(node('more', 'relation', ['compare']))
        proposal['nodes'][2]['depends_on'] = ['more']
        proposal['current_node_id'] = 'more'
        return proposal

    def test_completed_and_unresolved_same_goal_is_rejected_atomically(self):
        before, history = deepcopy(self.monitor.plan), deepcopy(self.monitor.history)
        proposal = self.proposal()
        original = deepcopy(proposal)
        response = action()
        response['unresolved'] = ['verify the middle chair']
        with self.assertRaisesRegex(ValueError, 'Completed task compare.*unresolved'):
            self.monitor.update(proposal, response, [])
        self.assertEqual(self.monitor.plan, before)
        self.assertEqual(self.monitor.history, history)
        self.assertEqual(proposal, original)

    def test_case_and_whitespace_cannot_hide_same_goal(self):
        response = action()
        response['unresolved'] = ['  VERIFY  the\n MIDDLE chair  ']
        with self.assertRaisesRegex(ValueError, 'Completed task compare.*unresolved'):
            self.monitor.update(self.proposal(), response, [])

    def test_node_label_suffix_cannot_hide_completed_goal(self):
        before, history = deepcopy(self.monitor.plan), deepcopy(self.monitor.history)
        response = action()
        response['unresolved'] = ['verify the middle chair (compare)']
        with self.assertRaisesRegex(ValueError, 'Completed task compare.*unresolved'):
            self.monitor.update(self.proposal(), response, [])
        self.assertEqual(self.monitor.plan, before)
        self.assertEqual(self.monitor.history, history)

    def test_node_label_case_and_spacing_are_normalized(self):
        response = action()
        response['unresolved'] = [' VERIFY the middle CHAIR ( COMPARE ) ']
        with self.assertRaisesRegex(ValueError, 'Completed task compare.*unresolved'):
            self.monitor.update(self.proposal(), response, [])

    def test_tagged_pending_goal_is_still_allowed(self):
        response = action()
        response['unresolved'] = ['verify the middle chair (compare)']
        self.monitor.update(deepcopy(self.start), response, [])
        self.assertEqual(self.monitor.plan['nodes'][1]['status'], 'pending')

    def test_unresolved_other_task_preserves_valid_completed_state(self):
        response = action()
        response['unresolved'] = ['more']
        self.monitor.update(self.proposal(), response, [])
        self.assertEqual(self.monitor.plan['nodes'][1]['status'], 'completed')
        self.assertEqual(self.monitor.plan['current_node_id'], 'more')

    def test_pending_task_may_still_be_unresolved(self):
        response = action()
        response['unresolved'] = ['verify the middle chair']
        self.monitor.update(deepcopy(self.start), response, [])
        self.assertEqual(self.monitor.plan['nodes'][1]['status'], 'pending')

    def test_no_unresolved_claim_does_not_add_a_completion_requirement(self):
        self.monitor.update(self.proposal(), action(), [])
        self.assertEqual(self.monitor.plan['nodes'][1]['status'], 'completed')

    def test_abstention_must_explain_missing_evidence_atomically(self):
        proposal = deepcopy(self.start)
        proposal['current_node_id'] = 'verify'
        before, history = deepcopy(self.monitor.plan), deepcopy(self.monitor.history)
        for missing in ([], ['  ']):
            response = action('finish')
            response['unresolved'] = missing
            with self.subTest(unresolved=missing), self.assertRaisesRegex(
                    ValueError, 'Abstention must state.*unresolved'):
                self.monitor.update(proposal, response, [])
            self.assertEqual(self.monitor.plan, before)
            self.assertEqual(self.monitor.history, history)

    def test_explained_early_abstention_keeps_pending_tasks_and_candidates(self):
        proposal = deepcopy(self.start)
        proposal['current_node_id'] = 'verify'
        response = action('finish')
        response['unresolved'] = ['The comparison group is not visible in the available views.']
        self.monitor.update(proposal, response, [])
        self.assertEqual(self.monitor.plan['nodes'][1]['status'], 'pending')
        self.assertEqual(self.monitor.plan['nodes'][2]['status'], 'pending')
        self.assertEqual(self.monitor.active_ids(), [1, 2])

    def test_rollback_resets_before_consistency_check(self):
        self.monitor.update(self.proposal(), action(), [])
        proposal = deepcopy(self.monitor.plan)
        proposal['reopen_node_ids'] = ['compare']
        proposal['current_node_id'] = 'compare'
        response = action()
        response['unresolved'] = ['verify the middle chair']
        self.monitor.update(proposal, response, [])
        self.assertEqual(self.monitor.plan['nodes'][1]['status'], 'pending')
        self.assertEqual(self.monitor.plan['nodes'][3]['status'], 'pending')


if __name__ == '__main__':
    unittest.main()
