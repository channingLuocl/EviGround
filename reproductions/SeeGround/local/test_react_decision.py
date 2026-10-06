import json
import unittest
from unittest.mock import patch

from active_diagnostic import EVIDENCE_CAPABILITIES
from decision_api import DecisionAPIError
from react_decision import evidence_tasks, jev_action, react_backend, text_state


def state():
    return {'remaining_candidate_ids': list(range(1, 9)), 'category_candidates': list(range(1, 9)),
            'available_dimensions': list(EVIDENCE_CAPABILITIES),
            'evidence_capabilities': EVIDENCE_CAPABILITIES, 'tools_remaining': 2,
            'evidence_state': {'status': 'missing'}}


def answer(questions, choice):
    return {'model': 'jev-mock', 'answers': {key: {'type': 'choice', 'choice': choice,
            'confidence': 1., 'probabilities': {option: float(option == choice)
             for option in question['criteria']}} for key, question in questions.items()},
            'usage': {'input_tokens': 2}, 'latency_s': .01, 'validation_warnings': []}


class ReactDecisionTests(unittest.TestCase):
    def test_default_backend_is_jev_and_invalid_backend_fails(self):
        with patch.dict('os.environ', {}, clear=True):
            self.assertEqual(react_backend(), 'jev')
        with patch.dict('os.environ', {'SEEGROUND_REACT_BACKEND': 'invalid'}):
            with self.assertRaises(ValueError):
                react_backend()

    def test_planner_state_retains_checks_but_removes_images_and_internal_calls(self):
        source = {'history': [{'images': ['/private/view.png'], 'internal_steps': [
            {'tool': 'render', 'image': '/private/view.png', 'focus_ids': [1]},
            {'tool': 'visual_check', 'call': {'response': 'raw verifier'}}],
            'checks': [{'id': 1, 'verdict': 'uncertain', 'reason': 'occluded',
                        'image': '/private/view.png'}]}]}
        cleaned = text_state(source)
        self.assertNotIn('/private', json.dumps(cleaned))
        self.assertNotIn('raw verifier', json.dumps(cleaned))
        self.assertEqual(cleaned['history'][0]['checks'][0]['reason'], 'occluded')
        self.assertIn('images', source['history'][0])

    def test_tasks_cover_all_targets_and_offer_alternative_views(self):
        tasks = evidence_tasks(state())
        for dimension in ('position', 'distance', 'size', 'viewpoint'):
            self.assertEqual(tasks[dimension]['dimension'], dimension)
        appearances = [task for task in tasks.values() if task.get('dimension') == 'appearance']
        self.assertEqual(len(appearances), 8)
        self.assertEqual({oid for task in appearances for oid in task['focus_ids']}, set(range(1, 9)))
        self.assertTrue(all(len(task['focus_ids']) <= 6 for task in appearances))
        self.assertEqual({task['azimuth_deg'] for task in appearances}, {90, 180})
        self.assertEqual({task['context'] for task in appearances}, {'scene', 'isolated'})

    def test_budget_exhaustion_offers_only_finish_or_stop(self):
        current = state();current['tools_remaining'] = 0
        self.assertEqual(set(evidence_tasks(current)), {'finish', 'stop'})

    def test_selected_task_maps_to_public_action_and_preserves_distribution(self):
        def mock(backend, current, questions):
            self.assertEqual(backend, 'jev')
            self.assertEqual(current['instructions'], 'rules')
            return answer(questions, 'size')
        call = {'step': 1};calls = [call]
        with patch('react_decision.decide', side_effect=mock):
            result = jev_action(state(), 'rules', call, calls)
        action = json.loads(result['_content'])
        self.assertEqual(action['dimension'], 'size')
        self.assertEqual(call['answers']['next_task']['choice'], 'size')
        self.assertEqual(call['reason_source'], 'selected_task_description')
        self.assertEqual(len(calls), 1)

    def test_finish_target_choices_use_only_live_candidates_and_are_counted(self):
        current = state();current['remaining_candidate_ids'] = [7, 8]
        def mock(backend, current, questions):
            if 'target' in questions:
                self.assertEqual(set(questions['target']['criteria']), {'7', '8'})
                return answer(questions, '8')
            return answer(questions, 'finish')
        call = {'step': 1};calls = [call]
        with patch('react_decision.decide', side_effect=mock):
            result = jev_action(current, 'rules', call, calls)
        self.assertEqual(json.loads(result['_content'])['predicted_id'], 8)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]['stage'], 'active_target_decision')
        self.assertEqual(result['_target_decision']['_baseline']['model'], 'jev-mock')

    def test_empty_live_scope_offers_no_appearance_and_uses_initial_fallback(self):
        current = state();current['remaining_candidate_ids'] = []
        self.assertFalse(any(task.get('dimension') == 'appearance'
                             for task in evidence_tasks(current).values()))

    def test_invalid_jev_response_is_recorded_without_qwen_fallback(self):
        call = {'step': 1}
        with patch('react_decision.decide', side_effect=DecisionAPIError('bad choice', {'answers': {}})):
            with self.assertRaisesRegex(ValueError, 'bad choice'):
                jev_action(state(), 'rules', call, [call])
        self.assertEqual(call['error_response'], {'answers': {}})


if __name__ == '__main__':
    unittest.main()
