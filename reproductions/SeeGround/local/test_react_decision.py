import copy
import json
import unittest
from unittest.mock import patch

from active_diagnostic import EVIDENCE_CAPABILITIES, evidence_request
from decision_api import DecisionAPIError
from react_decision import evidence_tasks, jev_action, jev_state, react_backend, text_state


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


def joint_size_state(audit_repetitions=1):
    current = state()
    missing = ['height measurement incomplete for candidate 12']
    pending = [{'attribute': 'tall', 'role': 'reference', 'phrase': 'two tall bookshelves'}]
    evidence = [
        {'candidate_id': 10, 'status': 'pass', 'attribute_evidence': [
            {'candidate_id': 10, 'status': 'pass', 'reference_ids': [20, 21],
             'interpretation': 'reference_comparison'}]},
        {'candidate_id': 11, 'status': 'fail', 'attribute_evidence': [],
         'exclusion_reason': 'joint predicates fail in all legal assignments'},
        {'candidate_id': 12, 'status': 'unknown', 'attribute_evidence': [
            {'candidate_id': 12, 'status': 'unknown', 'reference_ids': [20, 22],
             'reason': missing[0]}]},
    ]
    audit = [{'verbose_audit_only': 'full assignment proof ' * 100}] * audit_repetitions
    hypotheses = [
        {'references': {'bookshelf_1': 20, 'bookshelf_2': 21}, 'candidate_ids': [10]},
        {'references': {'bookshelf_1': 20, 'bookshelf_2': 22}, 'candidate_ids': [12]},
        {'references': {'bookshelf_1': 21, 'bookshelf_2': 22}, 'candidate_ids': [10, 12]},
    ]
    joint = {'status': 'missing', 'missing': missing, 'candidate_ids': [10, 12],
             'stable_candidate_ids': [], 'pending_size_conditions': pending,
             'hypotheses': [{**hypothesis, 'measurements': audit} for hypothesis in hypotheses],
             'geometry': {'size_constraints': audit}}
    details = {'dimension': 'size', 'candidate_evidence': evidence,
               'comparison_pool_ids': [10, 11, 12, 20, 21, 22],
               'joint_size_diagnosis': joint, 'size_constraints': audit,
               'size_decision': {'joint_hypotheses': joint['hypotheses'], 'attributes': audit}}
    current.update(query='short bookshelf in between the two tall bookshelves',
                   remaining_candidate_ids=[10, 12], category_candidates=[10, 11, 12],
                   evidence_state={'status': 'missing', 'missing': missing,
                                   'pending_size_conditions': pending},
                   observation={'tool': 'evidence', 'dimension': 'size', 'missing': missing,
                                'candidate_evidence': evidence, 'details': details,
                                'internal_steps': [details, details]})
    return current


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

    def test_joint_size_planner_state_does_not_grow_with_duplicate_local_assignment_audits(self):
        small = joint_size_state()
        large = joint_size_state(audit_repetitions=100)
        original = copy.deepcopy(large)
        compact = text_state(large)
        self.assertEqual(compact, text_state(small))
        self.assertLess(len(json.dumps(compact)), len(json.dumps(large)) / 20)
        self.assertNotIn('verbose_audit_only', json.dumps(compact))
        self.assertEqual(large, original)
        self.assertEqual(compact['observation']['candidate_evidence'],
                         large['observation']['candidate_evidence'])
        summary = compact['observation']['details']['joint_size_summary']
        joint = large['observation']['details']['joint_size_diagnosis']
        self.assertEqual(summary['hypotheses'], [
            {'references': h['references'], 'candidate_ids': h['candidate_ids']}
            for h in joint['hypotheses']])
        self.assertEqual(summary['candidate_ids'], [10, 12])
        self.assertEqual(summary['stable_candidate_ids'], [])
        self.assertEqual(summary['missing'], joint['missing'])
        self.assertEqual(summary['pending_size_conditions'], joint['pending_size_conditions'])
        self.assertEqual(compact['evidence_state'], large['evidence_state'])

    def test_jev_planner_receives_all_joint_reference_hypotheses_and_pending_conditions(self):
        current = joint_size_state(audit_repetitions=100)
        original = copy.deepcopy(current)
        expected = text_state(current)['observation']['details']['joint_size_summary']

        def mock(backend, supplied, questions):
            self.assertEqual(backend, 'jev')
            self.assertEqual(supplied['observation']['details']['joint_size_summary'], expected)
            self.assertEqual(supplied['observation']['candidate_evidence'],
                             current['observation']['candidate_evidence'])
            self.assertEqual([item['status'] for item in supplied['observation']['candidate_evidence']],
                             ['pass', 'fail', 'unknown'])
            self.assertEqual(supplied['evidence_state'], current['evidence_state'])
            self.assertEqual(supplied['observation']['missing'], current['observation']['missing'])
            self.assertNotIn('verbose_audit_only', json.dumps(supplied))
            return answer(questions, 'stop')

        call = {'step': 1}
        with patch('react_decision.decide', side_effect=mock):
            result = jev_action(current, 'rules', call, [call])
        self.assertEqual(json.loads(result['_content'])['action'], 'stop')
        self.assertEqual(current, original)

    def test_jev_state_keeps_decision_summaries_and_drops_verbose_audit_data(self):
        source = state()
        source.update(query='skinny cabinet', instructions='long duplicate rules',
                      parsed_query={'Target': ['cabinet'], 'query_rewrite': {'verbose': 'x' * 1000}},
                      target_table=[{'bbox_id': 1, 'category': 'cabinet', 'bbox_3d': [0] * 6}],
                      reference_table=[{'bbox_id': 9, 'category': 'stove', 'bbox_3d': [0] * 6}],
                      observation={'tool': 'initial_diagnosis', 'geometry': {
                          'between_triplets': [{'large': 'x' * 1000}],
                          'between_candidate_evidence': [{'target_id': 1, 'status': 'pass'}]}})
        compact = jev_state(source)
        encoded = json.dumps(compact)
        self.assertNotIn('long duplicate rules', encoded)
        self.assertNotIn('query_rewrite', encoded)
        self.assertNotIn('bbox_3d', encoded)
        self.assertNotIn('between_triplets', encoded)
        self.assertIn('between_candidate_evidence', encoded)
        self.assertEqual(compact['target_candidates'], [{'bbox_id': 1, 'category': 'cabinet'}])

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
        current['observation'] = {'next_evidence': [
            {'action': 'evidence', 'dimension': 'appearance', 'focus_ids': [1],
             'azimuth_deg': 270, 'elevation_deg': 35, 'context': 'scene', 'reason': 'new view'}]}
        self.assertEqual(set(evidence_tasks(current)), {'finish', 'stop'})

    def test_suggested_reference_view_is_selectable_and_keeps_default_tasks(self):
        current = state()
        current['reference_table'] = [{'bbox_id': 9, 'category': 'tv'}]
        suggestion = {'action': 'evidence', 'dimension': 'appearance', 'focus_ids': [1, 9],
                      'azimuth_deg': 270, 'elevation_deg': 35, 'context': 'scene',
                      'reason': 'inspect occluded reference color'}
        current['observation'] = {'next_evidence': [suggestion]}
        tasks = evidence_tasks(current)
        self.assertTrue(set(evidence_tasks(state())).issubset(tasks))
        suggested_key = next(key for key, task in tasks.items() if task == suggestion)
        def mock(backend, supplied, questions):
            self.assertEqual(questions['next_task']['criteria'][suggested_key], suggestion)
            return answer(questions, suggested_key)
        call = {'step': 1}
        with patch('react_decision.decide', side_effect=mock):
            result = jev_action(current, 'rules', call, [call])
        action = json.loads(result['_content'])
        self.assertEqual(action, suggestion)
        self.assertEqual(evidence_request(action, current['remaining_candidate_ids'],
                                         [{'bbox_id': oid} for oid in range(1, 10)]), suggestion)
        self.assertEqual(call['reason_source'], 'selected_task_description')
        self.assertEqual(current['observation']['next_evidence'], [suggestion])

    def test_invalid_suggestions_are_not_exposed_as_jev_tasks(self):
        valid = {'action': 'evidence', 'dimension': 'appearance', 'focus_ids': [1],
                 'azimuth_deg': 270, 'elevation_deg': 35, 'context': 'scene', 'reason': 'inspect'}
        for change in ({'attributes': ['red']}, {'alternate_view': {}},
                       {'focus_ids': [99]}, {'focus_ids': [9]}, {'focus_ids': [True]},
                       {'azimuth_deg': 360}, {'elevation_deg': 0}, {'context': 'invalid'},
                       {'action': 'finish'}, {'dimension': 'position'}, {'reason': ''}):
            with self.subTest(change=change):
                current = state()
                current['reference_table'] = [{'bbox_id': 9, 'category': 'tv'}]
                current['observation'] = {'next_evidence': [{**valid, **change}]}
                self.assertEqual(evidence_tasks(current), evidence_tasks(state()))

    def test_duplicate_suggestion_does_not_add_an_identical_task(self):
        current = state()
        current['observation'] = {'next_evidence': [
            {'action': 'evidence', 'dimension': 'appearance', 'focus_ids': list(range(1, 7)),
             'azimuth_deg': 90, 'elevation_deg': 25, 'context': 'scene',
             'reason': 'same view with another reason'}]}
        self.assertEqual(evidence_tasks(current), evidence_tasks(state()))

    def test_suggestions_respect_choice_limit_without_dropping_default_tasks(self):
        current = state()
        current['remaining_candidate_ids'] = list(range(1, 373))
        current['category_candidates'] = current['remaining_candidate_ids'][:]
        defaults = evidence_tasks(current)
        current['observation'] = {'next_evidence': [
            {'action': 'evidence', 'dimension': 'appearance', 'focus_ids': [1],
             'azimuth_deg': angle, 'elevation_deg': 35, 'context': 'scene', 'reason': 'new view'}
            for angle in [0, 45, 270]]}
        tasks = evidence_tasks(current)
        self.assertEqual(len(tasks), 255)
        self.assertTrue(set(defaults).issubset(tasks))

    def test_selected_task_maps_to_public_action_and_preserves_distribution(self):
        def mock(backend, current, questions):
            self.assertEqual(backend, 'jev')
            self.assertEqual(current['instructions'], 'rules')
            self.assertEqual(current['remaining_candidate_ids'], list(range(1, 9)))
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
