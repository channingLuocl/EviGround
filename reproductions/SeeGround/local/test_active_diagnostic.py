import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from active_diagnostic import (apply_computed_evidence, _check_payload,
                               active_ground, compute_observation,
                               position_candidate_evidence, render_action)
from diagnostic import diagnose


def obj(oid, category, x, y, z=.5):
    return {'bbox_id': oid, 'target': category,
            'bbox_3d': [x, y, z, .4, .4, 1.]}


def parsed():
    return {
        'Target': ['window'], 'Anchor': [], 'Relations': [],
        'Observer': {'origin': {}, 'facing': {}}, 'Viewpoint': '',
        'Attributes': [], 'Unhandled': [], 'diagnostic_missing': [],
    }


class ActiveDiagnosticTests(unittest.TestCase):
    def run_actions(self, actions, limit=4, compute=None):
        objects = [obj(1, 'window', 0, 0), obj(2, 'window', 2, 0)]
        messages_seen = []
        responses = iter(actions)
        def fake_chat(messages, **kwargs):
            messages_seen.append(messages)
            return {'_content': json.dumps(next(responses)),
                    '_baseline': {'model': 'mock', 'latency_s': .01},
                    '_finish_reason': 'stop', 'usage': {}}
        with tempfile.TemporaryDirectory() as directory, patch(
                'active_diagnostic.chat', side_effect=fake_chat), patch(
                'active_diagnostic.render_action', return_value={
                    'tool': 'render', 'image': 'mock.png', 'focus_ids': [1, 2]}), patch(
                'active_diagnostic.image_part', return_value={'type': 'text', 'text': 'mock image'}), patch(
                'active_diagnostic.compute_observation', wraps=compute or compute_observation):
            trace, calls = {}, []
            result = active_ground(
                'window', parsed(), objects, objects, objects,
                np.zeros((2, 3)), np.zeros((2, 3), dtype=np.uint8),
                np.array([1, 2]), Path(directory), calls, trace, limit, True)
        return result, messages_seen

    def test_explicit_stop_preserves_unresolved_candidates(self):
        result, _ = self.run_actions([{'action': 'stop', 'reason': 'no distinguishing evidence'}])
        self.assertIsNone(result['predicted_id'])
        self.assertIsNone(result['bbox_3d'])
        self.assertFalse(result['accepted'])
        self.assertEqual(result['decision_status'], 'missing')
        self.assertEqual(result['active_trace']['final_candidate_ids'], [1, 2])

    def test_invalid_finish_can_recover(self):
        result, messages = self.run_actions([
            {'action': 'finish', 'predicted_id': 99, 'reason': 'invalid ID'},
            {'action': 'stop', 'reason': 'cannot distinguish candidates'},
        ])
        self.assertEqual(result['active_trace']['rejected_actions'], 1)
        self.assertEqual(result['active_trace']['successful_tools'], 0)
        self.assertEqual(result['active_trace']['action_attempts'], 2)
        self.assertIn('Controller correction', messages[1][0]['content'])

    def test_repeated_actions_stop_at_attempt_limit(self):
        action = {'action': 'compute', 'dimension': 'appearance', 'reason': 'inspect'}
        result, _ = self.run_actions([action] * 7, limit=2)
        trace = result['active_trace']
        self.assertEqual(trace['stop_reason'], 'action_attempt_budget_exhausted')
        self.assertEqual(trace['successful_tools'], 1)
        self.assertEqual(trace['rejected_actions'], 6)
        self.assertEqual(trace['action_attempts'], 7)
        self.assertFalse(result['accepted'])

    def test_ignored_tool_budget_does_not_execute_extra_tool(self):
        result, _ = self.run_actions([
            {'action': 'compute', 'dimension': 'size', 'reason': 'inspect size'},
            {'action': 'compute', 'dimension': 'appearance', 'reason': 'inspect color'},
        ], limit=1)
        self.assertEqual(result['active_trace']['stop_reason'], 'tool_budget_exhausted')
        self.assertEqual(result['active_trace']['successful_tools'], 1)
        self.assertIsNone(result['predicted_id'])

    def test_visual_check_cannot_revive_excluded_candidate(self):
        result, messages = self.run_actions([
            {'action': 'render', 'reason': 'inspect'},
            {'action': 'visual_check', 'checks': [
                {'id': 1, 'verdict': 'no', 'reason': 'visible mismatch'}], 'reason': 'exclude'},
            {'action': 'visual_check', 'checks': [
                {'id': 1, 'verdict': 'yes', 'reason': 'changed mind'}], 'reason': 'revive'},
            {'action': 'finish', 'predicted_id': 2, 'reason': 'remaining match'},
        ])
        self.assertEqual(result['active_trace']['final_candidate_ids'], [2])
        self.assertEqual(result['active_trace']['rejected_actions'], 1)
        update = result['active_trace']['steps'][1]['observation']['candidate_update']
        self.assertEqual(update, {'before': [1, 2], 'removed_ids': [1], 'after': [2]})
        state = json.loads(messages[2][1]['content'][-1]['text'])
        self.assertEqual(state['evidence_state']['remaining_candidate_ids'], [2])

    def test_visual_evidence_preserves_computed_exclusions(self):
        def compute(*args):
            return {'tool': 'compute', 'dimension': 'position', 'candidate_evidence': [
                {'candidate_id': 1, 'status': 'fail'}, {'candidate_id': 2, 'status': 'pass'}]}
        result, _ = self.run_actions([
            {'action': 'compute', 'dimension': 'position', 'reason': 'check relation'},
            {'action': 'render', 'reason': 'inspect remaining appearance'},
            {'action': 'visual_check', 'checks': [
                {'id': 2, 'verdict': 'yes', 'reason': 'visible match'}], 'reason': 'check appearance'},
            {'action': 'finish', 'predicted_id': 2, 'reason': 'combined evidence'},
        ], compute=compute)
        self.assertEqual(result['active_trace']['final_candidate_ids'], [2])
        self.assertEqual(result['active_trace']['successful_tools'], 3)

    def test_duplicate_visual_verdicts_are_rejected(self):
        with self.assertRaises(ValueError):
            _check_payload({'checks': [
                {'id': 1, 'verdict': 'yes', 'reason': 'match'},
                {'id': 1, 'verdict': 'no', 'reason': 'mismatch'}]},
                [1], [obj(1, 'window', 0, 0)], parsed())

    def test_invalid_render_does_not_consume_successful_tool_budget(self):
        objects = [obj(1, 'window', 0, 0), obj(2, 'window', 2, 0)]
        responses = iter([
            {'_content': json.dumps({
                'action': 'render', 'focus_ids': [1, 2, 3, 4, 5, 6, 7],
                'azimuth_deg': 90, 'elevation_deg': 25,
                'context': 'scene', 'reason': 'too many focus instances',
            }), '_baseline': {'model': 'mock', 'latency_s': .01},
             '_finish_reason': 'stop', 'usage': {}},
            {'_content': json.dumps({
                'action': 'finish', 'predicted_id': 1,
                'reason': 'fallback after invalid tool request',
            }), '_baseline': {'model': 'mock', 'latency_s': .01},
             '_finish_reason': 'stop', 'usage': {}},
        ])
        parsed_query = parsed()
        with tempfile.TemporaryDirectory() as directory, patch(
                'active_diagnostic.chat', side_effect=lambda *args, **kwargs: next(responses)):
            trace, calls = {}, []
            result = active_ground(
                'window', parsed_query, objects, objects, objects,
                np.zeros((2, 3)), np.zeros((2, 3), dtype=np.uint8),
                np.array([1, 2]), Path(directory), calls, trace, 1, True)
        self.assertEqual(result['predicted_id'], 1)
        self.assertEqual(trace['stop_reason'], 'model_finish')
        self.assertEqual(len(trace['steps']), 2)
        self.assertTrue(trace['steps'][0]['observation']['no_evidence_acquired'])

    def test_duplicate_successful_action_is_recoverable(self):
        objects = [obj(1, 'window', 0, 0), obj(2, 'window', 2, 0)]
        action = {'action': 'compute', 'dimension': 'appearance',
                  'reason': 'inspect appearance'}
        responses = iter([
            {'_content': json.dumps(action),
             '_baseline': {'model': 'mock', 'latency_s': .01},
             '_finish_reason': 'stop', 'usage': {}},
            {'_content': json.dumps(action),
             '_baseline': {'model': 'mock', 'latency_s': .01},
             '_finish_reason': 'stop', 'usage': {}},
            {'_content': json.dumps({'action': 'finish', 'predicted_id': 1,
                                     'reason': 'finish after duplicate'}),
             '_baseline': {'model': 'mock', 'latency_s': .01},
             '_finish_reason': 'stop', 'usage': {}},
        ])
        messages_seen = []
        def fake_chat(messages, **kwargs):
            messages_seen.append(messages)
            return next(responses)
        with tempfile.TemporaryDirectory() as directory, patch(
                'active_diagnostic.chat', side_effect=fake_chat):
            trace, calls = {}, []
            result = active_ground(
                'window', parsed(), objects, objects, objects,
                np.zeros((2, 3)), np.zeros((2, 3), dtype=np.uint8),
                np.array([1, 2]), Path(directory), calls, trace, 2, True)
        self.assertEqual(result['predicted_id'], 1)
        self.assertTrue(trace['steps'][1]['observation']['no_evidence_acquired'])
        self.assertIn('Controller correction', messages_seen[2][0]['content'])

    def test_position_evidence_distinguishes_fail_from_unknown(self):
        diagnosis = {
            'candidate_ids': [1, 2, 3], 'stable_candidate_ids': [], 'missing': [],
            'geometry': {
                'relation_pairs': [],
                'between_candidate_evidence': [
                    {'target_id': 1, 'passing_reference_pairs': [[7, 8]],
                     'unknown_reference_pairs': [], 'failing_reference_pairs': []},
                    {'target_id': 2, 'passing_reference_pairs': [],
                     'unknown_reference_pairs': [], 'failing_reference_pairs': [[7, 8]]},
                    {'target_id': 3, 'passing_reference_pairs': [],
                     'unknown_reference_pairs': [[7, 8]], 'failing_reference_pairs': []},
                ],
            },
        }
        evidence = position_candidate_evidence(diagnosis)
        self.assertEqual([item['status'] for item in evidence],
                         ['pass', 'fail', 'unknown'])
        update = apply_computed_evidence(diagnosis, evidence, 'position')
        self.assertEqual(update['removed_ids'], [2])
        self.assertEqual(diagnosis['candidate_ids'], [1, 3])

    def test_render_rejects_overloaded_focus(self):
        objects = [obj(index, 'window', index, 0) for index in range(1, 8)]
        with self.assertRaises(ValueError):
            render_action(
                {'focus_ids': list(range(1, 8)), 'azimuth_deg': 90,
                 'elevation_deg': 25, 'context': 'scene'}, objects,
                np.zeros((1, 3)), np.zeros((1, 3), dtype=np.uint8),
                np.array([1]), Path('/tmp'), 1)

    def test_each_dimension_is_an_explicit_model_selectable_tool(self):
        objects = [obj(1, 'window', 0, 0), obj(2, 'window', 2, 0)]
        diagnosis = {
            'candidate_ids': [1, 2], 'geometry': {
                'relation_pairs': [], 'between_candidate_evidence': [],
                'between_triplets': [], 'candidate_metrics': [],
                'reference_metrics': [],
            }, 'hypotheses': [], 'visual_needs': [],
        }
        for dimension in ('viewpoint', 'position', 'distance', 'appearance', 'size'):
            with self.subTest(dimension=dimension):
                result = compute_observation(dimension, 'window', parsed(), diagnosis,
                                             objects, objects)
                self.assertEqual(result['tool'], 'compute')
                self.assertEqual(result['dimension'], dimension)

    def test_visual_check_only_accepts_rendered_candidate_ids(self):
        objects = [obj(1, 'window', 0, 0), obj(2, 'window', 2, 0)]
        result = _check_payload(
            {'checks': [{'id': 1, 'verdict': 'no', 'reason': 'visible mismatch'}]},
            [1], objects, parsed())
        self.assertEqual(result[0]['id'], 1)
        self.assertEqual(result[0]['verdict'], 'no')
        with self.assertRaises(ValueError):
            _check_payload(
                {'checks': [{'id': 2, 'verdict': 'no', 'reason': 'not rendered'}]},
                [1], objects, parsed())

    def test_visual_no_excludes_candidate_with_annotated_categories(self):
        objects = [obj(1, 'window', 0, 0), obj(2, 'window', 2, 0)]
        initial = diagnose(parsed(), objects, objects, [0, 0, .5], trusted_categories=True)
        self.assertEqual(initial['candidate_ids'], [1, 2])
        checked = diagnose(parsed(), objects, objects, [0, 0, .5],
                           checks=[{'id': 1, 'category': 'window', 'verdict': 'no'}],
                           trusted_categories=True)
        self.assertEqual(checked['candidate_ids'], [2])
        self.assertEqual(checked['status'], 'resolved')

    def test_visual_uncertain_preserves_candidate(self):
        objects = [obj(1, 'window', 0, 0), obj(2, 'window', 2, 0)]
        checked = diagnose(parsed(), objects, objects, [0, 0, .5],
                           checks=[{'id': 1, 'category': 'window', 'verdict': 'uncertain'}],
                           trusted_categories=True)
        self.assertEqual(checked['candidate_ids'], [1, 2])
        self.assertEqual(checked['status'], 'missing')


if __name__ == '__main__':
    unittest.main()
