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
        with tempfile.TemporaryDirectory() as directory, patch(
                'active_diagnostic.chat', side_effect=lambda *args, **kwargs: next(responses)):
            trace, calls = {}, []
            result = active_ground(
                'window', parsed(), objects, objects, objects,
                np.zeros((2, 3)), np.zeros((2, 3), dtype=np.uint8),
                np.array([1, 2]), Path(directory), calls, trace, 2, True)
        self.assertEqual(result['predicted_id'], 1)
        self.assertTrue(trace['steps'][1]['observation']['no_evidence_acquired'])

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
