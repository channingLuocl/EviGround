import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from r18_react import ground, measure
from test_r8_grounding import prediction


class ReactTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {'SEEGROUND_V4': '1', 'SEEGROUND_V7': '1',
            'SEEGROUND_V7_4': '1', 'SEEGROUND_V9': '1', 'SEEGROUND_V9_SOURCE': 'r8',
            'SEEGROUND_R8_GROUNDING_REVISION': 'r18'})
        env.start(); self.addCleanup(env.stop)
        self.objects = [prediction(1, ['chair', 'stool'], (0, 0, 1)),
                        prediction(2, ['desk', 'table'], (2, 0, .5))]
        self.query = 'the chair next to the desk'
        self.parsed = {'Target': ['chair'], 'Anchor': ['desk'], '_v4_budget': {}}

    def action(self, kind, focus=(), oid=None):
        value = {'action': kind, 'reason': 'Compare the described objects', 'focus_ids': list(focus),
            'context': 'isolated' if kind == 'inspect' else None,
            'azimuth_deg': 90 if kind == 'inspect' else None,
            'elevation_deg': 30 if kind == 'inspect' else None,
            'predicted_id': oid, 'evidence': [], 'unresolved': [], 'frame': None}
        if oid is not None:
            value['evidence'] = [{'source_quote': self.query, 'object_ids': [oid, 2],
                                  'observation': 'Visible chair beside the described desk'}]
        return {'_content': json.dumps(value), '_finish_reason': 'stop', '_baseline': {}, 'usage': {}}

    def run_actions(self, replies, limit=2, render_side_effect=None):
        trace = {'steps': [], 'internal_usage': {'compute': 0, 'render': 0, 'visual_check': 0}}
        calls = []
        with tempfile.TemporaryDirectory() as directory, \
                patch('v74_grounding.scene_views', return_value=[{'image': 'overview.png'}, {'image': 'top.png'}]), \
                patch('r18_react.image_part', return_value={}), \
                patch('active_diagnostic.render_action', return_value={'image': 'reference.png',
                    'tool': 'render', 'focus_ids': [2], 'marked_ids': [1, 2], 'context': 'isolated'},
                    side_effect=render_side_effect) as renderer, \
                patch('r18_react.chat', side_effect=replies) as model:
            result = ground(self.query, self.parsed, self.objects, self.objects[:1],
                np.zeros((2, 3)), np.zeros((2, 3)), np.array([1, 2]), Path(directory),
                calls, {}, trace, limit)
        return result, trace, model, renderer

    def test_reference_inspection_and_measurements_feed_next_decision(self):
        result, trace, model, renderer = self.run_actions([
            self.action('inspect', [2]), self.action('measure', [1, 2]), self.action('finish', oid=1)])
        self.assertEqual(result['predicted_id'], 1)
        self.assertEqual(model.call_count, 3)
        self.assertEqual(renderer.call_count, 1)
        self.assertEqual([o['tool'] for o in trace['r18_observations']], ['render', 'measure'])
        state = json.loads(model.call_args.args[0][1]['content'][0]['text'])
        self.assertEqual(state['observations'][0]['focus_ids'], [2])
        self.assertEqual(len(state['observations'][1]['directed_pair_measurements']), 2)
        self.assertNotIn('target_id', state)
        self.assertNotIn('correct', state)
        self.assertEqual(trace['final_candidate_ids'], [1])

    def test_unknown_tool_id_returns_error_observation_without_rendering(self):
        result, trace, model, renderer = self.run_actions([
            self.action('inspect', [99]), self.action('finish', oid=1)])
        renderer.assert_not_called()
        self.assertTrue(trace['r18_observations'][0]['no_evidence_acquired'])
        self.assertEqual(result['predicted_id'], 1)

    def test_large_requested_group_is_rendered_in_bounded_panels(self):
        self.objects = [prediction(i, ['chair', 'stool']) for i in range(1, 10)]
        result, trace, model, renderer = self.run_actions([
            self.action('inspect', list(range(1, 10))), self.action('finish', oid=1)])
        self.assertEqual(renderer.call_count, 2)
        self.assertEqual([call.args[0]['focus_ids'] for call in renderer.call_args_list],
                         [list(range(1, 7)), [7, 8, 9]])
        self.assertEqual(trace['r18_observations'][0]['focus_ids'], list(range(1, 10)))
        self.assertEqual(result['predicted_id'], 1)

    def test_unknown_final_id_is_not_accepted(self):
        with self.assertRaisesRegex(ValueError, 'allowed target'):
            self.run_actions([self.action('finish', oid=99)])

    def test_no_evidence_acquisition_after_budget_exhaustion(self):
        with self.assertRaisesRegex(ValueError, 'must finish'):
            self.run_actions([self.action('measure', [1, 2]), self.action('inspect', [2])], limit=1)

    def test_pair_measurements_preserve_direction_and_height(self):
        result = measure([1, 2], self.objects, self.parsed, self.objects[:1], self.query)
        forward, reverse = result['directed_pair_measurements']
        self.assertEqual(forward['delta_xyz_m'], [-2., 0., .5])
        self.assertEqual(reverse['delta_xyz_m'], [2., 0., -.5])
        self.assertEqual(forward['center_3d_m'], reverse['center_3d_m'])

    def test_secondary_class_finish_requests_inspection_before_choice(self):
        self.objects[0] = prediction(1, ['stool', 'chair'], (0, 0, 1))
        result, trace, model, renderer = self.run_actions([
            self.action('finish', oid=1), self.action('inspect', [1, 2]),
            self.action('finish', oid=1)])
        self.assertEqual(model.call_count, 3)
        self.assertEqual(trace['r18_observations'][0]['tool'], 'finish_check')
        self.assertFalse(trace['r18_observations'][0]['accepted'])
        self.assertEqual(result['predicted_id'], 1)
        self.assertEqual(trace['r18_final_decision']['unresolved'], [])

    def test_initial_state_preserves_geometry_and_relative_classifier_scores(self):
        result, trace, model, renderer = self.run_actions([self.action('finish', oid=1)])
        initial = trace['r18_initial_input']
        self.assertIn('center_x_m', initial['objects'][0])
        self.assertEqual(initial['objects'][0]['category_hypotheses'][1], {'category': 'stool', 'rank': 2, 'gap_from_top1': 0.25})

    def test_observer_frame_uses_model_chosen_origin_and_facing(self):
        objects = [prediction(1, ['desk'], (0, 2, 1)),
                   prediction(2, ['door'], (0, 0, 1)),
                   prediction(3, ['chair'], (1, 2, 1)),
                   prediction(4, ['chair'], (-1, 2, 1))]
        result = measure([3, 4], objects, self.parsed, objects[2:], self.query,
                         {'origin_ids': [2], 'facing_ids': [1]})
        frame = result['selected_observer_frame']
        self.assertEqual(frame['right_xy'], [1., 0.])
        self.assertEqual([row[0] for row in frame['rows_left_to_right']], [4, 3])
        self.assertEqual(frame['origin_ids'], [2])
        self.assertEqual(frame['facing_ids'], [1])

    def test_unknown_frame_does_not_discard_independent_pair_measurements(self):
        result = measure([1, 2], self.objects, self.parsed, self.objects[:1], self.query,
                         {'origin_ids': [99], 'facing_ids': [1]})
        self.assertIsNone(result['selected_observer_frame'])
        self.assertIn('known origin_ids', result['observer_frame_error'])
        self.assertEqual(len(result['directed_pair_measurements']), 2)

    def test_coincident_frame_does_not_discard_valid_distances(self):
        result = measure([1, 2], self.objects, self.parsed, self.objects[:1], self.query,
                         {'origin_ids': [2], 'facing_ids': [2]})
        self.assertIsNone(result['selected_observer_frame'])
        self.assertIn('coincide', result['observer_frame_error'])
        self.assertEqual(len(result['directed_pair_measurements']), 2)

    def test_null_camera_angles_use_documented_tool_defaults(self):
        inspect = self.action('inspect', [1, 2])
        value = json.loads(inspect['_content'])
        value.update(azimuth_deg=None, elevation_deg=None)
        inspect['_content'] = json.dumps(value)
        result, trace, model, renderer = self.run_actions([
            inspect, self.action('finish', oid=1)])
        request = renderer.call_args.args[0]
        self.assertEqual(request['azimuth_deg'], 315)
        self.assertEqual(request['elevation_deg'], 30)
        self.assertEqual(trace['r18_observations'][0]['camera_defaults_used'],
                         ['azimuth_deg', 'elevation_deg'])

    def test_failed_render_is_not_cached_as_acquired_evidence(self):
        view = {'image': 'reference.png', 'tool': 'render', 'focus_ids': [2],
                'marked_ids': [2], 'context': 'isolated'}
        result, trace, model, renderer = self.run_actions([
            self.action('inspect', [2]), self.action('inspect', [2]), self.action('finish', oid=1)],
            render_side_effect=[ValueError('No visible points in current view'), view])
        self.assertEqual(renderer.call_count, 2)
        self.assertTrue(trace['r18_observations'][0]['no_evidence_acquired'])
        self.assertEqual(trace['r18_observations'][1]['tool'], 'render')
        self.assertEqual(result['predicted_id'], 1)

    def test_scene_inspection_removes_anonymous_shell_without_gt_names(self):
        from active_diagnostic import render_action
        objects = self.objects + [{'bbox_id': 3, 'target': 'anonymous',
                                   'bbox_3d': [0, 0, 3, 10, 10, 1]}]
        xyz = np.array([[0., 0., 1.], [2., 0., .5], [0., 0., 3.]])
        with tempfile.TemporaryDirectory() as directory, patch('active_diagnostic.render',
                return_value={'points': 2, 'marked_ids': [1, 2], 'visible_pixels': 2,
                              'visible_instance_pixels': {'1': 1, '2': 1}}) as renderer:
            view = render_action({'focus_ids': [1, 2], 'context': 'scene',
                                  'azimuth_deg': 45, 'elevation_deg': 30},
                                 objects, xyz, np.zeros((3, 3)), np.array([1, 2, 3]),
                                 Path(directory), 0)
        self.assertEqual(renderer.call_args.args[0].tolist(), xyz[:2].tolist())
        self.assertEqual(view['hidden_supplemental_instance_ids'], [3])

    def test_requested_supplemental_instance_is_visible_in_cutaway(self):
        from active_diagnostic import render_action
        objects = self.objects + [{'bbox_id': 3, 'target': 'anonymous',
                                   'bbox_3d': [0, 0, 3, 10, 10, 1]}]
        xyz = np.array([[0., 0., 1.], [2., 0., .5], [0., 0., 3.]])
        with tempfile.TemporaryDirectory() as directory, patch('active_diagnostic.render',
                return_value={'points': 3, 'marked_ids': [1, 3], 'visible_pixels': 3,
                              'visible_instance_pixels': {'1': 1, '3': 2}}) as renderer:
            view = render_action({'focus_ids': [1, 3], 'context': 'scene',
                                  'azimuth_deg': 45, 'elevation_deg': 30},
                                 objects, xyz, np.zeros((3, 3)), np.array([1, 2, 3]),
                                 Path(directory), 0)
        self.assertEqual(len(renderer.call_args.args[0]), 3)
        self.assertEqual(view['hidden_supplemental_instance_ids'], [])


if __name__ == '__main__':
    unittest.main()
