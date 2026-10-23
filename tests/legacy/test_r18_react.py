import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from eviground.grounding.r18_react import compact_observations, current_views, ground, measure, measured_together
from tests.legacy.test_r8_grounding import prediction


class ReactTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {'EVIGROUND_V4': '1', 'EVIGROUND_V7': '1',
            'EVIGROUND_API_PROVIDER': 'openai',
            'EVIGROUND_V7_4': '1', 'EVIGROUND_V9': '1', 'EVIGROUND_V9_SOURCE': 'r8',
            'EVIGROUND_R8_GROUNDING_REVISION': 'r18'})
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

    def run_actions(self, replies, limit=2, render_side_effect=None, candidate_objects=None):
        trace = {'steps': [], 'internal_usage': {'compute': 0, 'render': 0, 'visual_check': 0}}
        calls = []
        candidate_objects = self.objects[:1] if candidate_objects is None else candidate_objects
        with tempfile.TemporaryDirectory() as directory, \
                patch('eviground.grounding.v74_grounding.scene_views', return_value=[{'image': 'overview.png'}, {'image': 'top.png'}]), \
                patch('eviground.grounding.r18_react.image_part', return_value={}), \
                patch('eviground.grounding.active_diagnostic.render_action', return_value={'image': 'reference.png',
                    'tool': 'render', 'focus_ids': [2], 'marked_ids': [1, 2], 'context': 'isolated'},
                    side_effect=render_side_effect) as renderer, \
                patch('eviground.grounding.r18_react.chat', side_effect=replies) as model:
            result = ground(self.query, self.parsed, self.objects, candidate_objects,
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
        encoded = state['observations'][1]['directed_pair_measurements']
        self.assertEqual([dict(zip(encoded['columns'], row)) for row in encoded['rows']],
                         trace['r18_observations'][1]['directed_pair_measurements'])
        self.assertNotIn('target_id', state)
        self.assertNotIn('correct', state)
        self.assertEqual(trace['final_candidate_ids'], [1])

    def test_unknown_tool_id_returns_error_observation_without_rendering(self):
        result, trace, model, renderer = self.run_actions([
            self.action('inspect', [99]), self.action('finish', oid=1)])
        renderer.assert_not_called()
        self.assertTrue(trace['r18_observations'][0]['no_evidence_acquired'])
        self.assertEqual(result['predicted_id'], 1)

    def test_duplicate_measure_error_survives_the_next_decision(self):
        result, trace, model, renderer = self.run_actions([
            self.action('measure', [1, 2]), self.action('measure', [1, 2]),
            self.action('finish', oid=1)])
        self.assertEqual(result['predicted_id'], 1)
        self.assertEqual(trace['r18_model_tool_calls'], 1)
        error = trace['r18_observations'][1]
        self.assertTrue(error['no_evidence_acquired'])
        self.assertIn('already acquired', error['error'])
        state = json.loads(model.call_args.args[0][1]['content'][0]['text'])
        self.assertEqual(state['observations'][1], error)
        encoded = state['observations'][0]['directed_pair_measurements']
        self.assertEqual([dict(zip(encoded['columns'], row)) for row in encoded['rows']],
                         trace['r18_observations'][0]['directed_pair_measurements'])

    def test_repeated_measurements_reserve_a_final_decision_turn(self):
        def reply(messages, **kwargs):
            state = json.loads(messages[1]['content'][0]['text'])
            if state['available_actions'] == ['finish']:
                return self.action('finish', oid=1)
            return self.action('measure', [1, 2])

        result, trace, model, renderer = self.run_actions(reply, limit=3)
        self.assertEqual(result['predicted_id'], 1)
        self.assertEqual(model.call_count, 15)
        self.assertEqual(trace['r18_model_tool_calls'], 1)
        state = json.loads(model.call_args.args[0][1]['content'][0]['text'])
        self.assertEqual(state['remaining_tool_calls'], 0)
        self.assertEqual(state['decision_phase'], 'final_choice')
        self.assertEqual(model.call_args.kwargs['response_schema']['schema']
                         ['properties']['action']['enum'], ['finish'])
        self.assertTrue(all(item.get('no_evidence_acquired')
                            for item in trace['r18_observations'][1:]))

    def test_final_reserved_turn_discloses_unmet_evidence_without_reopening_tools(self):
        self.objects.append(prediction(3, ['chair'], (4, 0, 1)))

        def reply(messages, **kwargs):
            state = json.loads(messages[1]['content'][0]['text'])
            if state['available_actions'] == ['finish']:
                return self.action('finish', oid=1)
            return self.action('measure', [1, 2])

        result, trace, model, renderer = self.run_actions(
            reply, limit=3, candidate_objects=[self.objects[0], self.objects[2]])
        self.assertEqual(result['predicted_id'], 1)
        self.assertEqual(model.call_count, 15)
        self.assertEqual(trace['r18_controller_tool_calls'], 0)
        self.assertIn('measure all target candidates and bound references for the spatial/size clause',
                      trace['r18_final_decision']['unresolved'])
        renderer.assert_not_called()

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
        result, trace, model, renderer = self.run_actions([
            self.action('measure', [1, 2]), self.action('inspect', [2]), self.action('finish', oid=1)],
            limit=1)
        observations = trace['r18_observations']
        self.assertEqual(model.call_count, 3)
        self.assertEqual([item['tool'] for item in observations], ['measure', 'budget_exhausted'])
        self.assertEqual(observations[-1]['requested_tool'], 'inspect')
        self.assertTrue(observations[-1]['no_evidence_acquired'])
        renderer.assert_not_called()
        self.assertEqual(result['predicted_id'], 1)

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

    def test_spatial_finish_forces_complete_measurement_pool(self):
        self.objects = [prediction(1, ['chair'], (0, 0, 1)),
                        prediction(2, ['desk'], (2, 0, .5)),
                        prediction(3, ['chair'], (4, 0, 1))]
        self.parsed = {'Target': ['chair'], 'Anchor': ['desk'],
                       'Relations': [{'relation': 'nearest'}], '_v4_budget': {}}
        self.query = 'the chair next to the desk'
        result, trace, model, renderer = self.run_actions([
            self.action('finish', oid=1), self.action('finish', oid=1),
            self.action('finish', oid=1), self.action('finish', oid=1),
            self.action('finish', oid=1)],
            candidate_objects=[self.objects[0], self.objects[2]])
        self.assertEqual(result['predicted_id'], 1)
        self.assertEqual(model.call_count, 5)
        forced = trace['r18_observations'][0]
        self.assertTrue(forced['controller_forced'])
        self.assertEqual(forced['tool'], 'measure')
        self.assertEqual({row['id'] for row in forced['objects']}, {1, 2, 3})

    def test_alternative_finish_requires_primary_rival_view(self):
        self.objects = [prediction(1, ['stool', 'chair'], (0, 0, 1)),
                        prediction(2, ['chair', 'stool'], (2, 0, 1))]
        self.parsed = {'Target': ['chair'], 'Anchor': [], '_v4_budget': {}}
        self.query = 'the chair'
        result, trace, model, renderer = self.run_actions([
            self.action('finish', oid=1), self.action('inspect', [1, 2]),
            self.action('finish', oid=1), self.action('finish', oid=1)], limit=2,
            candidate_objects=self.objects)
        self.assertEqual(result['predicted_id'], 1)
        self.assertEqual(trace['r18_observations'][0]['tool'], 'finish_check')
        self.assertEqual(renderer.call_count, 1)

    def test_initial_state_preserves_geometry_and_relative_classifier_scores(self):
        result, trace, model, renderer = self.run_actions([self.action('finish', oid=1)])
        initial = trace['r18_initial_input']
        self.assertIn('center_x_m', initial['objects'][0])
        self.assertEqual(initial['objects'][0]['category_hypotheses'][1], {'category': 'stool', 'rank': 2, 'gap_from_top1': 0.25})

    def test_guard_reminder_does_not_consume_evidence_budget(self):
        self.objects[0] = prediction(1, ['stool', 'chair'])
        result, trace, model, renderer = self.run_actions([
            self.action('finish', oid=1), self.action('inspect', [1, 2]),
            self.action('finish', oid=1)], limit=1)
        states = [json.loads(call.args[0][1]['content'][0]['text']) for call in model.call_args_list]
        self.assertEqual([s['remaining_tool_calls'] for s in states], [1, 1, 0])
        self.assertEqual([s['available_actions'] for s in states],
                         [['inspect', 'measure', 'finish'], ['inspect', 'measure', 'finish'], ['finish']])
        for state, call in zip(states, model.call_args_list):
            self.assertEqual(state['available_actions'],
                             call.kwargs['response_schema']['schema']['properties']['action']['enum'])
        self.assertEqual(states[-1]['decision_phase'], 'final_choice')
        self.assertIn('acquisition is CLOSED', model.call_args.args[0][0]['content'])
        self.assertEqual(model.call_args.kwargs['response_schema']['schema']['properties']['action']['enum'], ['finish'])
        self.assertEqual(trace['r18_model_tool_calls'], 1)
        self.assertEqual(trace['r18_controller_tool_calls'], 0)

    def test_reference_guard_does_not_skip_visual_fallback(self):
        self.objects.append(prediction(3, ['shelf'], (4, 0, 1)))
        self.parsed['candidate_generation'] = {'visual_fallback_ids': [3]}
        view = {'image': 'fallback.png', 'marked_ids': [2, 3], 'focus_ids': [2, 3]}
        result, trace, model, renderer = self.run_actions([
            self.action('measure', [1, 2, 3]), self.action('finish', oid=1),
            self.action('finish', oid=1), self.action('finish', oid=1),
            self.action('finish', oid=1)], candidate_objects=[self.objects[0], self.objects[2]],
            render_side_effect=[view])
        self.assertEqual(renderer.call_args.args[0]['focus_ids'], [2, 3])
        self.assertEqual(trace['r18_model_tool_calls'], 1)
        self.assertEqual(trace['r18_controller_tool_calls'], 1)
        self.assertEqual(trace['r18_final_decision']['unresolved'], [])

    def test_compaction_is_lossless_and_leaves_audit_evidence_unchanged(self):
        self.objects = [prediction(i, ['chair'], (i, i % 3, 1)) for i in range(20)]
        raw = measure(list(range(20)), self.objects, self.parsed, self.objects[:1], self.query)
        before = json.dumps(raw, sort_keys=True)
        encoded = compact_observations([raw])[0]['directed_pair_measurements']
        self.assertEqual([dict(zip(encoded['columns'], row)) for row in encoded['rows']],
                         raw['directed_pair_measurements'])
        self.assertEqual(json.dumps(raw, sort_keys=True), before)
        self.assertLess(len(json.dumps(compact_observations([raw]))), len(before) / 2)

    def test_later_measurement_supersedes_only_same_frame_subsets(self):
        small = measure([1], self.objects, self.parsed, self.objects[:1], self.query)
        whole = measure([1, 2], self.objects, self.parsed, self.objects[:1], self.query)
        self.assertEqual(len(compact_observations([small, whole])), 1)
        small['selected_observer_frame'] = {'origin_ids': [], 'facing_ids': [2]}
        self.assertEqual(len(compact_observations([small, whole])), 2)

    def test_disjoint_measurements_do_not_establish_a_complete_comparison(self):
        observations = [measure([i], self.objects, self.parsed, self.objects[:1], self.query)
                        for i in (1, 2)]
        self.assertFalse(measured_together([1, 2], observations))
        observations.append(measure([1, 2], self.objects, self.parsed, self.objects[:1], self.query))
        self.assertTrue(measured_together([1, 2], observations))

    def test_attached_images_are_bounded_and_old_views_remain_in_result(self):
        views = [{'image': f'view-{i}.png', 'marked_ids': [1, 2]} for i in range(4)]
        result, trace, model, renderer = self.run_actions([
            self.action('inspect', [1]), self.action('inspect', [2]),
            self.action('inspect', [1]), self.action('inspect', [2]),
            self.action('finish', oid=1)], limit=4, render_side_effect=views)
        self.assertEqual(len(result['images']), 6)
        self.assertEqual(renderer.call_count, 4)
        for call in model.call_args_list:
            self.assertLessEqual(len(call.args[0][1]['content']), 5)
        self.assertEqual([v['image'] for v in current_views([{'image': 'overview.png'}, *views])],
                         ['overview.png', 'view-1.png', 'view-2.png', 'view-3.png'])

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

    def test_observer_frame_provides_both_signed_axis_orders(self):
        objects = self.objects + [prediction(3, ['chair'], (3, -1, 1)),
                                  prediction(4, ['chair'], (1, 1, 1)),
                                  prediction(5, ['chair'], (-1, 0, 1))]
        result = measure([3, 4, 5], objects, self.parsed, objects[2:], self.query,
                         {'origin_ids': [1], 'facing_ids': [2]})
        frame = result['selected_observer_frame']
        self.assertEqual([row[0] for row in frame['rows_left_to_right']], [4, 5, 3])
        self.assertEqual([row[0] for row in frame['rows_forward_increasing']], [5, 4, 3])
        self.assertEqual([row[2] for row in frame['rows_forward_increasing']], [-1., 1., 3.])
        self.assertEqual(frame['columns'], ['target_id', 'right_m', 'forward_m'])
        self.assertIn('bounding-box centers', frame['coordinate_semantics'])

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
        from eviground.grounding.active_diagnostic import render_action
        objects = self.objects + [{'bbox_id': 3, 'target': 'anonymous',
                                   'bbox_3d': [0, 0, 3, 10, 10, 1]}]
        xyz = np.array([[0., 0., 1.], [2., 0., .5], [0., 0., 3.]])
        with tempfile.TemporaryDirectory() as directory, patch('eviground.grounding.active_diagnostic.render',
                return_value={'points': 2, 'marked_ids': [1, 2], 'visible_pixels': 2,
                              'visible_instance_pixels': {'1': 1, '2': 1}}) as renderer:
            view = render_action({'focus_ids': [1, 2], 'context': 'scene',
                                  'azimuth_deg': 45, 'elevation_deg': 30},
                                 objects, xyz, np.zeros((3, 3)), np.array([1, 2, 3]),
                                 Path(directory), 0)
        self.assertEqual(renderer.call_args.args[0].tolist(), xyz[:2].tolist())
        self.assertEqual(view['hidden_supplemental_instance_ids'], [3])

    def test_requested_supplemental_instance_is_visible_in_cutaway(self):
        from eviground.grounding.active_diagnostic import render_action
        objects = self.objects + [{'bbox_id': 3, 'target': 'anonymous',
                                   'bbox_3d': [0, 0, 3, 10, 10, 1]}]
        xyz = np.array([[0., 0., 1.], [2., 0., .5], [0., 0., 3.]])
        with tempfile.TemporaryDirectory() as directory, patch('eviground.grounding.active_diagnostic.render',
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
