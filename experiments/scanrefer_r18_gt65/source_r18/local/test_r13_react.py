import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from r13_react import ground, measure
from test_r8_grounding import prediction


class ReactTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {'SEEGROUND_V4': '1', 'SEEGROUND_V7': '1',
            'SEEGROUND_V7_4': '1', 'SEEGROUND_V9': '1', 'SEEGROUND_V9_SOURCE': 'r8',
            'SEEGROUND_R8_GROUNDING_REVISION': 'r13'})
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
            'predicted_id': oid, 'evidence': [], 'unresolved': []}
        if oid is not None:
            value['evidence'] = [{'source_quote': self.query, 'object_ids': [oid, 2],
                                  'observation': 'Visible chair beside the described desk'}]
        return {'_content': json.dumps(value), '_finish_reason': 'stop', '_baseline': {}, 'usage': {}}

    def run_actions(self, replies, limit=2):
        trace = {'steps': [], 'internal_usage': {'compute': 0, 'render': 0, 'visual_check': 0}}
        calls = []
        with tempfile.TemporaryDirectory() as directory, \
                patch('v74_grounding.scene_views', return_value=[{'image': 'overview.png'}, {'image': 'top.png'}]), \
                patch('r13_react.image_part', return_value={}), \
                patch('active_diagnostic.render_action', return_value={'image': 'reference.png',
                    'tool': 'render', 'focus_ids': [2], 'marked_ids': [2], 'context': 'isolated'}) as renderer, \
                patch('r13_react.chat', side_effect=replies) as model:
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
        self.assertEqual([o['tool'] for o in trace['r13_observations']], ['render', 'measure'])
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
        self.assertTrue(trace['r13_observations'][0]['no_evidence_acquired'])
        self.assertEqual(result['predicted_id'], 1)

    def test_large_requested_group_is_rendered_in_bounded_panels(self):
        self.objects = [prediction(i, ['chair', 'stool']) for i in range(1, 10)]
        result, trace, model, renderer = self.run_actions([
            self.action('inspect', list(range(1, 10))), self.action('finish', oid=1)])
        self.assertEqual(renderer.call_count, 2)
        self.assertEqual([call.args[0]['focus_ids'] for call in renderer.call_args_list],
                         [list(range(1, 7)), [7, 8, 9]])
        self.assertEqual(trace['r13_observations'][0]['focus_ids'], list(range(1, 10)))
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


if __name__ == '__main__':
    unittest.main()
