"""Contracts for r3 numeric evidence and one-pass inference."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from r8_precision import frame_coordinates, precision_evidence
from test_r8_grounding import prediction
from v74_grounding import ground_scene
from v4_policy import settings


class PrecisionTests(unittest.TestCase):
    def test_right_is_perpendicular_to_forward_and_translation_invariant(self):
        targets = [prediction(1, ['chair'], (2, 1, 0)), prediction(2, ['chair'], (-1, 2, 0))]
        frame = frame_coordinates([0, 0], [0, 1], targets)
        self.assertEqual(frame['rows_left_to_right'], [[2, -1., 2.], [1, 2., 1.]])
        self.assertAlmostEqual(np.dot(frame['right_xy'], frame['forward_xy']), 0.)
        shifted = [{**o, 'bbox_3d': [o['bbox_3d'][0]+10, o['bbox_3d'][1]-3, *o['bbox_3d'][2:]]}
                   for o in targets]
        self.assertEqual(frame['rows_left_to_right'], frame_coordinates([10, -3], [10, -2], shifted)['rows_left_to_right'])
        self.assertIsNone(frame_coordinates([1, 1], [1, 1], targets))

    def test_named_height_and_reference_order_do_not_confuse_width(self):
        target = prediction(1, ['box'], (0, 0, 2))
        low = {**prediction(2, ['desk']), 'bbox_3d': [0, 0, .4, 5, 2, .8]}
        high = {**prediction(3, ['table', 'desk']), 'bbox_3d': [0, 0, .6, 1, 1, 1.2]}
        result = precision_evidence('box on the taller desk', {'Target': ['box'], 'Anchor': ['desk']},
                                    [target, low, high], [target])
        self.assertEqual(result['objects'][1]['height_m'], .8)
        self.assertEqual(result['objects'][1]['extent_x_m'], 5)
        self.assertEqual(result['numeric_reference_orders'][0]['height_ascending'], [[2, .8, 1], [3, 1.2, 2]])
        self.assertEqual(result['relative_geometry'][0]['rows'][0][-1], 1.)
        self.assertNotIn('score', result['objects'][0]['category_hypotheses'][0])
        self.assertNotIn('target_id', result)

    def test_explicit_observer_must_have_an_original_quote(self):
        objects = [prediction(1, ['chair'], (1, 2, 1)), prediction(2, ['door'], (0, 0, 1))]
        parsed = {'Target': ['chair'], 'Anchor': [], 'Observer': {
            'origin': {'category': 'door', 'phrase': 'at the door'}, 'facing': {}}}
        frames = precision_evidence('at the door facing the chairs', parsed, objects, objects[:1])['observer_frames']
        self.assertEqual(frames[0]['origin_id'], 2)
        self.assertEqual(precision_evidence('the chair', parsed, objects, objects[:1])['observer_frames'], [])

    def test_r3_has_no_automatic_consistency_call(self):
        query = 'the chair'
        target = prediction(1, ['chair'])
        answer = {'_content': json.dumps({'predicted_id': 1, 'reason': 'Visible chair.',
            'evidence': [{'source_quote': query, 'object_ids': [1], 'observation': 'Chair shape.'}],
            'unresolved': []}), '_finish_reason': 'stop', '_baseline': {}}
        env = {'SEEGROUND_V4': '1', 'SEEGROUND_V7': '1', 'SEEGROUND_V7_4': '1',
            'SEEGROUND_V9': '1', 'SEEGROUND_V9_SOURCE': 'r8', 'SEEGROUND_R8_GROUNDING_REVISION': 'r3'}
        trace = {'steps': [], 'backtracks': [], 'internal_usage': {'render': 0, 'visual_check': 0}}
        with patch.dict(os.environ, env), tempfile.TemporaryDirectory() as directory, \
                patch('v74_grounding.scene_views', return_value=[{'image': 'test.png'}]), \
                patch('r8_grounding.isolated_target_views', return_value=[]), \
                patch('v74_grounding.image_part', return_value={}), \
                patch('v74_grounding.chat', return_value=answer) as model:
            result = ground_scene(query, {'Target': ['chair'], '_v4_budget': {}}, [target], [target],
                np.zeros((1, 3)), np.zeros((1, 3)), np.array([1]), Path(directory), [], {}, trace)
            self.assertEqual(settings()['version'], 'predicted-category-grounding-v9-r8-r3')
        self.assertEqual(model.call_count, 1)
        self.assertEqual(result['predicted_id'], 1)
        self.assertEqual(result['final_review']['status'], 'included_in_scene_grounding')


if __name__ == '__main__':
    unittest.main()
