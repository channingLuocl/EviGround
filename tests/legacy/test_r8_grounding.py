"""Evidence and rendering contracts for the opt-in r8 grounding revision."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from eviground.categories.categories_v9 import R8_SCORE_TYPE
from eviground.grounding.r8_grounding import camera_eye, isolated_target_views, query_evidence, target_head_hint
from eviground.grounding.evidence import visual_fallback_candidates
from eviground.rendering.run import project
from tests.legacy.test_v7_evidence import obj
from eviground.grounding.v4_policy import settings
from eviground.grounding.v74_grounding import ground_scene, scene_views


def prediction(oid, names, xyz=(0, 0, 0)):
    return {**obj(oid, names[0], xyz=xyz), 'category_status': 'uncertain',
        'category_score_type': R8_SCORE_TYPE,
        'category_hypotheses': [{'category': name, 'score': 3-rank/4,
            'rank': rank, 'gap_from_top1': (rank-1)/4} for rank, name in enumerate(names, 1)]}


class R8GroundingTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {'EVIGROUND_V4': '1', 'EVIGROUND_V7': '1',
            'EVIGROUND_V7_4': '1', 'EVIGROUND_V9': '1', 'EVIGROUND_V9_SOURCE': 'r8',
            'EVIGROUND_R8_GROUNDING_REVISION': 'r2'})
        env.start(); self.addCleanup(env.stop)

    def test_opt_in_version_keeps_r1_available(self):
        self.assertEqual(settings()['version'], 'predicted-category-grounding-v9-r8-r2')
        with patch.dict(os.environ, {'EVIGROUND_R8_GROUNDING_REVISION': 'r1'}):
            self.assertEqual(settings()['version'], 'predicted-category-grounding-v9-r8-r1')

    def test_target_head_hint_uses_quoted_noun_and_preserves_later_requested_target(self):
        query = 'the desk in the middle of the room, with a covered white box on the edge of it'
        self.assertEqual(target_head_hint(query, ['desk', 'box']),
                         {'category': 'desk', 'source_quote': 'desk'})
        self.assertIsNone(target_head_hint('The bed with papers on it, the pillow is on the side.',
                                          ['bed', 'pillow']))
        self.assertIsNone(target_head_hint('Facing the desk, choose a box.', ['desk', 'box']))

    def test_review_can_correct_proposal_but_cannot_choose_excluded_ids(self):
        query = 'the chair on the left'
        candidates = [prediction(1, ['chair']), prediction(2, ['chair'])]
        def answer(oid):
            return {'_content': json.dumps({'predicted_id': oid, 'reason': 'Compared original conditions',
                'evidence': [{'source_quote': query, 'object_ids': [oid],
                              'observation': 'Observed target shape and position'}], 'unresolved': []}),
                '_finish_reason': 'stop', '_baseline': {}}
        for excluded in ([], [2]):
            trace = {'steps': [], 'backtracks': [], 'internal_usage': {'render': 0, 'visual_check': 0}}
            calls = []
            with tempfile.TemporaryDirectory() as directory, \
                    patch('eviground.grounding.v74_grounding.scene_views', return_value=[{'image': 'test.png'}]), \
                    patch('eviground.grounding.r8_grounding.isolated_target_views', return_value=[]), \
                    patch('eviground.grounding.v74_grounding.image_part', return_value={}), \
                    patch('eviground.grounding.v74_grounding.chat', side_effect=[answer(1), answer(2)]) as model:
                result = ground_scene(query, {'Target': ['chair'], '_v4_budget': {}}, candidates,
                    candidates, np.zeros((2, 3)), np.zeros((2, 3)), np.array([1, 2]),
                    Path(directory), calls, {}, trace, excluded_ids=excluded)
            self.assertEqual(result['predicted_id'], None if excluded else 2)
            self.assertEqual(trace['r8_initial_scene_decision']['predicted_id'], 1)
            self.assertEqual([c['stage'] for c in calls],
                             ['v74_scene_grounding', 'r8_scene_consistency_review'])
            self.assertEqual(model.call_count, 2)

    def test_query_bindings_keep_alternative_target_and_reference_without_irrelevant_labels(self):
        targets = [prediction(1, ['chair', 'office chair', 'table'], xyz=(2, 0, 0)),
                   prediction(2, ['clothes', 'chair', 'box'], xyz=(-2, 0, 0))]
        reference = prediction(3, ['table', 'desk', 'sofa'], xyz=(0, 0, 0))
        result = query_evidence({'Target': ['chair'], 'Anchor': ['desk']},
                                targets+[reference], targets)
        self.assertEqual([b['primary_match'] for b in result['target_category_bindings']], [True, False])
        self.assertEqual(result['target_category_bindings'][1]['matched_hypotheses'][0]['rank'], 2)
        self.assertEqual([h['category'] for h in result['objects'][1]['category_hypotheses']],
                         ['clothes', 'chair'])
        self.assertEqual([h['category'] for h in result['objects'][2]['category_hypotheses']],
                         ['table', 'desk'])
        self.assertEqual([row[0] for row in result['numeric_target_orders']['center_x_ascending']], [2, 1])
        distances = result['numeric_reference_distances'][0]
        self.assertEqual(distances['reference_id'], 3)
        self.assertEqual(distances['rows'][0][1], 2)
        self.assertNotIn('target_id', result)
        self.assertEqual(len(targets[1]['category_hypotheses']), 3)  # no mutation

    def test_camera_contains_every_box_corner(self):
        for direction in ([0., -1., 1.], [1., 1., .3], [0., 0., 1.]):
            lower, upper = np.array([-3., -2., 0.]), np.array([3., 2., 3.])
            eye, at = camera_eye(lower, upper, direction)
            corners = np.array([[x, y, z] for x in (lower[0], upper[0])
                for y in (lower[1], upper[1]) for z in (lower[2], upper[2])])
            uv, depth = project(corners, eye, at, 1000)
            self.assertTrue((depth > 0).all())
            self.assertTrue(((uv >= 0) & (uv < 1000)).all(), (direction, uv))

    def test_cutaway_uses_prediction_coverage_instead_of_gt_or_top1_shell_labels(self):
        objects = [prediction(1, ['wall', 'chair']),
                   prediction(2, ['desk', 'table'], xyz=(1, 1, 1)),
                   obj(99, 'object')]
        points = np.array([[0., 0., 0.], [1., 1., 1.], [20., 20., 20.]])
        def rendered(xyz, rgb, path, **kwargs):
            self.assertEqual(len(xyz), 2)
            self.assertNotIn(99, kwargs['ids'])
            self.assertIn(1, kwargs['ids'])  # predicted wall is an eligible chair
            return {'marked_ids': [1, 2]}
        with tempfile.TemporaryDirectory() as directory, patch('eviground.grounding.v74_grounding.render', side_effect=rendered):
            views = scene_views({'Anchor': ['desk']}, objects, objects[:1], points,
                                np.zeros((3, 3)), np.array([1, 2, 99]), Path(directory))
        self.assertEqual(views[0]['hidden_supplemental_instance_ids'], [99])

    def test_isolated_views_render_every_target_with_native_renderer(self):
        candidates = [prediction(1, ['chair']), prediction(2, ['desk'], xyz=(1, 1, 1))]
        points = np.array([[0., 0., 0.], [1., 1., 1.]])
        with tempfile.TemporaryDirectory() as directory:
            views = isolated_target_views(candidates, points, np.zeros((2, 3)),
                                          np.array([1, 2]), directory)
            self.assertTrue(Path(views[0]['image']).is_file())
            self.assertEqual(views[0]['marked_ids'], [1, 2])
            self.assertIn('independently framed', views[0]['layout'])

    def test_visual_fallback_is_bounded_and_marks_unknown_category(self):
        target = prediction(1, ['lamp'], xyz=(0, 0, 1))
        reference = prediction(2, ['lamp'], xyz=(0, 1, 1))
        nearby = prediction(3, ['table'], xyz=(0, 1.1, .5))
        distant = prediction(4, ['chair'], xyz=(20, 20, 1))
        result = visual_fallback_candidates(
            'looking at both lamps, choose the lamp on the right',
            {'Target': ['lamp'], 'Anchor': ['lamp']},
            [target, reference, nearby, distant], [target, reference], [target, reference], limit=1)
        self.assertEqual([item['bbox_id'] for item in result], [3])
        self.assertEqual(result[0]['candidate_role'], 'visual_fallback')
        self.assertEqual(visual_fallback_candidates(
            'choose the lamp', {'Target': ['lamp'], 'Anchor': []},
            [target, nearby], [target], [], limit=3), [])

    def test_visual_fallback_prioritizes_farthest_unknown_candidate(self):
        door = prediction(10, ['door'], xyz=(0, 0, 1))
        near = prediction(11, ['chair'], xyz=(1, 0, 1))
        far = prediction(12, ['monitor'], xyz=(8, 0, 1))
        result = visual_fallback_candidates(
            'the desk farthest from the door',
            {'Target': ['desk'], 'Anchor': ['door'],
             'Relations': [{'relation': 'farthest', 'anchor_category': 'door'}]},
            [door, near, far], [], [door], limit=1)
        self.assertEqual([item['bbox_id'] for item in result], [12])
        self.assertEqual(result[0]['fallback_reason'], 'query_extreme')

    def test_visual_fallback_includes_low_geometry_extreme(self):
        seed = prediction(10, ['towel'], xyz=(0, 0, 2))
        low = prediction(11, ['box'], xyz=(0, 0, .1))
        middle = prediction(12, ['chair'], xyz=(0, 0, 1))
        result = visual_fallback_candidates(
            'the lowest folded towel',
            {'Target': ['towel'], 'Anchor': [], 'Relations': []},
            [seed, low, middle], [seed], [], limit=1)
        self.assertEqual([item['bbox_id'] for item in result], [11])
        self.assertEqual(result[0]['fallback_reason'], 'query_extreme')


if __name__ == '__main__':
    unittest.main()
