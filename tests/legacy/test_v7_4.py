"""Full-query coverage recovery; no evaluation target enters the model call."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from eviground.grounding.active_diagnostic import active_ground
from tests.legacy.test_v7_evidence import obj, parse
from eviground.grounding.v4_policy import settings, v73_enabled
from eviground.grounding.v74_grounding import ground_scene, needs_scene_reasoning, validate_decision


class V74Tests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {'EVIGROUND_V4': '1', 'EVIGROUND_V7': '1',
            'EVIGROUND_V7_4': '1', 'EVIGROUND_V7_3': '0', 'EVIGROUND_REACT_BACKEND': 'openai'})
        env.start(); self.addCleanup(env.stop)
        self.query = 'The box at the bottom of the stack'
        self.objects = [obj(3, 'box', xyz=(0, 0, 3)), obj(0, 'box', xyz=(0, 0, 1)),
                        obj(12, 'box', xyz=(0, 0, 2)), obj(18, 'table', xyz=(0, 0, .2))]
        self.parsed = parse(self.query, 'box', [], anchors=['stack'],
                            unhandled=['at the bottom of the stack'])
        self.parsed['_v4_budget'] = {'size': 0, 'relations': 0, 'review': 0}
        self.views = [{'image': 'test.png', 'marked_ids': [0, 3, 12, 18]}]
        self.reply = {'predicted_id': 0, 'reason': 'Box 0 is below both other boxes in this stack.',
            'evidence': [{'source_quote': self.query, 'object_ids': [0, 3, 12],
                          'observation': 'Same XY position; ID 0 has the lowest center and top.'}],
            'unresolved': []}
        self.cloud = np.array([[0., 0., 0.], [1., 1., 3.]])

    def answer(self, *args, **kwargs):
        return {'_content': json.dumps(self.reply), '_finish_reason': 'stop', 'usage': {},
                '_baseline': {'model': 'mock', 'latency_s': 0}}

    def test_version_inherits_parameter_contract(self):
        self.assertEqual(settings()['version'], 'relative-size-v7.4')
        self.assertTrue(v73_enabled())

    def test_source_bound_choice_and_invalid_counterexamples(self):
        self.assertEqual(validate_decision(self.reply, self.query, [0, 3, 12], {0, 3, 12}), self.reply)
        for changes in ({'predicted_id': 18}, {'predicted_id': False}, {'evidence': []},
                        {'evidence': [{**self.reply['evidence'][0], 'source_quote': 'not in query'}]},
                        {'evidence': [{**self.reply['evidence'][0], 'object_ids': [0, 999]}]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_decision({**self.reply, **changes}, self.query, [0, 3, 12], {0, 3, 12})
        validate_decision({**self.reply, 'predicted_id': None, 'evidence': []}, self.query, [0], {0})

    def test_unsupported_stack_keeps_multimodal_choice_instead_of_empty_rule_support(self):
        calls, trace = [], {}
        with tempfile.TemporaryDirectory() as directory, \
                patch('eviground.grounding.v74_grounding.scene_views', return_value=self.views), \
                patch('eviground.grounding.v74_grounding.image_part', return_value={'type': 'text', 'text': 'test image'}), \
                patch('eviground.grounding.v74_grounding.chat', side_effect=self.answer) as model, \
                patch('eviground.grounding.active_diagnostic.chat', side_effect=AssertionError('unsupported symbolic loop')):
            result = active_ground(self.query, self.parsed, self.objects, self.objects,
                self.objects[:3], self.cloud, np.zeros((2, 3), dtype=np.uint8), np.array([0, 3]),
                Path(directory), calls, trace, 3, True)
        self.assertEqual(result['predicted_id'], 0)
        self.assertEqual(result['prediction_source'], 'v74_multimodal_tentative')
        self.assertFalse(result['accepted'])
        self.assertEqual(model.call_count, 1)
        content = model.call_args.args[0][1]['content'][0]['text']
        payload = json.loads(content)
        evidence_schema = model.call_args.kwargs['response_schema']['schema']['properties']['evidence']
        self.assertEqual(evidence_schema['items']['properties']['source_quote']['enum'], [self.query])
        self.assertEqual(payload['allowed_target_ids'], [0, 3, 12])
        self.assertIn(18, [o['id'] for o in payload['objects']])
        for forbidden in ('target_id', 'correct', 'stimulus_candidate_ids'):
            self.assertNotIn(forbidden, payload)
        self.assertEqual(trace['planner_input'], 'text_and_scene_images')

    def test_empty_symbolic_pool_is_reconsidered_but_visual_exclusions_are_respected(self):
        self.assertTrue(needs_scene_reasoning({'candidate_ids': [], 'missing': []}))
        self.assertFalse(needs_scene_reasoning({'candidate_ids': [0, 3], 'missing': [
            'category/appearance evidence is uncertain or unreviewed']}))
        trace = {'steps': [], 'internal_usage': {'render': 0, 'visual_check': 0}, 'backtracks': []}
        with patch('eviground.grounding.v74_grounding.scene_views', return_value=self.views), \
                patch('eviground.grounding.v74_grounding.image_part', return_value={}), \
                patch('eviground.grounding.v74_grounding.chat', side_effect=self.answer) as model:
            result = ground_scene(self.query, self.parsed, self.objects, self.objects[:3],
                self.cloud, self.cloud, np.array([0, 3]), Path('.'), [],
                {'candidate_ids': [], 'missing': []}, trace, excluded_ids=[0])
        self.assertIsNone(result['predicted_id'])
        self.assertNotIn(0, model.call_args.kwargs['response_schema']['schema']['properties']['predicted_id']['enum'])
        with patch('eviground.grounding.v74_grounding.chat') as model:
            self.assertIsNone(ground_scene(self.query, self.parsed, self.objects, self.objects[:3],
                self.cloud, self.cloud, np.array([0, 3]), Path('.'), [], {}, trace, excluded_ids=[0, 3, 12]))
            model.assert_not_called()

    def test_model_stop_gets_scene_evidence_when_symbolic_support_is_empty(self):
        query = 'the box'
        parsed = parse(query, 'box', [])
        self.reply = {**self.reply, 'evidence': [{'source_quote': query, 'object_ids': [0],
                                              'observation': 'Observed box at the base.'}]}
        stopped = {'_content': json.dumps({'action': 'stop', 'reason': 'No symbolic discriminator'}),
                   '_finish_reason': 'stop', '_baseline': {}}
        with tempfile.TemporaryDirectory() as directory, \
                patch('eviground.grounding.active_diagnostic.chat', return_value=stopped) as planner, \
                patch('eviground.grounding.v74_grounding.scene_views', return_value=self.views), \
                patch('eviground.grounding.v74_grounding.image_part', return_value={}), \
                patch('eviground.grounding.v74_grounding.chat', side_effect=self.answer) as model:
            result = active_ground(query, parsed, self.objects, self.objects, self.objects[:3],
                self.cloud, self.cloud, np.array([0, 3]), Path(directory), [], {}, 3, True)
        self.assertEqual(result['predicted_id'], 0)
        self.assertEqual(planner.call_count, 1)
        self.assertEqual(model.call_count, 1)

    def test_local_render_keeps_named_reference_and_all_targets(self):
        from eviground.grounding.v74_grounding import scene_views
        parsed = {'Anchor': ['table']}
        points = np.array([o['bbox_3d'][:3] for o in self.objects])
        def rendered(xyz, rgb, path, **kwargs):
            return {'marked_ids': [int(i) for i in kwargs['ids'] if i >= 0]}
        with tempfile.TemporaryDirectory() as directory, patch('eviground.grounding.v74_grounding.render', side_effect=rendered):
            views = scene_views(parsed, self.objects, self.objects[:3], points, np.zeros((4, 3)),
                                np.array([3, 0, 12, 18]), Path(directory))
        self.assertEqual(set(views[1]['marked_ids']), {0, 3, 12, 18})


if __name__ == '__main__':
    unittest.main()
