"""Generic query nouns must not restrict targets to a literal classifier label."""
import json
import os
import unittest
from unittest.mock import patch

from r8_grounding import generic_target_pool, query_evidence
from r8_precision import precision_evidence
from real_scene import run_query
from test_categories_r8 import R8Tests
from test_r8_grounding import prediction
from test_v7_evidence import obj, parse


class GenericTargetTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {'SEEGROUND_V4': '1', 'SEEGROUND_V7': '1',
            'SEEGROUND_V7_4': '1', 'SEEGROUND_V9': '1', 'SEEGROUND_V9_SOURCE': 'r8',
            'SEEGROUND_R8_GROUNDING_REVISION': 'r12'})
        env.start(); self.addCleanup(env.stop)

    def test_only_nonspecific_target_names_enable_visual_pool(self):
        for names in (['object'], ['thing'], ['item'], ['white object']):
            self.assertTrue(generic_target_pool(names))
        for names in ([], ['chair'], ['object', 'chair'], ['picture frame']):
            self.assertFalse(generic_target_pool(names))
        with patch.dict(os.environ, {'SEEGROUND_R8_GROUNDING_REVISION': 'r10'}):
            self.assertFalse(generic_target_pool(['object']))

    def test_generic_payload_keeps_original_ranked_labels(self):
        objects = [prediction(1, ['chair', 'window']), prediction(2, ['desk', 'table'])]
        parsed = {'Target': ['object'], 'Anchor': []}
        result = query_evidence(parsed, objects, objects)
        compact = precision_evidence('the object on the left', parsed, objects, objects)
        self.assertIn('generic_target_visual_pool', result)
        self.assertEqual([h['category'] for h in compact['objects'][0]['category_hypotheses']],
                         ['chair', 'window'])
        self.assertEqual(objects[0]['category_hypotheses'][1]['rank'], 2)
        with patch.dict(os.environ, {'SEEGROUND_R8_GROUNDING_REVISION': 'r10'}):
            self.assertNotIn('generic_target_visual_pool', query_evidence(parsed, objects, objects))
            compact = precision_evidence('the object on the left', parsed, objects, objects)
            self.assertEqual(len(compact['objects'][0]['category_hypotheses']), 1)

    def test_generic_integration_admits_covered_targets_without_gt_or_supplemental_ids(self):
        fixture = R8Tests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        fixture.prepare()
        (fixture.scene/'scene_instances.json').write_text(json.dumps([obj(99, 'SECRET_GT_LABEL')]))
        query = 'the object at the bottom of the stack beside the desk'
        parsed = parse(query, 'object', [], anchors=['desk'], unhandled=['at the bottom of the stack'])
        reply = {'_content': json.dumps({'predicted_id': 1, 'reason': 'Visible object beside desk',
            'evidence': [{'source_quote': query, 'object_ids': [1, 2],
                'observation': 'Object at base beside desk'}], 'unresolved': [], 'numeric_choice': None}),
            '_finish_reason': 'stop', '_baseline': {}, 'usage': {}}
        with patch('real_scene.parse_query', return_value=parsed), \
                patch('wall_geometry.attach_wall_geometry', side_effect=lambda objects, *args: objects), \
                patch('v74_grounding.scene_views', return_value=[{'image': 'test.png'}]), \
                patch('r8_grounding.isolated_target_views', return_value=[]), \
                patch('v74_grounding.image_part', return_value={}), \
                patch('v74_grounding.chat', return_value=reply) as model:
            result = run_query(fixture.scene, query, fixture.root/'run',
                mode='active_diagnostic', object_table='v9_top5')
        payload = json.loads(model.call_args.args[0][1]['content'][0]['text'])
        self.assertEqual(result['predicted_id'], 1)
        self.assertEqual(result['category_target_ids'], [1, 2])
        self.assertEqual(payload['allowed_target_ids'], [1, 2])
        self.assertIn('generic_target_visual_pool', payload)
        self.assertNotIn('SECRET_GT_LABEL', json.dumps(payload))
        self.assertFalse(result['gt_categories_supplied_to_model'])
        self.assertFalse(result['query_target_id_supplied_to_model'])
        self.assertNotIn(99, payload['allowed_target_ids'])


if __name__ == '__main__':
    unittest.main()
