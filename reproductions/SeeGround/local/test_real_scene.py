import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from real_scene import run_query


def reply(content, finish='stop'):
    return {'_content':content,'_baseline':{'model':'mock','latency_s':.02},
            '_finish_reason':finish,'usage':{'total_tokens':10}}


class RealSceneTests(unittest.TestCase):
    def scene(self, root):
        objects=[{'bbox_id':1,'target':'water cooler','bbox_3d':[0,0,.5,.5,.5,1.]},
                 {'bbox_id':2,'target':'tv','bbox_3d':[2,0,1.,1.,.3,1.]}]
        (root/'objects.json').write_text(json.dumps(objects))
        rng=np.random.default_rng(3)
        xyz=np.vstack([rng.uniform(-.5,.5,(1000,3))*o['bbox_3d'][3:]+o['bbox_3d'][:3] for o in objects])
        np.savez(root/'scene.npz',xyz=xyz,rgb=np.full((2000,3),100,dtype=np.uint8),
                 ids=np.repeat([1,2],1000))

    def test_geometry_does_not_replace_missing_anchor_with_target(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.scene(root);out=root/'experiment'
            with patch('real_scene.chat',return_value=reply('{"Target":["water cooler"],"Anchor":["blackboard"]}')), \
                 patch('run.chat',return_value=reply('Predicted ID: 1', 'length')) as ground:
                result=run_query(root,'cooler by blackboard',out,mode='geometry')
            record=json.loads((out/'run.json').read_text())
            self.assertEqual(record['anchor_ids'],[])
            self.assertTrue(record['evidence']['anchor_missing'])
            self.assertEqual(record['evidence']['geometry']['pairs'],[])
            self.assertIn('measured distances',ground.call_args.args[0][0]['content'][-1]['text'])
            self.assertEqual(result['finish_reason'],'length')
            self.assertEqual(result['api_calls'],2)
            self.assertGreater(result['pipeline_latency_s'],0)

    def test_visual_review_recovers_mislabeled_instance_without_gt(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.scene(root);out=root/'experiment'
            with patch('real_scene.chat',return_value=reply('{"Target":["trash can"],"Anchor":["television"]}')), \
                 patch('visual_review.chat',return_value=reply('{"TargetIDs":[1],"AnchorIDs":[2],"UnknownIDs":[],"Observations":[]}')) as review, \
                 patch('run.chat',return_value=reply('Predicted ID: 1')):
                result=run_query(root,'trash can by television',out,mode='visual_review')
            record=json.loads((out/'run.json').read_text())
            self.assertEqual(record['category_target_ids'],[])
            self.assertEqual(record['target_ids'],[1])
            self.assertEqual(record['anchor_ids'],[2])
            self.assertEqual(len(result['images']),4)
            self.assertEqual(result['api_calls'],3)
            self.assertFalse((root/'objects_gt.json').exists())
            self.assertTrue(review.call_args.args[0][0]['content'][0]['image_url']['url'].startswith('data:image/png;base64,'))

    def test_parse_failure_keeps_call_record(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.scene(root);out=root/'experiment'
            with patch('real_scene.chat',return_value=reply('bad json')):
                with self.assertRaises(ValueError):
                    run_query(root,'test',out,mode='geometry')
            record=json.loads((out/'run.json').read_text())
            self.assertIn('error',record)
            self.assertEqual(record['calls'][0]['response'],'bad json')
            self.assertFalse((out/'prediction.json').exists())


if __name__=='__main__':unittest.main()
