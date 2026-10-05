import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from real_scene import run_query, parse_query


def reply(content, finish='stop'):
    return {'_content':content,'_baseline':{'model':'mock','latency_s':.02},
            '_finish_reason':finish,'usage':{'total_tokens':10}}


class RealSceneTests(unittest.TestCase):
    def test_diagnostic_query_is_separate_from_instructions_and_schema_has_scope(self):
        query='This trash can is next to a large black television.'
        answer={'Target':['trash can'],'Anchor':['tv'],'Relations':[],
                'Observer':{},'Attributes':[],'Unhandled':[]}
        with patch('real_scene.chat',return_value=reply(json.dumps(answer))) as api:
            parse_query(query,mode='diagnostic')
        messages=api.call_args.args[0]
        self.assertEqual(messages[0]['role'],'system')
        self.assertNotIn(query,messages[0]['content'])
        self.assertNotIn('fridge',messages[0]['content'])
        self.assertNotIn('the correct one facing the windows',messages[0]['content'])
        self.assertIn('scope',messages[0]['content'])
        self.assertIn('"Target":["category"]',messages[0]['content'])
        self.assertEqual(json.loads(messages[1]['content']),{'description':query})

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
            self.assertEqual(record['error_stage'],'parse_query')
            self.assertEqual(record['error_type'],'QwenAPIError')
            self.assertEqual(record['calls'][0]['response'],'bad json')
            self.assertFalse((out/'prediction.json').exists())

    def test_partial_schema_failure_reports_field_and_keeps_raw_answer(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.scene(root);out=root/'experiment'
            answer=reply('{"Observer":{},"Attributes":[],"Unhandled":[]}')
            with patch('real_scene.chat',return_value=answer):
                with self.assertRaisesRegex(ValueError,'Target must be an array/string'):
                    run_query(root,'window',out,mode='diagnostic')
            record=json.loads((out/'run.json').read_text())
            self.assertEqual(record['error_stage'],'parse_query')
            self.assertIn('top-level keys: Attributes, Observer, Unhandled',record['error'])
            self.assertEqual(record['calls'][0]['response'],answer['_content'])

    def test_server_reply_with_relation_but_missing_target_is_not_a_completed_prediction(self):
        answer={'Observer':{'origin':{'kind':'unspecified','category':'','phrase':''},
                            'facing':{'category':'','group':False,'phrase':''}},
                'Attributes':[], 'Unhandled':['The window nearest the front doors.'],
                'Relations':[{'target_category':'window','anchor_category':'front doors',
                              'relation':'nearest','scope':'','phrase':'nearest the front doors'}]}
        with patch('real_scene.chat',return_value=reply(json.dumps(answer))):
            with self.assertRaisesRegex(ValueError,'Target must be an array/string'):
                parse_query('The window nearest the front doors.',mode='diagnostic')

    def test_category_verdict_is_soft_and_camera_equals_ranked_mode(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.scene(root)
            parsed=reply('{"Target":["trash can"],"Anchor":["tv"],"Relations":[]}')
            with patch('real_scene.chat',return_value=parsed), \
                 patch('run.chat',return_value=reply('Predicted ID: 1')):
                run_query(root,'trash can by tv',root/'ranked',mode='ranked_geometry')
                with patch('category_review.chat',return_value=reply('{"id":1,"verdict":"no","reason":"unclear container"}')):
                    result=run_query(root,'trash can by tv',root/'checked',mode='category_review',review_limit=1)
            first=json.loads((root/'ranked/run.json').read_text())
            second=json.loads((root/'checked/run.json').read_text())
            self.assertEqual(first['target_ids'],second['target_ids'])
            self.assertEqual(first['anchor_ids'],second['anchor_ids'])
            self.assertEqual(first['camera_and_render'],second['camera_and_render'])
            self.assertEqual(second['category_review']['checks'][0]['verdict'],'no')
            self.assertEqual(len(result['images']),1)
            self.assertEqual(result['predicted_id'],1)
            self.assertEqual(result['api_calls'],3)
            self.assertFalse((root/'objects_gt.json').exists())

    def test_diagnostic_without_rgb_keeps_guess_and_missing_state(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.scene(root)
            parsed=reply(json.dumps({'Target':['water cooler'],'Anchor':['tv'],
                         'Relations':[{'target_category':'water cooler','anchor_category':'tv',
                                       'relation':'near','phrase':'by tv'}],
                         'Observer':{},'Attributes':[],'Unhandled':[]}))
            with patch('real_scene.chat',return_value=parsed), \
                 patch('rgb_review.chat') as review, \
                 patch('run.chat',return_value=reply('Predicted ID: 1')):
                result=run_query(root,'water cooler by tv',root/'diagnostic',mode='diagnostic',review_limit=1)
            review.assert_not_called()
            self.assertEqual(result['predicted_id'],1)
            self.assertEqual(result['decision_status'],'missing')
            self.assertFalse(result['accepted'])
            self.assertEqual(result['api_calls'],2)
            self.assertFalse((root/'objects_gt.json').exists())

    def test_diagnostic_model_disagreement_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.scene(root)
            from PIL import Image
            Image.new('RGB',(40,40),'white').save(root/'crop.png')
            (root/'rgb_crops.json').write_text(json.dumps({'scene':root.name,'crops':[
                {'id':1,'image':'crop.png','binding':'projected_instance','source_frame':'001.jpg'}]}))
            parsed=reply(json.dumps({'Target':['water cooler'],'Anchor':[],'Relations':[],
                                    'Observer':{},'Attributes':[],'Unhandled':[]}))
            with patch('real_scene.chat',return_value=parsed), \
                 patch('rgb_review.chat',return_value=reply('{"id":1,"verdict":"yes","reason":"visible"}')), \
                 patch('run.chat',return_value=reply('Predicted ID: 2')):
                result=run_query(root,'water cooler',root/'diagnostic',mode='diagnostic',review_limit=1)
            self.assertEqual(result['diagnosis']['status'],'resolved')
            self.assertEqual(result['decision_status'],'conflict')
            self.assertFalse(result['accepted'])
            self.assertEqual(result['api_calls'],3)

    def test_diagnostic_proposals_do_not_change_inspection_camera_or_labels(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.scene(root)
            # No anchor: the old diagnostic proposal mean used to shift this camera.
            parsed=reply(json.dumps({'Target':['water cooler'],'Anchor':[],'Relations':[],
                                    'Observer':{},'Attributes':[],'Unhandled':[]}))
            with patch('real_scene.chat',return_value=parsed), \
                 patch('run.chat',return_value=reply('Predicted ID: 1')):
                run_query(root,'water cooler',root/'ranked',mode='ranked_geometry')
                run_query(root,'water cooler',root/'diagnostic',mode='diagnostic',review_limit=2)
            first=json.loads((root/'ranked/run.json').read_text())
            second=json.loads((root/'diagnostic/run.json').read_text())
            self.assertEqual(second['review_proposal']['ids'],[1,2])
            self.assertEqual(first['target_ids'],second['target_ids'])
            self.assertEqual(first['camera_and_render'],second['camera_and_render'])
            self.assertEqual((root/'ranked/view.png').read_bytes(),(root/'diagnostic/view.png').read_bytes())


if __name__=='__main__':unittest.main()
