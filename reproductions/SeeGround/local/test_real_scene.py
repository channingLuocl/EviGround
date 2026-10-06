import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
from real_scene import run_query, parse_query
from qwen_api import QwenAPIError


def reply(content, finish='stop'):
    return {'_content':content,'_baseline':{'model':'mock','latency_s':.02},
            '_finish_reason':finish,'usage':{'total_tokens':10}}


class RealSceneTests(unittest.TestCase):
    def setUp(self):
        env=patch.dict('os.environ',{'SEEGROUND_REACT_BACKEND':'qwen'})
        env.start()
        self.addCleanup(env.stop)

    def test_active_parser_repairs_missing_attribute_binding_once(self):
        answer={'Target':['cabinet'],'Anchor':[],'Relations':[],
                'Observer':{},'Attributes':[],'Unhandled':[]}
        calls=[]
        with patch('real_scene.chat',side_effect=[reply(json.dumps(answer)),
                reply(json.dumps({'Attributes':[{'category':'cabinet','phrase':'skinny'}]}))]) as api:
            result=parse_query('Choose the skinny cabinet.',mode='active_diagnostic',calls=calls)
        self.assertEqual(api.call_count,2)
        self.assertEqual(result['Attributes'],[{'category':'cabinet','phrase':'skinny'}])
        self.assertNotIn('appearance condition not represented: skinny',result['diagnostic_missing'])
        self.assertEqual(calls[1]['stage'],'parse_query_attribute_repair')
        from diagnostic import diagnose
        objects=[{'bbox_id':1,'target':'cabinet','bbox_3d':[0,0,.5,.4,.4,1.]}]
        diagnosis=diagnose(result,objects,objects,[0,0,.5],trusted_categories=True)
        self.assertEqual(diagnosis['visual_needs'][0]['id'],1)
        self.assertEqual(diagnosis['status'],'missing')

    def test_active_parser_repairs_modifier_used_as_attribute_category(self):
        answer={'Target':['cabinet'],'Anchor':[],'Relations':[],
                'Observer':{},'Attributes':[{'category':'skinny','phrase':'skinny'}],
                'Unhandled':[]}
        with patch('real_scene.chat',side_effect=[
                reply(json.dumps(answer)),
                reply(json.dumps({'Attributes':[{'category':'cabinet','phrase':'skinny'}]}))]) as api:
            result=parse_query('Choose the skinny cabinet.',mode='active_diagnostic')
        self.assertEqual(api.call_count,2)
        self.assertEqual(result['Attributes'],[{'category':'cabinet','phrase':'skinny'}])
        self.assertNotIn('appearance condition not represented: skinny',
                         result['diagnostic_missing'])

    def test_attribute_repair_keeps_prior_fields_and_rejects_invented_binding(self):
        answer={'Target':['cabinet'],'Anchor':[],'Relations':[],
                'Observer':{},'Attributes':[],'Unhandled':[]}
        for correction in ({'Attributes':[{'category':'chair','phrase':'skinny'}]},
                           {'Attributes':[{'category':'cabinet','phrase':'red'}]},
                           {'Attributes':[]}):
            with self.subTest(correction=correction), patch('real_scene.chat',side_effect=[
                    reply(json.dumps(answer)),reply(json.dumps(correction))]) as api:
                result=parse_query('skinny cabinet',mode='active_diagnostic')
            self.assertEqual(api.call_count,2)
            self.assertEqual(result['Attributes'],[])
            self.assertIn('appearance condition not represented: skinny',result['diagnostic_missing'])

    def test_active_parser_does_not_retry_complete_attributes(self):
        answer={'Target':['cabinet'],'Anchor':[],'Relations':[],
                'Observer':{},'Attributes':[{'category':'cabinet','phrase':'skinny'}],'Unhandled':[]}
        with patch('real_scene.chat',return_value=reply(json.dumps(answer))) as api:
            parse_query('skinny cabinet',mode='active_diagnostic')
        self.assertEqual(api.call_count,1)

    def test_attribute_repair_does_not_clear_other_parser_failures(self):
        answer={'Target':['cabinet'],'Anchor':['stove'],'Relations':[
            {'target_category':'cabinet','anchor_category':'stove',
             'relation':'other','phrase':'beside the stove'}],
            'Observer':{},'Attributes':[],'Unhandled':[]}
        with patch('real_scene.chat',side_effect=[reply(json.dumps(answer)),
                reply(json.dumps({'Attributes':[{'category':'cabinet','phrase':'skinny'}]}))]):
            result=parse_query('skinny cabinet beside the stove',mode='active_diagnostic')
        self.assertIn('unsupported relation: beside the stove',result['diagnostic_missing'])
        self.assertNotIn('appearance condition not represented: skinny',result['diagnostic_missing'])

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
        self.assertTrue(api.call_args.kwargs['json_mode'])

    def scene(self, root):
        objects=[{'bbox_id':1,'target':'water cooler','bbox_3d':[0,0,.5,.5,.5,1.]},
                 {'bbox_id':2,'target':'tv','bbox_3d':[2,0,1.,1.,.3,1.]}]
        (root/'objects.json').write_text(json.dumps(objects))
        rng=np.random.default_rng(3)
        xyz=np.vstack([rng.uniform(-.5,.5,(1000,3))*o['bbox_3d'][3:]+o['bbox_3d'][:3] for o in objects])
        np.savez(root/'scene.npz',xyz=xyz,rgb=np.full((2000,3),100,dtype=np.uint8),
                 ids=np.repeat([1,2],1000))

    def test_active_evidence_integrates_render_verification_and_saved_trace(self):
        parsed={'Target':['water cooler'],'Anchor':[],'Relations':[],
                'Observer':{},'Attributes':[{'category':'water cooler','phrase':'large'}],
                'Unhandled':[]}
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.scene(root);out=root/'active'
            (root/'objects_gt.json').write_text((root/'objects.json').read_text())
            with patch('real_scene.chat',return_value=reply(json.dumps(parsed))), \
                 patch('active_diagnostic.chat',side_effect=[
                     reply(json.dumps({'action':'evidence','dimension':'appearance',
                                       'context':'isolated','reason':'check quoted appearance'})),
                     reply(json.dumps({'checks':[{'id':1,'verdict':'yes',
                                                   'reason':'visible large cooler'}]})),
                     reply(json.dumps({'action':'finish','predicted_id':1,
                                       'reason':'appearance verified for the only candidate'})),
                 ]) as active:
                result=run_query(root,'large water cooler',out,mode='active_diagnostic',
                                 object_table='gt',review_limit=2)
            record=json.loads((out/'run.json').read_text())
            saved=json.loads((out/'prediction.json').read_text())
            self.assertEqual(result['api_calls'],4)
            self.assertTrue(saved['accepted'])
            self.assertEqual(saved['predicted_id'],1)
            self.assertFalse(saved['query_target_id_supplied_to_model'])
            self.assertIn('query_rewrite',record['parsed_query'])
            self.assertEqual([call['stage'] for call in record['calls']],
                             ['parse_query','active_decision',
                              'active_evidence_visual_check','active_decision'])
            trace=saved['active_trace']
            self.assertEqual(trace,record['active_diagnostic'])
            self.assertEqual(trace['interface'],'evidence_by_dimension')
            self.assertEqual(trace['internal_usage'],{'compute':1,'render':1,'visual_check':1})
            evidence=trace['steps'][0]['observation']
            self.assertEqual(evidence['coverage']['checked_ids'],[1])
            self.assertEqual(evidence['coverage']['unknown_ids'],[])
            self.assertTrue(Path(evidence['images'][0]).exists())
            content=active.call_args_list[1].args[0][1]['content']
            self.assertTrue(content[0]['image_url']['url'].startswith('data:image/png;base64,'))
            self.assertEqual(json.loads(content[1]['text'])['attributes'],parsed['Attributes'])

    def test_jev_plans_from_text_and_only_qwen_verifies_actual_render(self):
        parsed={'Target':['water cooler'],'Anchor':[],'Relations':[],
                'Observer':{},'Attributes':[{'category':'water cooler','phrase':'large'}],
                'Unhandled':[]}
        decisions=iter(['appearance_1_isolated_90','finish','1'])
        def jev_reply(url, **kwargs):
            self.assertEqual(url,'https://api.typesafe.ai/v1/systemone')
            payload=kwargs['json']
            self.assertNotIn('image_url',json.dumps(payload))
            self.assertNotIn('data:image',json.dumps(payload))
            self.assertNotIn('active_view_',json.dumps(payload))
            self.assertNotIn('target_id',payload['state'])
            choice=next(decisions)
            answers={key:{'type':'choice','choice':choice,'confidence':1.,
                          'probabilities':{option:float(option==choice) for option in q['criteria']}}
                     for key,q in payload['questions'].items()}
            return Mock(ok=True,json=lambda:{'model':'jev-fixture','answers':answers,
                                              'usage':{'input_tokens':3}})
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.scene(root);out=root/'jev_active'
            (root/'objects_gt.json').write_text((root/'objects.json').read_text())
            with patch.dict('os.environ',{'SEEGROUND_REACT_BACKEND':'jev',
                                         'SEEGROUND_JEV_MODEL':'jev-fixture','TYPESAFE_API_KEY':'test-only'}), \
                 patch('real_scene.chat',return_value=reply(json.dumps(parsed))) as parser, \
                 patch('decision_api.requests.post',side_effect=jev_reply) as jev, \
                 patch('active_diagnostic.chat',return_value=reply(json.dumps({'checks':[
                     {'id':1,'verdict':'yes','reason':'visible large cooler'}]}))) as visual:
                result=run_query(root,'large water cooler',out,mode='active_diagnostic',
                                 object_table='gt',review_limit=2)
            self.assertTrue(result['accepted'])
            self.assertEqual(result['model'],'jev-fixture')
            self.assertEqual(result['api_calls'],5)
            self.assertEqual(jev.call_count,3)
            self.assertEqual(parser.call_count,1)
            self.assertEqual(visual.call_count,1)
            self.assertTrue(visual.call_args.args[0][0]['content'].startswith('Verify only'))
            self.assertTrue(visual.call_args.args[0][1]['content'][0]['image_url']['url'].startswith(
                'data:image/png;base64,'))
            trace=result['active_trace']
            self.assertEqual(trace['decision_backend'],'jev')
            self.assertEqual(trace['planner_input'],'text_only')
            record=json.loads((out/'run.json').read_text())
            self.assertEqual([call['stage'] for call in record['calls']],
                             ['parse_query','active_decision','active_evidence_visual_check',
                              'active_decision','active_target_decision'])
            self.assertEqual(record['calls'][2]['backend'],'qwen')
            self.assertEqual(record['calls'][2]['role'],'visual_verifier')
            self.assertEqual(record['calls'][1]['answers']['next_task']['choice'],
                             'appearance_1_isolated_90')
            self.assertNotIn('test-only',json.dumps(record))
            self.assertTrue(Path(result['images'][0]).exists())

    def test_missing_jev_configuration_fails_before_any_model_call(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.scene(root);out=root/'missing_jev'
            with patch.dict('os.environ',{'SEEGROUND_REACT_BACKEND':'jev','TYPESAFE_API_KEY':''}), \
                 patch('real_scene.chat') as parser,patch('active_diagnostic.chat') as visual, \
                 patch('decision_api.requests.post') as jev:
                with self.assertRaisesRegex(ValueError,'TYPESAFE_API_KEY'):
                    run_query(root,'water cooler',out,mode='active_diagnostic')
            parser.assert_not_called();visual.assert_not_called();jev.assert_not_called()
            record=json.loads((out/'run.json').read_text())
            self.assertEqual(record['error_stage'],'react_configuration')
            self.assertEqual(record['calls'],[])
            self.assertFalse((out/'prediction.json').exists())

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

    def test_diagnostic_uses_structural_instances_as_reference_only(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.scene(root);out=root/'experiment'
            (root/'scene_instances.json').write_text(json.dumps([
                {'bbox_id':3,'target':'stove','bbox_3d':[-2,0,.5,.4,.4,1.]},
                {'bbox_id':4,'target':'wall','bbox_3d':[2,0,.5,.4,.4,1.]},
            ]))
            answer=reply(json.dumps({
                'Target':['water cooler'], 'Anchor':['stove','wall'],
                'Relations':[{'target_category':'water cooler',
                              'anchor_categories':['stove','wall'],
                              'relation':'between','scope':'relative_anchors',
                              'phrase':'between the stove and the wall'}],
                'Observer':{},'Attributes':[],'Unhandled':[]
            }))
            with patch('real_scene.chat',return_value=answer), \
                 patch('run.chat',return_value=reply('Predicted ID: 1')):
                run_query(root,'the water cooler between the stove and the wall',out,mode='diagnostic')
            record=json.loads((out/'run.json').read_text())
            diagnosis=record['diagnosis']
            self.assertEqual(record['category_target_ids'],[1])
            self.assertEqual(record['category_anchor_ids'],[3,4])
            self.assertEqual(diagnosis['geometry']['between_triplets'][0]['anchor_ids'],[3,4])
            self.assertEqual(record['target_ids'],[1])

    def test_predicted_diagnostic_does_not_propose_structural_targets(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.scene(root);out=root/'experiment'
            (root/'scene_instances.json').write_text(json.dumps([
                {'bbox_id':3,'target':'wall','bbox_3d':[2,0,.5,.4,.4,1.]},
            ]))
            answer=reply(json.dumps({
                'Target':['water cooler'], 'Anchor':['wall'],
                'Relations':[{'target_category':'water cooler',
                              'anchor_category':'wall', 'relation':'near',
                              'scope':'relative_anchor', 'phrase':'near the wall'}],
                'Observer':{},'Attributes':[],'Unhandled':[]
            }))
            with patch('real_scene.chat',return_value=answer), \
                 patch('run.chat',return_value=reply('Predicted ID: 1')):
                run_query(root,'the water cooler near the wall',out,mode='diagnostic',review_limit=3)
            record=json.loads((out/'run.json').read_text())
            self.assertEqual(record['category_anchor_ids'],[3])
            self.assertNotIn(3,record['review_proposal']['ids'])

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
            with patch('real_scene.chat',return_value=reply('bad json')) as api:
                with self.assertRaises(ValueError):
                    run_query(root,'test',out,mode='geometry')
            record=json.loads((out/'run.json').read_text())
            self.assertIn('error',record)
            self.assertEqual(record['error_stage'],'parse_query')
            self.assertEqual(record['error_type'],'QwenAPIError')
            self.assertEqual(record['calls'][0]['response'],'bad json')
            self.assertEqual(api.call_count,2)
            self.assertEqual(record['calls'][1]['stage'],'parse_query_json_repair')
            self.assertFalse((out/'prediction.json').exists())

    def test_invalid_json_is_repaired_once_and_calls_are_counted(self):
        malformed='{"Target":["water cooler"],"Anchor":[],"Attributes":["{"category":"water cooler","phrase":"large"}]}'
        fixed={'Target':['water cooler'],'Anchor':[],
               'Attributes':[{'category':'water cooler','phrase':'large'}]}
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.scene(root);out=root/'experiment'
            with patch('real_scene.chat',side_effect=[reply(malformed),reply(json.dumps(fixed))]) as api, \
                 patch('run.chat',return_value=reply('Predicted ID: 1')):
                result=run_query(root,'large water cooler',out,mode='diagnostic')
            record=json.loads((out/'run.json').read_text())
            self.assertEqual(result['api_calls'],3)
            self.assertEqual(record['calls'][0]['response'],malformed)
            self.assertIn('json_error',record['calls'][0])
            self.assertEqual(record['calls'][1]['stage'],'parse_query_json_repair')
            self.assertEqual(record['parsed_query']['Attributes'],fixed['Attributes'])
            repair_input=json.loads(api.call_args_list[1].args[0][1]['content'])
            self.assertEqual(repair_input,{'description':'large water cooler','model_response':malformed})
            self.assertTrue(all(c.kwargs['json_mode'] for c in api.call_args_list))

    def test_truncated_invalid_json_is_not_repaired(self):
        with patch('real_scene.chat',return_value=reply('{"Target":', 'length')) as api:
            with self.assertRaises(QwenAPIError):
                parse_query('water cooler',mode='diagnostic')
        self.assertEqual(api.call_count,1)

    def test_truncated_repair_is_rejected(self):
        calls=[]
        with patch('real_scene.chat',side_effect=[reply('bad json'),reply('{"Target":[],"Anchor":[]}', 'length')]) as api:
            with self.assertRaisesRegex(QwenAPIError,'JSON repair did not complete normally'):
                parse_query('water cooler',mode='diagnostic',calls=calls)
        self.assertEqual(api.call_count,2)
        self.assertEqual(calls[1]['finish_reason'],'length')

    def test_partial_schema_failure_reports_field_and_keeps_raw_answer(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.scene(root);out=root/'experiment'
            answer=reply('{"Observer":{},"Attributes":[],"Unhandled":[]}')
            with patch('real_scene.chat',return_value=answer) as api:
                with self.assertRaisesRegex(ValueError,'Target must be an array/string'):
                    run_query(root,'window',out,mode='diagnostic')
            record=json.loads((out/'run.json').read_text())
            self.assertEqual(record['error_stage'],'parse_query')
            self.assertIn('top-level keys: Attributes, Observer, Unhandled',record['error'])
            self.assertEqual(record['calls'][0]['response'],answer['_content'])
            self.assertEqual(api.call_count,1)

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
                 patch('run.chat',return_value=reply('Predicted ID: 1')) as ground:
                result=run_query(root,'water cooler by tv',root/'diagnostic',mode='diagnostic',review_limit=1)
            review.assert_not_called()
            self.assertEqual(result['predicted_id'],1)
            self.assertEqual(result['decision_status'],'missing')
            self.assertEqual(result['diagnosis_policy'],'audit_only')
            record=json.loads((root/'diagnostic/run.json').read_text())
            self.assertEqual(record['diagnosis_policy'],'audit_only')
            prompt=ground.call_args.args[0][0]['content'][-1]['text']
            self.assertIn('Do not use these sets to exclude, prioritize or break ties',prompt)
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

    def test_gt_table_retrieves_mislabeled_target_and_is_sent_to_ground(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.scene(root)
            gt=json.loads((root/'objects.json').read_text());gt[0]['target']='trash can'
            gt[0]['evaluation_only_fixture']='DO_NOT_SEND_QUERY_ANSWER'
            (root/'objects_gt.json').write_text(json.dumps(gt))
            answer=reply(json.dumps({'Target':['trash can'],'Anchor':['tv'],
                         'Relations':[{'target_category':'trash can','anchor_category':'tv',
                                       'relation':'nearest','phrase':'nearest tv'}],
                         'Observer':{},'Attributes':[],'Unhandled':[]}))
            with patch('real_scene.chat',return_value=answer), \
                 patch('rgb_review.chat') as review, \
                 patch('run.chat',return_value=reply('Predicted ID: 1')) as ground:
                pred=run_query(root,'trash can nearest tv',root/'pred',mode='diagnostic',object_table='pred',review_limit=1)
                result=run_query(root,'trash can nearest tv',root/'gt',mode='diagnostic',object_table='gt',review_limit=1)
            review.assert_not_called()
            record=json.loads((root/'gt/run.json').read_text())
            self.assertEqual(record['category_target_ids'],[1])
            self.assertTrue(record['gt_categories_supplied_to_model'])
            self.assertFalse(record['query_target_id_supplied_to_model'])
            self.assertEqual(result['objects_file'],'objects_gt.json')
            self.assertEqual(result['decision_status'],'resolved')
            self.assertEqual(result['diagnosis_policy'],'supported_unique_candidate')
            self.assertEqual(pred['decision_status'],'missing')
            prompt=ground.call_args.args[0][0]['content'][-1]['text']
            self.assertIn('Use the unique stable_candidate_ids entry as strong supporting evidence',prompt)
            self.assertNotIn('Diagnosis decision policy: audit_only',prompt)
            self.assertIn('"target": "trash can"',prompt)
            self.assertNotIn('water cooler',prompt)
            self.assertNotIn('DO_NOT_SEND_QUERY_ANSWER',prompt)
            self.assertNotIn('csv_row',prompt)

    def test_invalid_or_missing_table_fails_without_api_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.scene(root)
            with patch('real_scene.chat') as api:
                with self.assertRaises(ValueError):run_query(root,'query',root/'a',object_table='other')
                with self.assertRaises(FileNotFoundError):run_query(root,'query',root/'b',object_table='gt')
            api.assert_not_called()


if __name__=='__main__':unittest.main()
