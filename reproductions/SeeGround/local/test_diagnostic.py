import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from diagnostic import validate_structure, diagnose
from evidence import validate_distance_relations
from rgb_review import load_crops, input_hashes, selective_review


def obj(oid, category, x, y):
    return {'bbox_id': oid, 'target': category, 'bbox_3d': [x, y, .5, .4, .4, 1.]}


def parsed(query, relations, origin=None, facing=None, attrs=None):
    data={'Target':['window'], 'Anchor':[], 'Relations':relations,
          'Observer':{'origin':origin or {}, 'facing':facing or {}}, 'Attributes':attrs or [], 'Unhandled':[]}
    return validate_structure(query, validate_distance_relations(query, data))


def supported(objects):
    return [{'id':o['bbox_id'], 'category':o['target'], 'verdict':'yes'} for o in objects]


def response(oid, verdict='yes', finish='stop'):
    return {'_content':json.dumps({'id':oid,'verdict':verdict,'reason':'visible evidence'}),
            '_baseline':{'model':'mock','latency_s':.01}, '_finish_reason':finish, 'usage':{}}


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.objects=[obj(1,'window',-1,1),obj(2,'window',1,1),obj(3,'tv',0,3)]
        self.query='Standing in the middle of the room facing the tv, the window on the right.'
        self.direction={'target_category':'window','anchor_category':'','relation':'right',
                        'scope':'side','phrase':'on the right'}
        self.origin={'kind':'room_center','category':'','phrase':'middle of the room'}
        self.facing={'category':'tv','group':False,'phrase':'facing the tv'}

    def diagnosis(self, objects=None, relation=None, origin=None, facing=None, checks=None):
        objects=self.objects if objects is None else objects
        data=parsed(self.query,[relation or self.direction],self.origin if origin is None else origin,
                    self.facing if facing is None else facing)
        return diagnose(data,objects,objects[:2],[0,0,.5],supported(objects) if checks is None else checks)

    def test_direction_reversal_and_rotation_invariance(self):
        self.assertEqual(self.diagnosis()['stable_candidate_ids'],[2])
        reverse=copy.deepcopy(self.objects);reverse[2]['bbox_3d'][1]=-3
        self.assertEqual(self.diagnosis(reverse)['stable_candidate_ids'],[1])
        rotated=copy.deepcopy(self.objects)
        for o in rotated:
            x,y=o['bbox_3d'][:2];o['bbox_3d'][:2]=[-y,x]
        self.assertEqual(self.diagnosis(rotated)['stable_candidate_ids'],[2])

    def test_missing_origin_never_becomes_room_center(self):
        result=self.diagnosis(origin={})
        self.assertEqual(result['status'],'missing')
        self.assertIsNone(result['hypotheses'][0]['observer'])
        self.assertEqual(result['candidate_ids'],[1,2])

    def test_multiple_facing_instances_are_shared_hypotheses_not_votes(self):
        objects=self.objects+[obj(4,'tv',0,-3)]
        result=self.diagnosis(objects)
        self.assertEqual(result['status'],'ambiguous')
        self.assertEqual(result['stable_candidate_ids'],[])
        self.assertEqual(result['candidate_ids'],[1,2])
        self.assertFalse(result['hypotheses_are_votes'])

    def test_multiple_named_origins_are_not_averaged(self):
        objects=self.objects+[obj(4,'door',-4,0),obj(5,'door',4,0)]
        query='Standing at the door facing the tv, the window on the right.'
        data=parsed(query,[self.direction],{'kind':'category','category':'door','phrase':'at the door'},self.facing)
        result=diagnose(data,objects,objects[:2],[0,0,.5],supported(objects))
        self.assertEqual({h['observer']['origin_id'] for h in result['hypotheses']},{4,5})

    def test_relative_direction_is_not_peer_extreme(self):
        objects=[obj(1,'window',1,1),obj(2,'window',2,1),obj(3,'tv',0,3),obj(4,'chair',0,1)]
        relation={**self.direction,'scope':'relative_anchor','anchor_category':'chair','phrase':'on the right of the chair'}
        data=parsed(self.query+' on the right of the chair',[relation],self.origin,self.facing)
        result=diagnose(data,objects,objects[:2],[0,0,.5],supported(objects))
        self.assertEqual(result['candidate_ids'],[1,2])
        self.assertEqual(result['status'],'ambiguous')

    def test_two_constraints_are_anded_in_same_frame(self):
        nearest={'target_category':'window','anchor_category':'chair','relation':'nearest','phrase':'nearest the chair'}
        objects=self.objects+[obj(4,'chair',-1,1.2)]
        data=parsed(self.query+' nearest the chair',[self.direction,nearest],self.origin,self.facing)
        result=diagnose(data,objects,objects[:2],[0,0,.5],supported(objects))
        self.assertEqual(result['candidate_ids'],[])
        self.assertEqual(result['status'],'conflict')

    def test_unknown_is_eligible_and_yes_does_not_beat_unknown(self):
        data=parsed('window',[])
        result=diagnose(data,self.objects,self.objects[:2],[0,0,.5],supported(self.objects[:1]))
        self.assertEqual(result['candidate_ids'],[1,2])
        self.assertEqual(result['status'],'missing')
        result=diagnose(data,self.objects,self.objects[:2],[0,0,.5],supported(self.objects[:1])+[
            {'id':2,'category':'window','verdict':'no'}])
        self.assertEqual(result['status'],'resolved')

    def test_near_and_unhandled_conditions_cannot_resolve(self):
        relation={'target_category':'window','anchor_category':'tv','relation':'near','phrase':'by the tv'}
        data=parsed('window by the tv',[relation])
        self.assertEqual(diagnose(data,self.objects,self.objects[:1],[0,0,.5],supported(self.objects))['status'],'missing')
        data=parsed('the large window',[])
        self.assertIn('appearance condition not represented: large',data['diagnostic_missing'])

    def test_no_facing_category_cannot_resolve_direction(self):
        checks=supported(self.objects[:2])+[{'id':3,'category':'tv','verdict':'no'}]
        self.assertEqual(self.diagnosis(checks=checks)['status'],'missing')

    def test_invalid_group_and_room_center_quotes_are_rejected(self):
        data=parsed(self.query,[self.direction],self.origin,{**self.facing,'group':True})
        self.assertEqual(data['Observer']['facing'],{})
        data=parsed('middle table facing the tv on the right',[self.direction],
                    {**self.origin,'phrase':'middle table'},self.facing)
        self.assertEqual(data['Observer']['origin'],{})

    def test_hypothesis_budget_does_not_make_a_truncated_claim(self):
        objects=self.objects+[obj(i,'door',i,0) for i in range(4,304)]
        relation={'target_category':'window','anchor_category':'door','relation':'nearest','phrase':'nearest door'}
        data=parsed('window nearest door',[relation])
        result=diagnose(data,objects,objects[:2],[0,0,.5],supported(objects))
        self.assertEqual(result['hypotheses'],[])
        self.assertEqual(result['status'],'missing')

    def test_null_category_binding_and_omitted_extreme_stay_missing(self):
        data=parsed('window nearest door',[{'target_category':'window','anchor_category':None,
                                          'relation':'nearest','phrase':'nearest door'}])
        self.assertEqual(data['Relations'],[])
        self.assertEqual(diagnose(data,self.objects,self.objects[:1],[0,0,.5],supported(self.objects))['status'],'missing')
        data=parsed('window nearest door',[])
        self.assertIn('distance extreme condition not represented',data['diagnostic_missing'])

    def test_hallucinated_unhandled_conditions_are_not_sent_as_query_evidence(self):
        query='window'
        raw={'Target':['window'],'Anchor':[],'Relations':[], 'Observer':{},'Attributes':[],
             'Unhandled':['face the fridge and move to a corner']}
        result=validate_structure(query,raw)
        self.assertEqual(result['Unhandled'],[])
        self.assertNotIn('fridge',str(result))
        self.assertTrue(result['diagnostic_missing'])

    def test_quoted_unhandled_condition_is_preserved(self):
        raw={'Target':['window'],'Anchor':[],'Relations':[], 'Observer':{},'Attributes':[],
             'Unhandled':['on the wall']}
        self.assertEqual(validate_structure('window on the wall',raw)['Unhandled'],['on the wall'])

    def test_plural_category_quote_is_valid_for_group_facing(self):
        data=parsed('middle of the room facing the windows on the right',[self.direction],
                    self.origin,{'category':'windows','group':True,'phrase':'facing the windows'})
        self.assertEqual(data['Observer']['facing']['category'],'windows')

    def test_anaphoric_target_is_explicitly_unresolved(self):
        raw={'Target':['one'],'Anchor':['cabinet'],'Relations':[], 'Observer':{},'Attributes':[], 'Unhandled':[]}
        data=validate_structure('facing the cabinets the one on the right',raw)
        self.assertIn('target category is an unresolved pronoun or generic noun',data['diagnostic_missing'])

    def test_annotated_categories_are_known_but_attributes_are_not(self):
        objects=[obj(1,'window',1,0),obj(2,'refrigerator',2,0)]
        data=parsed('window',[])
        result=diagnose(data,objects,objects,[0,0,.5],trusted_categories=True)
        self.assertEqual(result['candidate_ids'],[1])
        self.assertEqual(result['status'],'resolved')
        self.assertEqual(result['visual_needs'],[])
        data=parsed('black window',[],attrs=[{'category':'window','phrase':'black'}])
        result=diagnose(data,objects,objects,[0,0,.5],trusted_categories=True)
        self.assertEqual(result['candidate_ids'],[1])
        self.assertEqual(result['status'],'missing')
        self.assertEqual(result['visual_needs'][0]['attributes'],['black'])
        checks=[{'id':1,'category':'window','verdict':'yes'},{'id':2,'category':'window','verdict':'yes'}]
        result=diagnose(data,objects,objects,[0,0,.5],checks,trusted_categories=True)
        self.assertEqual(result['candidate_ids'],[1])
        self.assertEqual(result['status'],'resolved')


class RGBReviewTests(unittest.TestCase):
    def fixture(self, root):
        objects=[obj(1,'window',1,0),obj(2,'window',2,0)]
        Image.new('RGB',(40,40),'white').save(root/'crop.png')
        manifest={'scene':root.name,'crops':[{'id':1,'image':'crop.png','source_frame':'0001.jpg','binding':'projected_instance'}]}
        (root/'rgb_crops.json').write_text(json.dumps(manifest))
        return objects,manifest

    def test_no_rgb_causes_no_extra_api_call(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);objects=[obj(1,'window',1,0)]
            with patch('rgb_review.chat') as api:
                result=selective_review(parsed('window',[]),objects,objects,[0,0,.5],root,[],2)
            api.assert_not_called()
            self.assertEqual(result['status'],'missing')
            self.assertEqual(result['actions'][0]['action'],'skip_rgb_review')

    def test_supplied_rgb_can_support_a_scoped_singleton(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);objects,_=self.fixture(root);calls=[]
            with patch('rgb_review.chat',return_value=response(1)) as api:
                result=selective_review(parsed('window',[]),objects,objects[:1],[0,0,.5],root,calls,2)
            self.assertEqual(api.call_count,1)
            self.assertEqual(result['status'],'resolved')
            self.assertEqual(result['stable_candidate_ids'],[1])
            self.assertEqual(calls[0]['stage'],'rgb_review')

    def test_uncertain_or_bad_binding_answer_not_repeated(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);objects,_=self.fixture(root)
            for answer in (response(1,'uncertain'),response(2),response(1,finish='length')):
                with patch('rgb_review.chat',return_value=answer) as api:
                    result=selective_review(parsed('window',[]),objects,objects[:1],[0,0,.5],root,[],6)
                self.assertEqual(api.call_count,1)
                self.assertEqual(result['status'],'missing')
                self.assertEqual(result['checks'][0]['verdict'],'uncertain')

    def test_rejects_unknown_id_wrong_scene_or_path_escape(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);objects,manifest=self.fixture(root)
            for change in ({'id':99},{'image':'../crop.png'},{'binding':'pointcloud_render'}):
                updated=copy.deepcopy(manifest);updated['crops'][0].update(change)
                (root/'rgb_crops.json').write_text(json.dumps(updated))
                with self.assertRaises(ValueError):load_crops(root,objects)
            manifest['scene']='other';(root/'rgb_crops.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):load_crops(root,objects)

    def test_crop_hash_changes_with_image_not_gt(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);objects,_=self.fixture(root)
            first=input_hashes(root,objects)
            (root/'objects_gt.json').write_text('do not read')
            self.assertEqual(first,input_hashes(root,objects))
            Image.new('RGB',(40,40),'black').save(root/'crop.png')
            self.assertNotEqual(first,input_hashes(root,objects))


if __name__=='__main__':unittest.main()
