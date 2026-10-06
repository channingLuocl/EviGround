import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from diagnostic import recover_between_relations, recover_distance_relations, validate_structure, diagnose
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
    def test_size_recovery_preserves_compound_and_negated_conditions(self):
        for text in ('Choose the smallest red cabinet.',
                     'Choose the tallest cabinet near the door.',
                     'Choose the cabinet that is not the smallest.'):
            checked = validate_structure(text, {
                'Target': ['cabinet'], 'Anchor': [], 'Relations': [],
                'Observer': {}, 'Attributes': [], 'Unhandled': [text],
            })
            self.assertIn('unhandled condition: ' + text, checked['diagnostic_missing'])
            self.assertEqual(checked['Attributes'], [])

    def test_only_fully_covered_attribute_requests_clear_unhandled(self):
        for text, clear in [('choose the tall skinny shelf', True),
                            ('The tall skinny shelf.', True),
                            ('choose the tall skinny shelf near the door', False),
                            ('choose the not tall skinny shelf', False),
                            ('choose the tall skinny red shelf', False)]:
            checked = validate_structure(text, {
                'Target':['shelf'], 'Anchor':[], 'Relations':[], 'Observer':{},
                'Attributes':[{'category':'shelf','phrase':'tall'},
                              {'category':'shelf','phrase':'skinny'}], 'Unhandled':[text]})
            self.assertEqual(checked['Unhandled'], [] if clear else [text])

    def test_attribute_noun_phrase_is_not_an_unsupported_relation(self):
        for phrase, keep_error in [('tall skinny shelf', False),
                                   ('shelf beside the door', True)]:
            checked = validate_structure('choose the tall skinny shelf beside the door', {
                'Target': ['shelf'], 'Anchor': [], 'Observer': {}, 'Unhandled': [],
                'Attributes': [{'category': 'shelf', 'phrase': 'tall skinny'}],
                'Relations': [{'relation': 'other', 'target_category': 'shelf', 'phrase': phrase}],
            })
            self.assertEqual('unsupported relation: ' + phrase in checked['diagnostic_missing'],
                             keep_error)

    def test_shape_word_in_relation_quote_does_not_cover_missing_attribute(self):
        query = 'Choose the skinny cabinet between the stove and the wall.'
        relation = {'target_category': 'cabinet', 'anchor_categories': ['stove', 'wall'],
                    'relation': 'between', 'scope': '',
                    'phrase': 'the skinny cabinet between the stove and the wall'}
        data = {'Target': ['cabinet'], 'Anchor': ['stove', 'wall'], 'Relations': [relation],
                'Observer': {}, 'Attributes': [], 'Unhandled': []}
        checked = validate_structure(query, data)
        self.assertIn('appearance condition not represented: skinny', checked['diagnostic_missing'])
        objects = [obj(1, 'cabinet', 0, 0), obj(2, 'stove', -1, 0), obj(3, 'wall', 1, 0)]
        result = diagnose(checked, objects, objects[:1], [0, 0, .5], trusted_categories=True)
        self.assertEqual(result['candidate_ids'], [1])
        self.assertEqual(result['status'], 'missing')
        data['Attributes'] = [{'category': 'cabinet', 'phrase': 'skinny'}]
        checked = validate_structure(query, data)
        self.assertNotIn('appearance condition not represented: skinny', checked['diagnostic_missing'])
        result = diagnose(checked, objects, objects[:1], [0, 0, .5], trusted_categories=True)
        self.assertEqual(result['status'], 'missing')
        self.assertTrue(result['visual_needs'])

    def test_attribute_modifier_cannot_be_used_as_its_own_category(self):
        checked = validate_structure('Choose the skinny cabinet.', {
            'Target': ['cabinet'], 'Anchor': [], 'Relations': [], 'Observer': {},
            'Attributes': [{'category': 'skinny', 'phrase': 'skinny'}], 'Unhandled': [],
        })
        self.assertEqual(checked['Attributes'], [])
        self.assertIn('appearance condition not represented: skinny',
                      checked['diagnostic_missing'])

    def test_containment_cannot_resolve_even_with_positive_appearance_check(self):
        for verb in ('includes', 'contains', 'including', 'containing'):
            with self.subTest(verb=verb):
                query = 'counter ' + verb + ' the sink'
                data = validate_structure(query, {
                    'Target': ['counter'], 'Anchor': [], 'Relations': [], 'Observer': {},
                    'Attributes': [{'category': 'counter', 'phrase': verb + ' the sink'}], 'Unhandled': [],
                })
                objects = [obj(1, 'counter', 0, 0)]
                result = diagnose(data, objects, objects, [0, 0, .5], supported(objects), trusted_categories=True)
                self.assertEqual(result['status'], 'missing')
                self.assertIn('unsupported containment condition: ' + verb + ' the sink', result['missing'])
                self.assertEqual(result['candidate_ids'], [1])

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

    def test_recover_explicit_superlative_when_parser_omits_relation(self):
        raw={'Target':['monitor'],'Anchor':['window'],'Relations':[],
             'Observer':{},'Attributes':[],'Unhandled':[]}
        query='The monitor furthest away from the window.'
        recovered=recover_distance_relations(query,raw)
        data=validate_structure(query, validate_distance_relations(query,recovered))
        self.assertEqual(data['Relations'][0]['relation'],'farthest')
        self.assertEqual(data['Relations'][0]['anchor_category'],'window')
        self.assertNotIn('distance extreme condition not represented',data['diagnostic_missing'])

    def test_comparative_cannot_become_scene_wide_superlative(self):
        raw={'Target':['chair'],'Anchor':['window'],'Relations':[],
             'Observer':{},'Attributes':[],'Unhandled':[]}
        query='Of the two chairs, choose the one further from the window.'
        recovered=recover_distance_relations(query,raw)
        self.assertEqual(recovered['Relations'],[])
        # Even a model-proposed farthest operator must not rank the whole scene.
        raw['Relations']=[{'target_category':'chair','anchor_category':'window',
                          'relation':'farthest','phrase':query}]
        data=validate_structure(query, validate_distance_relations(query,raw))
        self.assertEqual(data['Relations'],[])

    def test_recovery_requires_exact_simple_binding_and_preserves_raw_input(self):
        raw={'Target':['chair'],'Anchor':['window'],'Relations':[]}
        for query in ('The table closest to the window.',
                      'The chair beside the table closest to the window.',
                      'The chair closest to the window on the right.',
                      'The chair not closest to the window.'):
            with self.subTest(query=query):
                self.assertEqual(recover_distance_relations(query,raw)['Relations'],[])
        for query,kind in (('The chair closest to the window.','nearest'),
                           ('The chair nearest the window.','nearest'),
                           ('The chair farthest from the window.','farthest'),
                           ('The chair furthest away from the window.','farthest')):
            result=recover_distance_relations(query,raw)
            self.assertEqual(result['Relations'][0]['relation'],kind)
            self.assertEqual(result['recovered_distance_relations'],result['Relations'])
        self.assertEqual(raw['Relations'],[])

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
        data=parsed('black window',[],attrs=[{'category':'window','phrase':'black'}])
        result=diagnose(data,objects,objects,[0,0,.5],trusted_categories=True)
        self.assertEqual(result['candidate_ids'],[1])
        self.assertEqual(result['status'],'missing')
        self.assertEqual(result['visual_needs'][0]['attributes'],['black'])
        checks=[{'id':1,'category':'window','verdict':'yes'},{'id':2,'category':'window','verdict':'yes'}]
        result=diagnose(data,objects,objects,[0,0,.5],checks,trusted_categories=True)
        self.assertEqual(result['candidate_ids'],[1])
        self.assertEqual(result['status'],'resolved')

    def test_vertical_relation_is_recovered_and_filters_candidates(self):
        objects=[obj(1,'cabinet',0,0), obj(2,'cabinet',0,0), obj(3,'refrigerator',0,0)]
        objects[1]['bbox_3d'][2] = 2.0
        raw={'Target':['cabinet'],'Anchor':['refrigerator'],'Relations':[],
             'Observer':{},'Attributes':[],'Unhandled':[]}
        data=validate_structure('cabinet above the refrigerator', raw)
        self.assertEqual(data['Relations'][0]['relation'], 'above')
        result=diagnose(data, objects, objects[:2], [0,0,.5], supported(objects),
                        trusted_categories=True)
        self.assertEqual(result['candidate_ids'], [2])
        self.assertEqual(result['geometry']['relation_pairs'][0]['relation'], 'above')

    def test_vertical_overlap_does_not_count_as_above(self):
        objects=[obj(1,'cabinet',0,0), obj(2,'cabinet',0,0), obj(3,'refrigerator',0,0)]
        objects[1]['bbox_3d'][2] = 0.8
        raw={'Target':['cabinet'],'Anchor':['refrigerator'],'Relations':[],
             'Observer':{},'Attributes':[],'Unhandled':[]}
        data=validate_structure('cabinet above the refrigerator', raw)
        result=diagnose(data, objects, objects[:2], [0,0,.5], supported(objects),
                        trusted_categories=True)
        self.assertEqual(result['candidate_ids'], [])
        pair=next(p for p in result['geometry']['relation_pairs'] if p['target_id'] == 2)
        self.assertLess(pair['target_bottom_minus_anchor_top_m'], 0.)

    def test_vertical_relation_allows_small_horizontal_gap(self):
        objects=[obj(1,'cabinet',.7,0), obj(2,'sink',0,0)]
        objects[0]['bbox_3d'][2] = 2.0
        raw={'Target':['cabinet'],'Anchor':['sink'],'Relations':[
            {'target_category':'cabinet','anchor_category':'sink','relation':'above',
             'scope':'relative_anchor','phrase':'above the sink'}],
             'Observer':{},'Attributes':[],'Unhandled':[]}
        data=validate_structure('cabinet above the sink', raw)
        result=diagnose(data, objects, objects[:1], [0,0,.5], supported(objects),
                        trusted_categories=True)
        self.assertEqual(result['candidate_ids'], [1])
        pair=result['geometry']['relation_pairs'][0]
        self.assertAlmostEqual(pair['horizontal_gap_m'], .3, places=6)
        self.assertTrue(pair['horizontal_alignment_passes'])
        self.assertTrue(pair['vertical_separation_passes'])
        self.assertEqual(result['geometry']['vertical_policy']['horizontal_gap_max_m'], .5)

    def test_between_relation_uses_two_reference_centers(self):
        objects=[obj(1,'cabinet',0,0), obj(2,'stove',-2,0), obj(3,'fridge',2,0)]
        raw={'Target':['cabinet'],'Anchor':['stove','fridge'],'Relations':[],
             'Observer':{},'Attributes':[],'Unhandled':[]}
        data=validate_structure('cabinet between the stove and the fridge',
                                recover_between_relations('cabinet between the stove and the fridge', raw))
        result=diagnose(data, objects, objects[:1], [0,0,.5], supported(objects),
                        trusted_categories=True)
        self.assertEqual(result['candidate_ids'], [1])
        evidence=result['geometry']['between_triplets'][0]
        self.assertAlmostEqual(evidence['projection_t'], .5, places=6)
        self.assertTrue(evidence['passes'])

    def test_between_rejects_lateral_and_endpoint_candidates(self):
        anchors=[obj(2,'stove',-2,0), obj(3,'fridge',2,0)]
        for target in (obj(1,'cabinet',0,1), obj(1,'cabinet',3,0)):
            objects=[target, *anchors]
            raw={'Target':['cabinet'],'Anchor':['stove','fridge'],'Relations':[],
                 'Observer':{},'Attributes':[],'Unhandled':[]}
            query='cabinet between the stove and the fridge'
            data=validate_structure(query, recover_between_relations(query, raw))
            result=diagnose(data, objects, objects[:1], [0,0,.5], supported(objects),
                            trusted_categories=True)
            self.assertEqual(result['candidate_ids'], [])
            self.assertFalse(result['geometry']['between_triplets'][0]['passes'])

    def test_size_and_floor_constraints_use_box_proxies(self):
        objects=[obj(1,'cabinet',0,0), obj(2,'cabinet',0,1)]
        objects[0]['bbox_3d'][3:] = [.4,.4,.5]
        objects[1]['bbox_3d'][3:] = [1.,1.,2.]
        raw={'Target':['cabinet'],'Anchor':[],'Relations':[],
             'Observer':{},'Attributes':[{'category':'cabinet','phrase':'smallest'}], 'Unhandled':[]}
        data=validate_structure('the smallest cabinet', raw)
        result=diagnose(data, objects, objects, [0,0,.5], supported(objects),
                        trusted_categories=True)
        self.assertEqual(result['candidate_ids'], [1])
        self.assertEqual(result['geometry']['size_constraints'][0]['metric'], 'volume_m3')
        self.assertEqual(result['visual_needs'],[])

    def test_candidate_size_metrics_are_soft_evidence_for_adjectives(self):
        objects=[obj(1,'cabinet',0,0), obj(2,'cabinet',1,0)]
        objects[0]['bbox_3d'][3:] = [.4,.4,.5]
        objects[1]['bbox_3d'][3:] = [1.,1.,2.]
        raw={'Target':['cabinet'],'Anchor':[],'Relations':[],
             'Observer':{},'Attributes':[{'category':'cabinet','phrase':'small'}], 'Unhandled':[]}
        data=validate_structure('the small cabinet', raw)
        result=diagnose(data, objects, objects, [0,0,.5], supported(objects),
                        trusted_categories=True)
        metrics={item['id']:item for item in result['geometry']['candidate_metrics']}
        self.assertEqual(metrics[1]['volume_rank'], 1)
        self.assertEqual(metrics[2]['volume_rank'], 2)
        self.assertEqual(result['candidate_ids'], [1,2])

    def test_skinny_exposes_shape_proxies_without_choosing_shortest_edge(self):
        objects=[obj(1,'cabinet',0,0),obj(2,'cabinet',1,0)]
        objects[0]['bbox_3d'][3:] = [.157, .310, .818]
        objects[1]['bbox_3d'][3:] = [.131, 1.025, .837]
        data=validate_structure('skinny cabinet', {
            'Target':['cabinet'],'Anchor':[],'Relations':[],'Observer':{},
            'Attributes':[{'category':'cabinet','phrase':'skinny'}],'Unhandled':[]})
        result=diagnose(data, objects, objects, [0,0,.5], trusted_categories=True)
        metrics={item['id']:item for item in result['geometry']['candidate_metrics']}
        self.assertLess(metrics[2]['horizontal_short_extent_m'],metrics[1]['horizontal_short_extent_m'])
        self.assertLess(metrics[1]['horizontal_long_extent_m'],metrics[2]['horizontal_long_extent_m'])
        self.assertAlmostEqual(metrics[1]['horizontal_aspect_ratio'], .310/.157)
        self.assertAlmostEqual(metrics[1]['height_to_horizontal_long_ratio'], .818/.310)
        self.assertEqual(result['candidate_ids'], [1,2])
        self.assertEqual(result['geometry']['size_constraints'], [])
        self.assertEqual(result['status'],'missing')


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
