import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from diagnostic import (recover_between_relations, recover_distance_relations,
                        refresh_condition_records, validate_structure, diagnose,
                        viewpoint_evidence)
from evidence import validate_distance_relations
from qwen_api import QwenAPIError
from rgb_review import load_crops, input_hashes, selective_review


def obj(oid, category, x, y):
    return {'bbox_id': oid, 'target': category, 'bbox_3d': [x, y, .5, .4, .4, 1.]}


def parsed(query, relations, origin=None, facing=None, attrs=None):
    data={'Target':['window'], 'Anchor':[], 'Relations':relations,
          'Observer':{'origin':origin or {}, 'facing':facing or {}}, 'Attributes':attrs or [], 'Unhandled':[]}
    return validate_structure(query, validate_distance_relations(query, data))


def supported(objects):
    return [{'id':o['bbox_id'], 'category':o['target'], 'verdict':'yes'} for o in objects]


def synthetic_oriented_sizes(objects):
    return [{'id': o['bbox_id'], 'status': 'known', 'source': 'complete synthetic cuboid',
             'oriented_volume_m3': float(np.prod(o['bbox_3d'][3:])),
             'oriented_footprint_m2': float(np.prod(o['bbox_3d'][3:5])),
             'metric_reliability': {'oriented_volume_m3': {'status': 'usable'},
                                    'oriented_footprint_m2': {'status': 'usable'}}}
            for o in objects]


def response(oid, verdict='yes', finish='stop'):
    return {'_content':json.dumps({'id':oid,'verdict':verdict,'reason':'visible evidence'}),
            '_baseline':{'model':'mock','latency_s':.01}, '_finish_reason':finish, 'usage':{}}


class DiagnosticTests(unittest.TestCase):
    def test_condition_record_preserves_exact_quote_and_target_role(self):
        checked = validate_structure('Choose the white cabinet.', {
            'Target': ['cabinet'], 'Anchor': [], 'Relations': [], 'Observer': {},
            'Attributes': [{'category': 'cabinet', 'phrase': 'white'}], 'Unhandled': [],
        })
        record = next(item for item in checked['condition_records']
                      if item['condition_type'] == 'appearance')
        self.assertEqual(record['source_quote'], 'white')
        self.assertEqual(record['target_role'], {'category': 'cabinet'})
        self.assertIsNone(record['reference_role'])
        self.assertEqual(record['binding_status'], 'bound')
        self.assertFalse(checked['diagnostic_missing'])

    def test_direct_same_category_modifier_binds_target(self):
        query = 'Choose the red shelf with a shelf above it.'
        checked = validate_structure(query, {
            'Target': ['shelf'], 'Anchor': ['shelf'],
            'Relations': [{'target_category': 'shelf', 'anchor_category': 'shelf',
                           'relation': 'above', 'scope': 'relative_anchor',
                           'phrase': 'a shelf above it'}],
            'Observer': {}, 'Attributes': [{'category': 'shelf', 'phrase': 'red'}],
            'Unhandled': [],
        })
        record = next(item for item in checked['condition_records']
                      if item['condition_type'] == 'appearance')
        self.assertEqual(record['source_quote'], 'red')
        self.assertEqual(record['binding_status'], 'bound')
        self.assertEqual(record['target_role'], {'category': 'shelf'})
        self.assertEqual(checked['Attributes'][0]['role'], 'target')
        self.assertNotIn('attribute target/reference role is ambiguous: red',
                         checked['diagnostic_missing'])

    def test_refresh_keeps_ambiguous_attribute_binding_reason(self):
        query = 'Choose the shelf with a shelf above it. One is red.'
        checked = validate_structure(query, {
            'Target': ['shelf'], 'Anchor': ['shelf'],
            'Relations': [{'target_category': 'shelf', 'anchor_category': 'shelf',
                           'relation': 'above', 'scope': 'relative_anchor',
                           'phrase': 'a shelf above it'}],
            'Observer': {}, 'Attributes': [{'category': 'shelf', 'phrase': 'red'}],
            'Unhandled': [],
        })
        refresh_condition_records(query, checked)
        record = next(item for item in checked['condition_records']
                      if item['condition_type'] == 'appearance')
        self.assertEqual(record['source_quote'], 'red')
        self.assertEqual(record['binding_status'], 'ambiguous')
        self.assertEqual(record['evidence_status'], 'missing')
        self.assertEqual(record['reason'],
                         'attribute target/reference role is ambiguous: red')

    def test_null_anchor_categories_does_not_break_condition_records(self):
        checked = validate_structure('tall cabinet nearest the door', {
            'Target': ['cabinet'], 'Anchor': ['door'],
            'Relations': [{'target_category': 'cabinet', 'anchor_category': 'door',
                           'anchor_categories': None, 'relation': 'nearest',
                           'phrase': 'nearest the door'}],
            'Observer': {}, 'Attributes': [{'category': 'cabinet', 'phrase': 'tall'}],
            'Unhandled': [],
        })
        record = next(item for item in checked['condition_records']
                      if item['condition_type'] == 'relation:nearest')
        self.assertEqual(record['reference_role'],
                         {'categories': ['door']})

    def test_scalar_anchor_categories_stays_missing(self):
        checked = validate_structure('tall cabinet nearest the door', {
            'Target': ['cabinet'], 'Anchor': ['door'],
            'Relations': [{'target_category': 'cabinet', 'anchor_category': 'door',
                           'anchor_categories': 17, 'relation': 'nearest',
                           'phrase': 'nearest the door'}],
            'Observer': {}, 'Attributes': [{'category': 'cabinet', 'phrase': 'tall'}],
            'Unhandled': [],
        })
        self.assertTrue(any(item['evidence_status'] == 'missing'
                            for item in checked['condition_records']))
        self.assertTrue(any(item.startswith('unsupported relation:')
                            for item in checked['diagnostic_missing']))

    def test_scalar_anchor_category_stays_missing_without_crashing(self):
        checked = validate_structure('cabinet above the refrigerator', {
            'Target': ['cabinet'], 'Anchor': ['refrigerator'],
            'Relations': [{'target_category': 'cabinet', 'anchor_category': 17,
                           'relation': 'above', 'scope': 'relative_anchor',
                           'phrase': 'above the refrigerator'}],
            'Observer': {}, 'Attributes': [], 'Unhandled': [],
        })
        self.assertIn('unsupported relation: above the refrigerator',
                      checked['diagnostic_missing'])
        self.assertTrue(any(item['binding_status'] == 'unbound'
                            for item in checked['condition_records']))

    def test_scalar_relations_root_is_recorded_without_crashing(self):
        checked = validate_structure('Choose the cabinet attached to a wall.', {
            'Target': ['cabinet'], 'Anchor': [], 'Relations': 17,
            'Observer': {}, 'Attributes': [], 'Unhandled': [],
        })
        self.assertIn('relation coverage not supplied by parser',
                      checked['diagnostic_missing'])
        self.assertTrue(any(item['condition_type'] == 'relation' and
                            item['binding_status'] == 'unbound'
                            for item in checked['condition_records']))

    def test_rejected_relation_does_not_bind_its_reference_attribute(self):
        checked = validate_structure('Choose the cabinet with red chair attached to a wall.', {
            'Target': ['cabinet'], 'Anchor': [],
            'Relations': [{'target_category': 'cabinet', 'anchor_category': 'chair',
                           'relation': 'other',
                           'phrase': 'chair attached to a wall'}],
            'Observer': {}, 'Attributes': [{'category': 'chair', 'phrase': 'red'}],
            'Unhandled': [],
        })
        record = next(item for item in checked['condition_records']
                      if item['condition_type'] == 'appearance')
        self.assertEqual(record['binding_status'], 'unbound')
        self.assertIsNone(record['reference_role'])
        self.assertIn('attribute category is not a supplied target/reference: chair',
                      checked['diagnostic_missing'])
        relation_record = next(item for item in checked['condition_records']
                               if item['condition_type'] == 'relation:other')
        self.assertEqual(relation_record['binding_status'], 'unbound')
        self.assertIsNone(relation_record['reference_role'])

    def test_malformed_relation_is_not_dropped_before_diagnosis(self):
        checked = validate_structure('Choose the cabinet attached to a wall.', {
            'Target': ['cabinet'], 'Anchor': [], 'Relations': [None],
            'Observer': {}, 'Attributes': [], 'Unhandled': [],
        })
        self.assertIn('parser produced a non-object relation entry',
                      checked['diagnostic_missing'])
        self.assertTrue(any(item['condition_type'] == 'coverage_gap' and
                            item['evidence_status'] == 'missing'
                            for item in checked['condition_records']))

    def test_missing_attributes_stay_missing(self):
        checked = validate_structure('The folded vertical towel.', {
            'Target': ['towel'], 'Anchor': [], 'Relations': [], 'Observer': {},
            'Attributes': [], 'Unhandled': [],
        })
        for phrase in ('folded', 'vertical'):
            self.assertTrue(any(item['source_quote'] == phrase and
                                item['evidence_status'] == 'missing'
                                for item in checked['condition_records']))
            self.assertIn('appearance condition not represented: ' + phrase,
                          checked['diagnostic_missing'])

    def test_direction_without_observer_frame_stays_missing(self):
        checked = validate_structure('the chair on the left', {
            'Target': ['chair'], 'Anchor': [],
            'Relations': [{'target_category': 'chair', 'anchor_category': '',
                           'relation': 'left', 'scope': 'side', 'phrase': 'on the left'}],
            'Observer': {}, 'Attributes': [], 'Unhandled': [],
        })
        self.assertIn('observer origin or facing direction is unspecified',
                      checked['diagnostic_missing'])
        self.assertTrue(any(item['condition_type'] == 'viewpoint' and
                            item['evidence_status'] == 'missing'
                            for item in checked['condition_records']))

    def test_simple_comparative_size_recovery(self):
        for word, metric in (('biggest', 'volume_m3'), ('larger', 'volume_m3'),
                             ('smaller', 'volume_m3'),
                             ('bigger', 'volume_m3'), ('taller', 'height_m'),
                             ('shorter', 'height_m')):
            with self.subTest(word=word):
                query = f'The {word} cabinet.'
                checked = validate_structure(query, {
                    'Target': ['cabinet'], 'Anchor': [], 'Relations': [],
                    'Observer': {}, 'Attributes': [], 'Unhandled': [query],
                })
                self.assertEqual(checked['Attributes'],
                                 [{'category': 'cabinet', 'phrase': word}])
                self.assertEqual(checked['Unhandled'], [])
                self.assertNotIn('appearance condition not represented: ' + word,
                                 checked['diagnostic_missing'])

    def test_compound_comparative_size_stays_unhandled(self):
        for query in ('Choose the larger red cabinet.',
                      'The taller cabinet near the door.'):
            with self.subTest(query=query):
                checked = validate_structure(query, {
                    'Target': ['cabinet'], 'Anchor': ['door'] if 'door' in query else [],
                    'Relations': [], 'Observer': {}, 'Attributes': [],
                    'Unhandled': [query],
                })
                self.assertEqual(checked['Attributes'], [])
                self.assertEqual(checked['Unhandled'], [query])
                self.assertIn('appearance condition not represented: ' +
                              next(word for word in ('larger', 'smaller', 'taller')
                                   if word in query.lower()), checked['diagnostic_missing'])

    def test_explicit_pair_size_recovers_but_requires_two_complete_members(self):
        objects = [obj(1, 'cabinet', 0, 0), obj(2, 'cabinet', 1, 0),
                   obj(3, 'cabinet', 2, 0)]
        objects[1]['bbox_3d'][3:] = [1., 1., 1.]
        objects[2]['bbox_3d'][3:] = [2., 2., 2.]
        for query in ('The smaller cabinet of the two.', 'The smaller of the two cabinets.'):
            with self.subTest(query=query):
                checked = validate_structure(query, {
                    'Target': ['cabinet'], 'Anchor': [], 'Relations': [], 'Observer': {},
                    'Attributes': [], 'Unhandled': [query],
                })
                self.assertEqual(checked['Unhandled'], [])
                attribute, = checked['Attributes']
                self.assertEqual(attribute['phrase'], 'smaller')
                self.assertEqual(attribute['comparison_scope'], 'explicit_pair')
                self.assertEqual(attribute['required_member_count'], 2)
                for pool in (objects[:1], objects):
                    result = diagnose(checked, pool, pool, [0, 0, .5], trusted_categories=True,
                                      size_measurements=synthetic_oriented_sizes(pool))
                    self.assertEqual(result['status'], 'missing')
                    self.assertEqual(result['candidate_ids'], [o['bbox_id'] for o in pool])
                    condition, = result['geometry']['size_constraints']
                    self.assertEqual(condition['scope_status'], 'unknown')
                    self.assertTrue(all(c['status'] == 'unknown' for c in condition['candidate_evidence']))
                result = diagnose(checked, objects[:2], objects[:2], [0, 0, .5],
                                  trusted_categories=True,
                                  size_measurements=synthetic_oriented_sizes(objects[:2]))
                self.assertEqual(result['candidate_ids'], [1])
                self.assertEqual(result['status'], 'resolved')
                self.assertEqual(result['geometry']['size_constraints'][0]['comparison_pool_ids'], [1, 2])
                unknown = diagnose(checked, objects[:2], objects[:2], [0, 0, .5],
                                   trusted_categories=True,
                                   size_measurements=synthetic_oriented_sizes(objects[:1]))
                self.assertEqual(unknown['candidate_ids'], [1, 2])
                self.assertEqual(unknown['status'], 'missing')
                condition, = unknown['geometry']['size_constraints']
                self.assertEqual(condition['unknown_measurement_ids'], [2])
                self.assertTrue(all(c['status'] == 'unknown' for c in condition['candidate_evidence']))

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

    def test_nothing_underneath_cannot_resolve_when_parser_omits_exclusion(self):
        query = 'the tall white table with nothing underneath'
        for relations in ([], [{'target_category': 'table', 'anchor_category': 'shelf',
                               'relation': 'below', 'scope': 'relative_anchor',
                               'phrase': 'underneath'}]):
            with self.subTest(relations=relations):
                data = validate_structure(query, {
                    'Target': ['table'], 'Anchor': [], 'Relations': relations,
                    'Observer': {}, 'Attributes': [{'category': 'table', 'phrase': 'tall'},
                                                  {'category': 'table', 'phrase': 'white'}],
                    'Unhandled': [],
                })
                self.assertIn('negation/exclusion is unsupported', data['diagnostic_missing'])
                self.assertEqual(data['Relations'], [])
                objects = [obj(1, 'table', 0, 0)]
                result = diagnose(data, objects, objects, [0, 0, .5], supported(objects),
                                  trusted_categories=True)
                self.assertEqual(result['status'], 'missing')
                self.assertEqual(result['candidate_ids'], [1])

    def test_omitted_underneath_condition_cannot_resolve_from_appearance(self):
        query = 'the white table underneath the shelf'
        data = validate_structure(query, {
            'Target': ['table'], 'Anchor': [], 'Relations': [], 'Observer': {},
            'Attributes': [{'category': 'table', 'phrase': 'white'}], 'Unhandled': [],
        })
        self.assertIn('vertical condition not represented', data['diagnostic_missing'])
        objects = [obj(1, 'table', 0, 0)]
        result = diagnose(data, objects, objects, [0, 0, .5], supported(objects),
                          trusted_categories=True)
        self.assertEqual(result['status'], 'missing')

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

    def test_viewpoint_evidence_records_language_frame_sources_and_assumptions(self):
        result = self.diagnosis()
        evidence = viewpoint_evidence(
            validate_structure(self.query, {
                'Target': ['window'], 'Anchor': ['tv'],
                'Relations': [self.direction], 'Observer': {
                    'origin': self.origin, 'facing': self.facing,
                }, 'Viewpoint': 'middle of the room facing the tv',
                'Attributes': [], 'Unhandled': [],
            }), result)
        self.assertEqual(evidence['status'], 'known')
        self.assertTrue(evidence['conditions_share_hypothesis'])
        self.assertTrue(evidence['language_reference_frames'])
        frame = evidence['language_reference_frames'][0]['frame']
        self.assertEqual(frame['frame_type'], 'language_reference')
        self.assertEqual(frame['source_quotes']['origin'], 'middle of the room')
        self.assertEqual(frame['source_quotes']['facing'], 'facing the tv')
        self.assertTrue(frame['assumptions'])
        self.assertFalse(frame['capture_camera']['affects_language_frame'])

    def test_viewpoint_orientation_only_stays_unknown(self):
        data = validate_structure('facing the tv, the window on the right', {
            'Target': ['window'], 'Anchor': ['tv'],
            'Relations': [{**self.direction, 'phrase': 'on the right'}],
            'Observer': {'origin': {}, 'facing': self.facing},
            'Viewpoint': 'facing the tv', 'Attributes': [], 'Unhandled': [],
        })
        result = diagnose(data, self.objects, self.objects[:2], [0, 0, .5],
                          supported(self.objects))
        evidence = viewpoint_evidence(data, result)
        self.assertEqual(evidence['status'], 'unknown')
        self.assertEqual(evidence['language_reference_frames'], [])
        self.assertIn('description constrains orientation only; observer origin is unspecified',
                      evidence['missing'])
        self.assertIn('observer origin or facing direction is unspecified', result['missing'])

    def test_intrinsic_front_back_without_frame_stays_unknown(self):
        data = validate_structure('the window in front of the chair', {
            'Target': ['window'], 'Anchor': ['chair'],
            'Relations': [{'target_category': 'window', 'anchor_category': 'chair',
                           'relation': 'front', 'scope': 'relative_anchor',
                           'reference_frame': 'intrinsic',
                           'phrase': 'in front of the chair'}],
            'Observer': {}, 'Viewpoint': '', 'Attributes': [], 'Unhandled': [],
        })
        result = diagnose(data, self.objects + [obj(4, 'chair', 1, 1)], self.objects[:2],
                          [0, 0, .5], supported(self.objects + [obj(4, 'chair', 1, 1)]))
        evidence = viewpoint_evidence(data, result)
        self.assertEqual(evidence['status'], 'unknown')
        self.assertIn('object-intrinsic directional frame is unavailable: in front of the chair',
                      evidence['missing'])
        self.assertEqual(data['Relations'], [])
        self.assertEqual(result['status'], 'missing')

    def test_facing_chair_backs_records_independent_orientation_gap(self):
        query = 'Facing the back of the chairs, the chair on the left.'
        data = validate_structure(query, {
            'Target': ['chair'], 'Anchor': ['chair'],
            'Relations': [{'target_category': 'chair', 'anchor_category': 'chair',
                           'anchor_categories': ['chair'], 'relation': 'left',
                           'scope': 'relative_anchor', 'phrase': 'the chair on the left'}],
            'Observer': {'origin': {}, 'facing': {
                'category': 'chair', 'group': False,
                'phrase': 'Facing the back of the chairs'}},
            'Attributes': [], 'Unhandled': [],
        })
        reason = 'object-intrinsic facing orientation is unavailable: Facing the back of the chairs'
        self.assertEqual(data['Observer']['facing']['reference_frame'], 'intrinsic')
        self.assertEqual(data['Observer']['facing']['orientation_part'], 'back')
        self.assertTrue(data['Observer']['facing']['group'])
        self.assertIn(reason, data['diagnostic_missing'])
        records = [item for item in data['condition_records']
                   if item['condition_type'] == 'viewpoint:intrinsic_orientation']
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['source_quote'], 'Facing the back of the chairs')
        self.assertEqual(records[0]['other_role'], {
            'role': 'facing_orientation', 'category': 'chair', 'part': 'back'})
        self.assertEqual(records[0]['evidence_status'], 'missing')
        objects = [obj(17, 'chair', -1, 1), obj(18, 'chair', 1, 1)]
        result = diagnose(data, objects, objects, [0, 0, .5], trusted_categories=True)
        self.assertEqual(result['status'], 'missing')
        self.assertTrue(all(item['passes'] is None for item in result['geometry']['direction_evidence']))
        self.assertIn(reason, viewpoint_evidence(data, result)['missing'])

    def test_facing_back_quote_is_recovered_when_parser_shortens_it(self):
        query = 'Facing the back of the chairs, the chair on the left.'
        data = validate_structure(query, {
            'Target': ['chair'], 'Anchor': [], 'Relations': [],
            'Observer': {'origin': {}, 'facing': {
                'category': 'chair', 'group': False, 'phrase': 'facing the chairs'}},
            'Attributes': [], 'Unhandled': [],
        })
        self.assertEqual(data['Observer']['facing']['phrase'], 'Facing the back of the chairs')
        self.assertTrue(data['Observer']['facing']['group'])
        self.assertIn('object-intrinsic facing orientation is unavailable: Facing the back of the chairs',
                      data['diagnostic_missing'])

    def test_facing_bed_head_and_foot_cannot_substitute_center_direction(self):
        for part in ('head', 'foot'):
            with self.subTest(part=part):
                quote = f'facing the {part} of the bed'
                query = f'Standing in the middle of the room {quote}, the window on the right.'
                data = validate_structure(query, {
                    'Target': ['window'], 'Anchor': ['bed'], 'Relations': [self.direction],
                    'Observer': {'origin': self.origin, 'facing': {
                        'category': 'bed', 'group': False, 'phrase': 'facing the bed'}},
                    'Attributes': [], 'Unhandled': [],
                })
                record, = [item for item in data['condition_records']
                           if item['condition_type'] == 'viewpoint:intrinsic_orientation']
                self.assertEqual(record['source_quote'], quote)
                self.assertEqual(record['other_role']['part'], part)
                self.assertEqual(record['evidence_status'], 'missing')
                objects = self.objects + [obj(4, 'bed', 0, 3)]
                result = diagnose(data, objects, self.objects[:2], [0, 0, .5],
                                  trusted_categories=True)
                self.assertEqual(result['status'], 'missing')
                self.assertEqual(result['candidate_ids'], [1, 2])
                evidence = viewpoint_evidence(data, result)
                self.assertEqual(evidence['status'], 'unknown')
                self.assertEqual(evidence['language_reference_frames'], [])
                self.assertIn('object-intrinsic facing orientation is unavailable: ' + quote,
                              evidence['missing'])

    def test_empty_observer_schema_does_not_claim_viewpoint_evidence(self):
        data = {'Observer': {'origin': {'kind': 'unspecified', 'category': '', 'phrase': ''},
                             'facing': {'category': '', 'phrase': '', 'group': False}},
                'Viewpoint': '', 'Relations': []}
        evidence = viewpoint_evidence(data)
        self.assertEqual(evidence['status'], 'not_applicable')
        self.assertEqual(evidence['observer'], {'origin': {}, 'facing': {}})
        self.assertEqual(evidence['missing'], [])

    def test_facing_object_back_cannot_use_center_direction_with_explicit_origin(self):
        query = 'Standing in the middle of the room facing the back of the chair, the window on the right.'
        data = validate_structure(query, {
            'Target': ['window'], 'Anchor': ['chair'], 'Relations': [self.direction],
            'Observer': {'origin': self.origin, 'facing': {
                'category': 'chair', 'group': False, 'phrase': 'facing the back of the chair'}},
            'Attributes': [], 'Unhandled': [],
        })
        objects = self.objects + [obj(4, 'chair', 0, 3)]
        result = diagnose(data, objects, self.objects[:2], [0, 0, .5], trusted_categories=True)
        self.assertFalse(data['Observer']['facing']['group'])
        self.assertEqual(result['status'], 'missing')
        self.assertEqual(result['candidate_ids'], [1, 2])
        self.assertTrue(all(item['observer'] is None for item in result['hypotheses']))
        self.assertEqual(viewpoint_evidence(data, result)['language_reference_frames'], [])

    def test_side_direction_survives_rewrite_with_other_relation(self):
        from query_rewrite import rewrite_query
        query = ('Standing in the middle of the room facing the tv, '
                 'the window on the right below the table.')
        objects = self.objects + [{'bbox_id': 4, 'target': 'table',
                                   'bbox_3d': [0, 1, 2, 4, 1, 1]}]
        data = validate_structure(query, {
            'Target': ['window'], 'Anchor': ['table'],
            'Relations': [self.direction,
                          {'target_category': 'window', 'anchor_category': 'table',
                           'relation': 'below', 'scope': 'relative_anchor',
                           'phrase': 'below the table'}],
            'Observer': {'origin': self.origin, 'facing': self.facing},
            'Attributes': [], 'Unhandled': [],
        })
        data['query_rewrite'] = rewrite_query(query, data)
        result = diagnose(data, objects, objects[:2], [0, 0, .5],
                          trusted_categories=True)
        self.assertEqual(result['status'], 'resolved')
        self.assertEqual(result['candidate_ids'], [2])
        self.assertTrue(result['geometry']['direction_evidence'])

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
        self.assertEqual(data['Attributes'][0]['phrase'], 'large')
        result = diagnose(data, self.objects, self.objects[:2], [0, 0, .5],
                          trusted_categories=True)
        self.assertEqual(result['status'], 'missing')
        condition, = result['geometry']['size_constraints']
        self.assertEqual(condition['attribute'], 'large')
        self.assertEqual(condition['metric'], 'vertical_face_area_m2')
        self.assertEqual(condition['status'], 'unknown')

    def test_unbound_target_with_direction_evidence_stays_missing(self):
        query = 'When standing in the middle of the room facing the windows the correct one is on the right.'
        data = validate_structure(query, {
            'Target': [], 'Anchor': ['windows'],
            'Relations': [{'target_category': 'windows', 'anchor_category': '',
                           'relation': 'right', 'scope': 'side', 'phrase': 'on the right'}],
            'Observer': {
                'origin': {'kind': 'room_center', 'category': '',
                           'phrase': 'standing in the middle of the room'},
                'facing': {'category': 'window', 'group': True,
                           'phrase': 'facing the windows'},
            },
            'Attributes': [], 'Unhandled': ['the correct one'],
        })
        windows = [obj(1, 'window', -1., 0.), obj(2, 'window', 1., 0.)]
        result = diagnose(data, windows, windows, [0., 0., .5], supported(windows),
                          trusted_categories=True)
        self.assertEqual(result['candidate_ids'], [1, 2])
        self.assertTrue(all(not row['removal_allowed'] for row in result['candidate_removal_audit']))
        self.assertEqual(result['status'], 'missing')
        self.assertIn('requires one bound target category', result['missing'])

    def test_near_records_aabb_gap_without_filtering_or_center_ranking(self):
        objects = [obj(1, 'window', 1, 0), obj(2, 'window', 3, 0), obj(3, 'tv', 0, 0)]
        # The larger target reaches the anchor despite its more distant center.
        objects[1]['bbox_3d'][3] = 5.6
        relation = {'target_category': 'window', 'anchor_category': 'tv',
                    'relation': 'near', 'phrase': 'near the tv'}
        data = parsed('window near the tv', [relation])
        result = diagnose(data, objects, objects[:2], [0, 0, .5], supported(objects))
        pairs = {pair['target_id']: pair for pair in result['geometry']['proximity_pairs']}
        self.assertAlmostEqual(pairs[1]['aabb_gap_m'], .6)
        self.assertAlmostEqual(pairs[2]['aabb_gap_m'], 0.)
        self.assertLess(pairs[1]['center_distance_m'], pairs[2]['center_distance_m'])
        self.assertEqual(pairs[2]['relative_rank'], 1)
        self.assertEqual(pairs[1]['relative_rank'], 2)
        self.assertTrue(all(pair['passes'] is None for pair in pairs.values()))
        self.assertEqual(result['candidate_ids'], [1, 2])
        self.assertEqual(result['status'], 'missing')
        self.assertIn('near has no calibrated predicate; distances are evidence only', result['missing'])

    def test_near_gap_ties_do_not_gain_an_id_based_preference(self):
        objects = [obj(8, 'window', -.1, 0), obj(2, 'window', .2, 0), obj(3, 'tv', 0, 0)]
        relation = {'target_category': 'window', 'anchor_category': 'tv',
                    'relation': 'near', 'phrase': 'near the tv'}
        result = diagnose(parsed('window near the tv', [relation]), objects, objects[:2],
                          [0, 0, .5], supported(objects))
        pairs = result['geometry']['proximity_pairs']
        self.assertEqual([pair['relative_rank'] for pair in pairs], [1, 1])
        self.assertEqual([pair['best_ids'] for pair in pairs], [[2, 8], [2, 8]])
        self.assertEqual([pair['relative_percentile'] for pair in pairs], [0., 0.])
        self.assertEqual(result['candidate_ids'], [2, 8])
        self.assertTrue(all('zero gap does not prove contact' in pair['policy'] for pair in pairs))

    def test_nearest_diagnosis_preserves_distance_ties(self):
        targets = [obj(8, 'window', 1.0, 0), obj(2, 'window', 1.006, 0)]
        anchor = obj(3, 'tv', 0, 0)
        relation = {'target_category': 'window', 'anchor_category': 'tv',
                    'relation': 'nearest', 'phrase': 'nearest the tv'}
        data = parsed('window nearest the tv', [relation])
        result = diagnose(data, targets + [anchor], targets, [0, 0, .5],
                          supported(targets + [anchor]))
        rankings = result['geometry']['distance_rankings']
        self.assertEqual(len(rankings), 1)
        self.assertEqual(rankings[0]['best_ids'], [2, 8])
        self.assertEqual(rankings[0]['tie_tolerance_m'], .01)
        self.assertEqual(result['candidate_ids'], [2, 8])
        self.assertEqual(result['status'], 'ambiguous')

    def test_distance_extreme_waits_for_complete_color_subset(self):
        targets = [obj(1, 'window', 1, 0), obj(2, 'window', 3, 0)]
        anchor = obj(3, 'tv', 0, 0)
        query = 'Choose the red window nearest the tv.'
        data = parsed(query, [{'target_category': 'window', 'anchor_category': 'tv',
                               'relation': 'nearest', 'phrase': 'nearest the tv'}],
                      attrs=[{'category': 'window', 'phrase': 'red'}])
        unknown = diagnose(data, targets + [anchor], targets, [0, 0, .5],
                           trusted_categories=True)
        self.assertEqual(unknown['candidate_ids'], [1, 2])
        self.assertEqual(unknown['geometry']['distance_rankings'][0]['best_ids'], [])
        self.assertEqual(unknown['status'], 'missing')
        checks = [
            {'id': target['bbox_id'], 'category': 'window', 'verdict': 'yes',
             'visibility': 'visible', 'reason': 'window visible',
             'attribute_checks': [{'attribute': 'red', 'verdict': verdict,
                                   'visibility': 'visible', 'reason': 'surface color visible'}]}
            for target, verdict in zip(targets, ['no', 'yes'])]
        bound = diagnose(data, targets + [anchor], targets, [0, 0, .5], checks,
                         trusted_categories=True)
        self.assertEqual(bound['candidate_ids'], [2])
        self.assertEqual(bound['geometry']['distance_rankings'][0]['comparison_pool_ids'], [2])
        self.assertEqual(bound['status'], 'resolved')

    def test_size_filter_does_not_redefine_nearest_comparison_pool(self):
        targets = [obj(1, 'cabinet', 1, 0), obj(2, 'cabinet', 3, 0)]
        targets[1]['bbox_3d'][3:] = [2., 2., 2.]
        anchor = obj(3, 'tv', 0, 0)
        query = 'Choose the largest cabinet nearest the tv.'
        data = validate_structure(query, {
            'Target': ['cabinet'], 'Anchor': ['tv'], 'Observer': {}, 'Unhandled': [],
            'Relations': [{'target_category': 'cabinet', 'anchor_category': 'tv',
                           'relation': 'nearest', 'phrase': 'nearest the tv'}],
            'Attributes': [{'category': 'cabinet', 'phrase': 'largest'}],
        })
        result = diagnose(data, targets + [anchor], targets, [0, 0, .5],
                          trusted_categories=True)
        self.assertEqual(result['candidate_ids'], [])
        self.assertEqual(result['status'], 'conflict')
        self.assertEqual(result['geometry']['distance_rankings'][0]['best_ids'], [1])
        self.assertEqual(result['geometry']['distance_rankings'][0]['comparison_pool_ids'], [1, 2])

    def test_missing_window_face_size_does_not_resolve_from_nearest_alone(self):
        targets = [obj(1, 'window', 1, 0), obj(2, 'window', 3, 0)]
        targets[1]['bbox_3d'][3:] = [2., 2., 2.]
        anchor = obj(3, 'tv', 0, 0)
        data = parsed('Choose the largest window nearest the tv.', [
            {'target_category': 'window', 'anchor_category': 'tv',
             'relation': 'nearest', 'phrase': 'nearest the tv'}],
            attrs=[{'category': 'window', 'phrase': 'largest'}])
        result = diagnose(data, targets + [anchor], targets, [0, 0, .5],
                          trusted_categories=True)
        self.assertEqual(result['candidate_ids'], [1])
        self.assertEqual(result['status'], 'missing')
        self.assertEqual(result['geometry']['distance_rankings'][0]['comparison_pool_ids'], [1, 2])
        condition, = result['geometry']['size_constraints']
        self.assertEqual(condition['metric'], 'vertical_face_area_m2')
        self.assertEqual(condition['unknown_measurement_ids'], [1, 2])

    def test_diagnosis_uses_horizontal_metric_when_center_metric_disagrees(self):
        near_horiz = obj(1, 'window', 0, 0)
        near_horiz['bbox_3d'][2] = 10.
        near_center = obj(2, 'window', 3, 0)
        anchor = obj(3, 'table', 0, 0)
        relation = {'target_category': 'window', 'anchor_category': 'table',
                    'relation': 'nearest', 'metric': 'horizontal_center',
                    'phrase': 'nearest the table'}
        data = parsed('window nearest the table', [relation])
        result = diagnose(data, [near_horiz, near_center, anchor],
                          [near_horiz, near_center], [0, 0, .5],
                          supported([near_horiz, near_center, anchor]))
        ranking = result['geometry']['distance_rankings'][0]
        self.assertEqual(ranking['metric'], 'horizontal_center')
        self.assertEqual(ranking['best_ids'], [1])
        self.assertEqual(result['candidate_ids'], [1])

    def test_plane_distance_mixed_measurements_remains_unknown(self):
        known = obj(1, 'window', 1, 2)
        unknown = obj(2, 'window', 8, 2)
        wall = obj(3, 'wall', 0, 0)
        wall['wall_geometry'] = {
            'valid': True, 'origin_xy_m': [0, 0], 'normal_xy': [0, 1],
            'tangent_xy': [1, 0], 'tangent_extent_m': [0, 4],
        }
        relation = {'target_category': 'window', 'anchor_category': 'wall',
                    'relation': 'nearest', 'metric': 'plane',
                    'phrase': 'nearest the wall'}
        data = parsed('window nearest the wall', [relation])
        result = diagnose(data, [known, unknown, wall], [known, unknown],
                          [0, 0, .5], supported([known, unknown, wall]))
        ranking = result['geometry']['distance_rankings'][0]
        self.assertEqual(ranking['comparison_status'], 'unknown')
        self.assertEqual(ranking['unknown_target_ids'], [2])
        self.assertEqual(ranking['best_ids'], [])
        self.assertEqual(result['candidate_ids'], [1, 2])
        self.assertEqual(result['status'], 'missing')

    def test_rewritten_distance_uses_binding_metric_alias_and_tolerance(self):
        from query_rewrite import rewrite_query
        targets = [obj(1, 'window', 1.0, 0), obj(2, 'window', 1.015, 0)]
        targets[0]['bbox_3d'][2] = 10.
        anchors = [obj(3, 'tv', 10, 0), obj(4, 'tv', 0, 0)]
        query = 'the window nearest the tv'
        relation = {'target_category': 'window', 'anchor_category': 'tv',
                    'relation': 'nearest', 'phrase': 'nearest the tv',
                    'metric': 'horizontal', 'anchor_id': 4, 'tie_tolerance_m': .02}
        data = parsed(query, [relation])
        data['_verified_reference_bindings'] = {
            'tv': {'id': 4, 'status': 'bound', 'source': 'synthetic verified instance identity'}}
        data['query_rewrite'] = rewrite_query(query, data)
        result = diagnose(data, targets + anchors, targets, [0, 0, .5],
                          supported(targets + anchors))
        ranking = result['geometry']['distance_rankings'][0]
        self.assertEqual(ranking['anchor_id'], 4)
        self.assertEqual(ranking['metric'], 'horizontal_center')
        self.assertEqual(ranking['tie_tolerance_m'], .02)
        self.assertEqual(ranking['best_ids'], [1, 2])
        self.assertEqual(result['candidate_ids'], [1, 2])
        self.assertEqual(result['status'], 'ambiguous')

    def test_distance_reference_role_cannot_override_conflicting_instance_ids(self):
        targets = [obj(1, 'window', 1, 0), obj(2, 'window', 9, 0)]
        anchors = [obj(3, 'tv', 0, 0), obj(4, 'tv', 10, 0)]
        query = 'the window nearest the tv and farthest from the tv'
        relations = [
            {'target_category': 'window', 'anchor_category': 'tv',
             'relation': 'nearest', 'phrase': 'nearest the tv', 'anchor_id': 3},
            {'target_category': 'window', 'anchor_category': 'tv',
             'relation': 'farthest', 'phrase': 'farthest from the tv', 'anchor_id': 4},
        ]
        result = diagnose(parsed(query, relations), targets + anchors, targets,
                          [0, 0, .5], supported(targets + anchors))
        self.assertEqual(result['candidate_ids'], [1, 2])
        self.assertEqual(result['status'], 'missing')
        self.assertIn('distance reference has incompatible explicit instance bindings: tv',
                      result['missing'])
        self.assertEqual(result['geometry']['distance_rankings'], [])

    def test_viewpoint_distance_cache_is_specific_to_each_origin_hypothesis(self):
        targets = [obj(1, 'window', 1, 2), obj(2, 'window', 9, 2)]
        objects = targets + [obj(3, 'tv', 5, 10), obj(4, 'door', 0, 0),
                             obj(5, 'door', 10, 0)]
        query = 'Standing at the door facing the tv, the window nearest the tv on the right.'
        nearest = {'target_category': 'window', 'anchor_category': 'tv',
                   'relation': 'nearest', 'metric': 'viewpoint', 'phrase': 'nearest the tv'}
        origin = {'kind': 'category', 'category': 'door', 'phrase': 'at the door'}
        data = parsed(query, [nearest, self.direction], origin, self.facing)
        result = diagnose(data, objects, targets, [0, 0, .5], supported(objects))
        rankings = result['geometry']['distance_rankings']
        self.assertEqual([ranking['best_ids'] for ranking in rankings], [[1], [2]])
        measurements = {
            hypothesis['observer']['origin_id']: [row for row in hypothesis['measurements']
                if row.get('relation') == 'nearest' and row['status'] == 'pass'][0]['target_id']
            for hypothesis in result['hypotheses']
        }
        self.assertEqual(measurements, {4: 1, 5: 2})

    def test_near_multiple_references_keep_independent_candidate_rankings(self):
        objects = [obj(1, 'window', -1, 0), obj(2, 'window', 1, 0),
                   obj(3, 'tv', -1, 0), obj(4, 'tv', 1, 0)]
        relation = {'target_category': 'window', 'anchor_category': 'tv',
                    'relation': 'near', 'phrase': 'near the tv'}
        result = diagnose(parsed('window near the tv', [relation]), objects, objects[:2],
                          [0, 0, .5], supported(objects))
        pairs = result['geometry']['proximity_pairs']
        self.assertEqual(len(pairs), 4)
        self.assertEqual({pair['anchor_id']: pair['best_ids'] for pair in pairs},
                         {3: [1], 4: [2]})
        self.assertTrue(all(pair['candidate_count'] == 2 for pair in pairs))
        self.assertEqual(result['candidate_ids'], [1, 2])
        for hypothesis in result['hypotheses']:
            self.assertEqual({pair['anchor_id'] for pair in hypothesis['measurements']},
                             {hypothesis['references']['tv']})

    def test_near_aabb_gap_includes_vertical_separation(self):
        objects = [obj(1, 'window', 0, 0), obj(2, 'tv', 0, 0)]
        objects[1]['bbox_3d'][2] = 2.5
        relation = {'target_category': 'window', 'anchor_category': 'tv',
                    'relation': 'near', 'phrase': 'near the tv'}
        result = diagnose(parsed('window near the tv', [relation]), objects, objects[:1],
                          [0, 0, .5], supported(objects))
        pair = result['geometry']['proximity_pairs'][0]
        self.assertEqual(pair['horizontal_aabb_gap_m'], 0.)
        self.assertEqual(pair['aabb_gap_m'], 1.)
        self.assertEqual(result['candidate_ids'], [1])
        self.assertEqual(result['status'], 'missing')

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
        checks=[{'id':1,'category':'window','verdict':'yes','visibility':'visible',
                 'attribute_checks':[{'attribute':'black','verdict':'yes',
                                      'visibility':'visible','reason':'black surface visible'}]},
                {'id':2,'category':'window','verdict':'yes'}]
        result=diagnose(data,objects,objects,[0,0,.5],checks,trusted_categories=True)
        self.assertEqual(result['candidate_ids'],[1])
        self.assertEqual(result['status'],'resolved')

    def test_trusted_category_survives_unknown_visual_check_but_visible_no_excludes(self):
        cabinet=obj(1, 'cabinet', 0, 0)
        data=validate_structure('cabinet', {
            'Target':['cabinet'], 'Anchor':[], 'Relations':[], 'Observer':{},
            'Attributes':[], 'Unhandled':[],
        })
        uncertain=diagnose(data, [cabinet], [cabinet], [0,0,.5], [{
            'id':1, 'category':'cabinet', 'verdict':'uncertain',
            'visibility':'unknown', 'reason':'low clarity',
        }], trusted_categories=True)
        self.assertEqual(uncertain['candidate_ids'], [1])
        self.assertEqual(uncertain['status'], 'resolved')
        negative=diagnose(data, [cabinet], [cabinet], [0,0,.5], [{
            'id':1, 'category':'cabinet', 'verdict':'no',
            'visibility':'visible', 'inspection_status':'inspected',
            'reason':'visible object is not a cabinet',
        }], trusted_categories=True)
        self.assertEqual(negative['candidate_ids'], [])
        self.assertEqual(negative['status'], 'conflict')

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

    def test_vertical_recovery_ignores_later_embedded_clause_for_geometry(self):
        query = 'Choose the cabinet above the shelf with a painting above it.'
        objects = [obj(1, 'cabinet', 0, 0), obj(2, 'cabinet', 0, 0),
                   obj(3, 'shelf', 0, 0)]
        objects[0]['bbox_3d'][2] = 2.0
        raw = {'Target': ['cabinet'], 'Anchor': ['shelf'], 'Relations': [],
               'Observer': {}, 'Attributes': [], 'Unhandled': []}
        data = validate_structure(query, raw)
        self.assertEqual(data['Relations'][0]['relation'], 'above')
        result = diagnose(data, objects, objects[:2], [0, 0, .5],
                          supported(objects), trusted_categories=True)
        self.assertEqual(result['candidate_ids'], [1])

    def test_vertical_overlap_does_not_count_as_above(self):
        objects=[obj(1,'cabinet',0,0), obj(2,'cabinet',0,0), obj(3,'refrigerator',0,0)]
        objects[1]['bbox_3d'][2] = 0.8
        raw={'Target':['cabinet'],'Anchor':['refrigerator'],'Relations':[],
             'Observer':{},'Attributes':[],'Unhandled':[]}
        data=validate_structure('cabinet above the refrigerator', raw)
        result=diagnose(data, objects, objects[:2], [0,0,.5], supported(objects),
                        trusted_categories=True)
        self.assertEqual(result['candidate_ids'], [2])
        self.assertEqual(result['status'], 'missing')
        pair=next(p for p in result['geometry']['relation_pairs'] if p['target_id'] == 2)
        self.assertLess(pair['target_bottom_minus_anchor_top_m'], 0.)
        self.assertEqual(pair['status'], 'unknown')
        self.assertIsNone(pair['passes'])

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

    def test_between_boundary_unknown_has_reason_and_stays_missing(self):
        anchors=[obj(2,'stove',-2,0), obj(3,'fridge',2,0)]
        for target, reason in (
                (obj(1,'cabinet',0,.3), 'lateral boundary'),
                (obj(1,'cabinet',-1.98,0), 'endpoint boundary')):
            with self.subTest(reason=reason):
                objects=[target, *anchors]
                query='cabinet between the stove and the fridge'
                raw={'Target':['cabinet'],'Anchor':['stove','fridge'],'Relations':[],
                     'Observer':{},'Attributes':[],'Unhandled':[]}
                data=validate_structure(query, recover_between_relations(query, raw))
                result=diagnose(data, objects, objects[:1], [0,0,.5], supported(objects),
                                trusted_categories=True)
                evidence=result['geometry']['between_triplets'][0]
                self.assertEqual(evidence['status'], 'unknown')
                self.assertIsNone(evidence['passes'])
                self.assertIn(reason, evidence['reason'])
                self.assertEqual(result['candidate_ids'], [1])
                self.assertEqual(result['status'], 'missing')
                self.assertIn(evidence['reason'], result['missing'])

    def test_between_endpoint_boundary_covers_both_sides_of_finite_segment(self):
        query='cabinet between the stove and the fridge'
        raw={'Target':['cabinet'],'Anchor':['stove','fridge'],'Relations':[],
             'Observer':{},'Attributes':[],'Unhandled':[]}
        data=validate_structure(query, recover_between_relations(query, raw))
        for length in (.04, 4.):
            for reverse in (False, True):
                anchors=[obj(2,'stove',length if reverse else 0,0),
                         obj(3,'fridge',0 if reverse else length,0)]
                for x, expected in ((-.02, 'unknown'), (length+.02, 'unknown'),
                                    (-.1, 'fail'), (length+.1, 'fail'),
                                    (length/2, 'unknown' if length == .04 else 'pass')):
                    with self.subTest(length=length, reverse=reverse, x=x):
                        target=obj(1,'cabinet',x,0)
                        objects=[target, *anchors]
                        result=diagnose(data, objects, [target], [0,0,.5], supported(objects),
                                        trusted_categories=True)
                        evidence=result['geometry']['between_triplets'][0]
                        self.assertEqual(evidence['status'], expected)
                        self.assertEqual(evidence['within_segment'], 0 <= x <= length)
                        self.assertIs(evidence['passes'],
                                      True if expected == 'pass' else False if expected == 'fail' else None)
                        self.assertEqual(result['candidate_ids'], [] if expected == 'fail' else [1])
                        if expected == 'unknown':
                            self.assertIn('endpoint boundary', evidence['reason'])
                            self.assertEqual(result['status'], 'missing')

    def test_between_missing_references_preserve_unknown_candidates(self):
        query='cabinet between the stove and the fridge'
        raw={'Target':['cabinet'],'Anchor':['stove','fridge'],'Relations':[],
             'Observer':{},'Attributes':[],'Unhandled':[]}
        data=validate_structure(query, recover_between_relations(query, raw))
        targets=[obj(1,'cabinet',0,0), obj(4,'cabinet',3,0)]
        for anchors, expected_ids in (([], [None, None]),
                                     ([obj(2,'stove',-2,0)], [2, None]),
                                     ([obj(3,'fridge',2,0)], [None, 3])):
            for apply_size_constraints in (True, False):
                with self.subTest(anchor_ids=expected_ids,
                                  apply_size_constraints=apply_size_constraints):
                    objects=[*targets, *anchors]
                    result=diagnose(data, objects, targets, [0,0,.5],
                                    supported(objects), trusted_categories=True,
                                    apply_size_constraints=apply_size_constraints)
                    self.assertEqual(result['candidate_ids'], [1,4])
                    self.assertEqual(result['status'], 'missing')
                    self.assertEqual(result['geometry']['between_pass_ids'], [])
                    self.assertEqual(result['geometry']['between_unknown_ids'], [1,4])
                    self.assertEqual(len(result['geometry']['between_triplets']), 2)
                    for evidence in result['geometry']['between_triplets']:
                        self.assertEqual(evidence['anchor_ids'], expected_ids)
                        self.assertEqual(evidence['status'], 'unknown')
                        self.assertIsNone(evidence['passes'])
                        self.assertIn('unbound', evidence['reason'])
                        self.assertIn(evidence['reason'], result['missing'])

    def test_unknown_between_does_not_override_known_geometric_failure(self):
        query='cabinet between the stove and the fridge and between the printer and the desk'
        relations=[
            {'target_category':'cabinet','anchor_categories':['stove','fridge'],
             'relation':'between','phrase':'between the stove and the fridge'},
            {'target_category':'cabinet','anchor_categories':['printer','desk'],
             'relation':'between','phrase':'between the printer and the desk'}]
        objects=[obj(1,'cabinet',3,0), obj(2,'stove',-2,0), obj(3,'fridge',2,0)]
        for ordered_relations in (relations, list(reversed(relations))):
            with self.subTest(first=ordered_relations[0]['phrase']):
                data=validate_structure(query, {
                    'Target':['cabinet'],'Anchor':['stove','fridge','printer','desk'],
                    'Relations':ordered_relations, 'Observer':{},
                    'Attributes':[], 'Unhandled':[]})
                result=diagnose(data, objects, objects[:1], [0,0,.5],
                                supported(objects), trusted_categories=True)
                self.assertEqual(result['candidate_ids'], [])
                self.assertEqual(result['status'], 'missing')
                evidence=result['geometry']['between_triplets']
                self.assertEqual({item['status'] for item in evidence}, {'fail','unknown'})
                self.assertTrue(any(item['passes'] is False for item in evidence))
                self.assertTrue(any(item['passes'] is None for item in evidence))

    def test_between_same_category_binds_distinct_reference_instances(self):
        objects=[obj(1,'cabinet',0,0), obj(2,'table',-2,0), obj(3,'table',2,0)]
        query='cabinet between the tables'
        raw={'Target':['cabinet'],'Anchor':['table','table'],'Relations':[
            {'target_category':'cabinet','anchor_categories':['table','table'],
             'relation':'between','scope':'relative_anchors',
             'phrase':'between the tables'}],
             'Observer':{},'Attributes':[],'Unhandled':[]}
        data=validate_structure(query, raw)
        result=diagnose(data, objects, objects[:1], [0,0,.5], supported(objects),
                        trusted_categories=True)
        self.assertEqual(result['candidate_ids'], [1])
        pairs = {tuple(item['anchor_ids'])
                 for item in result['geometry']['between_triplets']}
        self.assertEqual(pairs, {(2, 3), (3, 2)})
        for hypothesis in result['hypotheses']:
            bindings = hypothesis['references']
            self.assertNotEqual(bindings['table'], bindings['__between_0_1_table'])

    def test_other_two_between_support_ignores_self_reference_hypotheses(self):
        objects=[obj(6,'box',0,0), obj(7,'box',-2,0), obj(8,'box',2,0)]
        query="It's the box that's between the other two boxes"
        raw={'Target':['box'],'Anchor':['box','box'],'Relations':[
            {'target_category':'box','anchor_categories':['box','box'],
             'relation':'between','scope':'relative_anchors',
             'phrase':'between the other two boxes'}],
             'Observer':{},'Attributes':[],'Unhandled':[]}
        data=validate_structure(query, raw)
        result=diagnose(data, objects, objects, [0,0,.5], supported(objects),
                        trusted_categories=True, query=query)
        support=result['candidate_hypothesis_support']
        self.assertTrue(support['enabled'])
        self.assertEqual(support['category_pool_ids'], [6,7,8])
        self.assertEqual(next(row for row in support['candidates']
                              if row['candidate_id'] == 6)['status'], 'pass')
        self.assertEqual(result['candidate_ids'], [6])
        self.assertEqual(result['stable_candidate_ids'], [])

    def test_other_two_policy_ignores_empty_parser_observer_fields(self):
        objects=[obj(6,'box',0,0), obj(7,'box',-2,0), obj(8,'box',2,0)]
        query="It's the box that's between the other two boxes"
        raw={'Target':['box'],'Anchor':['box','box'],'Relations':[
            {'target_category':'box','anchor_categories':['box','box'],
             'relation':'between','scope':'relative_anchors',
             'phrase':'between the other two boxes'}],
             'Observer':{'origin':{}, 'facing':{
                 'category':'', 'group':False, 'phrase':''}},
             'Attributes':[],'Unhandled':[]}
        data=validate_structure(query, raw)
        result=diagnose(data, objects, objects, [0,0,.5], supported(objects),
                        trusted_categories=True, query=query)
        self.assertTrue(result['candidate_hypothesis_support']['enabled'])
        self.assertEqual(result['candidate_ids'], [6])

    def test_other_two_policy_stays_disabled_with_an_extra_attribute(self):
        objects=[obj(6,'box',0,0), obj(7,'box',-2,0), obj(8,'box',2,0)]
        query="It's the red box that's between the other two boxes"
        raw={'Target':['box'],'Anchor':['box','box'],'Relations':[
            {'target_category':'box','anchor_categories':['box','box'],
             'relation':'between','scope':'relative_anchors',
             'phrase':'between the other two boxes'}],
             'Observer':{},'Attributes':[{'category':'box','phrase':'red'}],
             'Unhandled':[]}
        data=validate_structure(query, raw)
        result=diagnose(data, objects, objects, [0,0,.5], supported(objects),
                        trusted_categories=True, query=query)
        self.assertFalse(result['candidate_hypothesis_support']['enabled'])

    def test_between_same_category_requires_two_instances(self):
        objects=[obj(1,'cabinet',0,0), obj(2,'table',-2,0)]
        query='cabinet between the tables'
        raw={'Target':['cabinet'],'Anchor':['table','table'],'Relations':[
            {'target_category':'cabinet','anchor_categories':['table','table'],
             'relation':'between','scope':'relative_anchors',
             'phrase':'between the tables'}],
             'Observer':{},'Attributes':[],'Unhandled':[]}
        data=validate_structure(query, raw)
        result=diagnose(data, objects, objects[:1], [0,0,.5], supported(objects),
                        trusted_categories=True)
        self.assertEqual(result['candidate_ids'], [1])
        self.assertEqual(result['status'], 'missing')
        self.assertTrue(all(not row['removal_allowed'] for row in result['candidate_removal_audit']))
        self.assertIn('between reference instances are not distinctly bound: table',
                      result['missing'])

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

    def test_compound_size_extremes_require_the_same_target(self):
        objects = [obj(1, 'cabinet', 0, 0), obj(2, 'cabinet', 1, 0)]
        objects[0]['bbox_3d'][3:] = [2., 2., 1.]
        objects[1]['bbox_3d'][3:] = [.5, .5, 3.]
        data = validate_structure('the largest tallest cabinet', {
            'Target': ['cabinet'], 'Anchor': [], 'Relations': [], 'Observer': {},
            'Attributes': [{'category': 'cabinet', 'phrase': 'largest tallest'}],
            'Unhandled': [],
        })
        result = diagnose(data, objects, objects, [0, 0, .5], supported(objects),
                          trusted_categories=True)
        self.assertEqual(result['status'], 'conflict')
        self.assertEqual(result['candidate_ids'], [])
        records = result['geometry']['size_constraints']
        self.assertEqual([(row['metric'], row['selected_ids']) for row in records],
                         [('volume_m3', [1]), ('height_m', [2])])

    def test_compound_size_extremes_accept_a_shared_winner(self):
        objects = [obj(1, 'cabinet', 0, 0), obj(2, 'cabinet', 1, 0)]
        objects[0]['bbox_3d'][3:] = [2., 2., 3.]
        objects[1]['bbox_3d'][3:] = [.5, .5, 1.]
        data = validate_structure('the largest tallest cabinet', {
            'Target': ['cabinet'], 'Anchor': [], 'Relations': [], 'Observer': {},
            'Attributes': [{'category': 'cabinet', 'phrase': 'largest tallest'}],
            'Unhandled': [],
        })
        result = diagnose(data, objects, objects, [0, 0, .5], supported(objects),
                          trusted_categories=True)
        self.assertEqual(result['status'], 'resolved')
        self.assertEqual(result['candidate_ids'], [1])
        self.assertEqual(len(result['geometry']['size_constraints']), 2)

    def test_compound_size_extremes_require_the_same_reference(self):
        objects = [obj(1, 'cabinet', 0, 0), obj(2, 'cabinet', 1, 0),
                   obj(3, 'window', 0, 0)]
        objects[0]['bbox_3d'][3:] = [2., 2., 1.]
        objects[1]['bbox_3d'][3:] = [.5, .5, 3.]
        objects[2]['bbox_3d'][2] = 3.
        query = 'window above the largest tallest cabinet'
        data = validate_structure(query, {
            'Target': ['window'], 'Anchor': ['cabinet'],
            'Relations': [{'target_category': 'window', 'anchor_category': 'cabinet',
                           'relation': 'above', 'scope': 'relative_anchor',
                           'phrase': 'above the largest tallest cabinet'}],
            'Observer': {},
            'Attributes': [{'category': 'cabinet', 'phrase': 'largest tallest'}],
            'Unhandled': [],
        })
        result = diagnose(data, objects, objects[2:], [0, 0, .5], supported(objects),
                          trusted_categories=True)
        self.assertNotEqual(result['status'], 'resolved')
        self.assertIn('missing reference category: cabinet', result['missing'])
        records = result['geometry']['size_constraints']
        self.assertEqual([(row['metric'], row['selected_ids'], row['role'])
                          for row in records],
                         [('volume_m3', [1], 'reference'),
                          ('height_m', [2], 'reference')])

    def test_deferred_size_conditions_are_pending_before_scope_binding(self):
        objects = [obj(1, 'cabinet', 0, 0), obj(2, 'cabinet', 1, 0)]
        data = validate_structure('the red largest tallest cabinet', {
            'Target': ['cabinet'], 'Anchor': [], 'Relations': [], 'Observer': {},
            'Attributes': [{'category': 'cabinet', 'phrase': 'red largest tallest'}],
            'Unhandled': [],
        })
        result = diagnose(data, objects, objects, [0, 0, .5],
                          trusted_categories=True, apply_size_constraints=False)
        self.assertEqual(result['pending_size_conditions'], [
            {'attribute': 'largest', 'phrase': 'red largest tallest', 'role': 'target'},
            {'attribute': 'tallest', 'phrase': 'red largest tallest', 'role': 'target'},
        ])
        self.assertEqual(result['status'], 'missing')
        self.assertIn('target size conditions lack complete evidence', result['missing'])
        self.assertTrue(all(c['comparison_pool_ids'] == [1, 2]
                            for c in result['geometry']['size_constraints']))
        self.assertIn('category/appearance evidence is uncertain or unreviewed', result['missing'])

    def test_deferred_size_keeps_floor_predicate_applied(self):
        objects = [obj(1, 'cabinet', 0, 0), obj(2, 'cabinet', 1, 0)]
        objects[1]['bbox_3d'][2] = 2.
        data = validate_structure('the cabinet on the floor', {
            'Target': ['cabinet'], 'Anchor': [], 'Relations': [], 'Observer': {},
            'Attributes': [{'category': 'cabinet', 'phrase': 'on the floor'}],
            'Unhandled': [],
        })
        result = diagnose(data, objects, objects, [0, 0, .5], supported(objects),
                          trusted_categories=True, apply_size_constraints=False)
        self.assertEqual(result['candidate_ids'], [1])
        self.assertEqual(result['pending_size_conditions'], [])
        self.assertEqual(result['geometry']['floor_constraint']['selected_ids'], [1])

    def test_ordinary_small_uses_frozen_relative_comparison_without_rewriting_source(self):
        objects=[obj(1,'cabinet',0,0), obj(2,'cabinet',1,0)]
        objects[0]['bbox_3d'][3:] = [.4,.4,.5]
        objects[1]['bbox_3d'][3:] = [1.,1.,2.]
        raw={'Target':['cabinet'],'Anchor':[],'Relations':[],
             'Observer':{},'Attributes':[{'category':'cabinet','phrase':'small'}], 'Unhandled':[]}
        data=validate_structure('the small cabinet', raw)
        result=diagnose(data, objects, objects, [0,0,.5], supported(objects),
                        trusted_categories=True, size_measurements=synthetic_oriented_sizes(objects))
        condition, = result['geometry']['size_constraints']
        self.assertEqual(condition['comparison_pool_ids'], [1, 2])
        self.assertEqual(condition['original_quote'], 'small')
        self.assertEqual(condition['form'], 'adjective')
        self.assertEqual(condition['interpretation'], 'contextual_set_comparison')
        self.assertEqual({c['candidate_id']: c['status'] for c in condition['candidate_evidence']},
                         {1: 'pass', 2: 'fail'})
        self.assertEqual(result['candidate_ids'], [1])
        self.assertEqual(result['status'], 'resolved')
        unmeasured = diagnose(data, objects, objects, [0, 0, .5],
                              trusted_categories=True)
        self.assertEqual(unmeasured['candidate_ids'], [1, 2])
        self.assertEqual(unmeasured['status'], 'missing')
        self.assertTrue(all(c['status'] == 'unknown'
                            for c in unmeasured['geometry']['size_constraints'][0]['candidate_evidence']))

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

    def test_size_and_color_use_independent_full_category_conditions(self):
        objects = [obj(1, 'cabinet', 0, 0), obj(2, 'cabinet', 2, 0)]
        objects[0]['bbox_3d'][3:] = [.8, .8, 1.0]
        objects[1]['bbox_3d'][3:] = [1.2, 1.2, 1.0]
        raw = {'Target': ['cabinet'], 'Anchor': [], 'Relations': [], 'Observer': {},
               'Attributes': [{'category': 'cabinet', 'phrase': 'red largest'}],
               'Unhandled': []}
        query = 'Choose the red largest cabinet.'
        data = validate_structure(query, raw)
        missing_scope = diagnose(data, objects, objects, [0, 0, .5],
                                 trusted_categories=True)
        self.assertEqual(missing_scope['status'], 'missing')
        self.assertEqual(missing_scope['geometry']['size_constraints'][0]['comparison_pool_ids'], [1, 2])
        self.assertEqual(missing_scope['geometry']['size_constraints'][0]['selected_ids'], [2])
        self.assertIn('category/appearance evidence is uncertain or unreviewed', missing_scope['missing'])

        checks = [
            {'id': 1, 'category': 'cabinet', 'verdict': 'yes',
             'attribute_checks': [{'attribute': 'red', 'verdict': 'yes',
                                  'reason': 'red finish is visible', 'visibility': 'visible'}],
             'reason': 'cabinet and red finish are visible', 'visibility': 'visible'},
            {'id': 2, 'category': 'cabinet', 'verdict': 'yes',
             'attribute_checks': [{'attribute': 'red', 'verdict': 'no',
                                  'reason': 'finish is blue', 'visibility': 'visible'}],
             'reason': 'cabinet is visible', 'visibility': 'visible'},
        ]
        bound = diagnose(data, objects, objects, [0, 0, .5], checks,
                         trusted_categories=True)
        self.assertEqual(bound['candidate_ids'], [])
        self.assertEqual(bound['status'], 'conflict')
        self.assertEqual(bound['geometry']['size_constraints'][0]['comparison_pool_ids'], [1, 2])
        self.assertEqual(bound['geometry']['size_constraints'][0]['selected_ids'], [2])

    def test_explicit_color_subset_requires_all_members_before_size_comparison(self):
        objects = [obj(1, 'cabinet', 0, 0), obj(2, 'cabinet', 2, 0),
                   obj(3, 'cabinet', 4, 0)]
        for item, width in zip(objects, (.8, 1.2, 2.)):
            item['bbox_3d'][3:] = [width, width, 1.]
        query = 'Choose the largest of the red cabinets.'
        data = validate_structure(query, {
            'Target': ['cabinet'], 'Anchor': [], 'Relations': [], 'Observer': {},
            'Attributes': [{'category': 'cabinet', 'phrase': 'largest'},
                           {'category': 'cabinet', 'phrase': 'red'}], 'Unhandled': [],
        })
        checks = [
            {'id': item['bbox_id'], 'category': 'cabinet', 'role': 'target',
             'verdict': 'yes', 'visibility': 'visible', 'reason': 'cabinet visible',
             'attribute_checks': [{'attribute': 'red',
                                   'verdict': 'no' if item['bbox_id'] == 3 else 'yes',
                                   'visibility': 'visible', 'reason': 'surface color visible'}]}
            for item in objects]
        unbound = diagnose(data, objects, objects, [0, 0, .5], checks[:2],
                           trusted_categories=True)
        self.assertEqual(unbound['status'], 'missing')
        self.assertEqual(unbound['candidate_ids'], [1, 2, 3])
        condition, = unbound['geometry']['size_constraints']
        self.assertEqual(condition['scope_kind'], 'attribute_subset')
        self.assertEqual(condition['scope_status'], 'unknown')
        self.assertEqual(condition['scope_unknown_ids'], [3])
        self.assertTrue(all(c['status'] == 'unknown' for c in condition['candidate_evidence']))
        bound = diagnose(data, objects, objects, [0, 0, .5], checks,
                         trusted_categories=True)
        self.assertEqual(bound['status'], 'resolved')
        self.assertEqual(bound['candidate_ids'], [2])
        self.assertEqual(bound['geometry']['size_constraints'][0]['comparison_pool_ids'], [1, 2])


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
            with patch('rgb_review.chat',return_value={
                    '_content': json.dumps({'id': 1, 'verdict': 'yes',
                                            'category_verdict': 'yes',
                                            'visibility': 'visible', 'reason': 'visible window',
                                            'attribute_checks': []}),
                    '_baseline': {'model': 'mock', 'latency_s': .01},
                    '_finish_reason': 'stop', 'usage': {}}) as api:
                result=selective_review(parsed('window',[]),objects,objects[:1],[0,0,.5],root,calls,2)
            self.assertEqual(api.call_count,1)
            self.assertEqual(result['status'],'resolved')
            self.assertEqual(result['stable_candidate_ids'],[1])
            self.assertEqual(calls[0]['stage'],'rgb_review')

    def test_uncertain_or_bad_binding_answer_not_repeated(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);objects,_=self.fixture(root)
            for answer in ({'_content': json.dumps({'id': 1, 'verdict': 'uncertain',
                                                        'visibility': 'unknown', 'reason': 'unclear',
                                                        'attribute_checks': []}),
                                '_baseline': {'model': 'mock', 'latency_s': .01},
                                '_finish_reason': 'stop', 'usage': {}},
                               response(2), response(1,finish='length')):
                with patch('rgb_review.chat',return_value=answer) as api:
                    result=selective_review(parsed('window',[]),objects,objects[:1],[0,0,.5],root,[],6)
                self.assertEqual(api.call_count,1)
                self.assertEqual(result['status'],'missing')
                self.assertEqual(result['checks'][0]['verdict'],'uncertain')

    def test_non_normal_rgb_completion_keeps_visible_positive_unknown(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);objects,_=self.fixture(root)
            for finish in ('length', 'content_filter', 'tool_calls', 'error', None):
                with self.subTest(finish=finish):
                    answer={'_content':json.dumps({'id':1,'verdict':'yes',
                                                  'category_verdict':'yes',
                                                  'visibility':'visible','reason':'visible window',
                                                  'attribute_checks':[]}),
                            '_baseline':{'model':'mock','latency_s':.01},
                            '_finish_reason':finish,'usage':{}}
                    calls=[]
                    with patch('rgb_review.chat',return_value=answer) as api:
                        result=selective_review(parsed('window',[]),objects,objects[:1],
                                                [0,0,.5],root,calls,6)
                    self.assertEqual(api.call_count,1)
                    self.assertEqual(result['status'],'missing')
                    self.assertEqual(result['checks'][0]['verdict'],'uncertain')
                    self.assertEqual(result['checks'][0]['reason'],'RGB review failed; no conclusion')
                    self.assertIn('did not complete normally',result['checks'][0]['error'])
                    self.assertEqual(calls[0]['finish_reason'],finish)

    def test_rgb_api_error_is_recorded_without_retrying_same_need(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);objects,_=self.fixture(root);calls=[]
            with patch('rgb_review.chat',side_effect=QwenAPIError('upstream API error')) as api:
                result=selective_review(parsed('window',[]),objects,objects[:1],
                                        [0,0,.5],root,calls,6)
            self.assertEqual(api.call_count,1)
            self.assertEqual(result['status'],'missing')
            self.assertEqual(result['checks'][0]['verdict'],'uncertain')
            self.assertEqual(result['checks'][0]['reason'],'RGB review failed; no conclusion')
            self.assertEqual(result['checks'][0]['error'],'upstream API error')
            self.assertEqual(calls[0]['error'],'upstream API error')

    def test_rgb_attribute_verdicts_keep_visibility_and_projection_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);objects,_=self.fixture(root)
            data=parsed('black window',[],attrs=[{'category':'window','phrase':'black'}])
            for verdict,visibility,status in [('yes','visible','resolved'),
                                              ('no','visible','conflict'),
                                              ('yes','occluded','missing'),
                                              ('yes','low_detail','missing')]:
                with self.subTest(verdict=verdict,visibility=visibility):
                    answer={'id':1,'verdict':'yes','category_verdict':'yes',
                            'visibility':'visible','reason':'window visible',
                            'attribute_checks':[{'attribute':'black','verdict':verdict,
                                                 'visibility':visibility,'reason':'surface color evidence'}]}
                    with patch('rgb_review.chat',return_value={
                            '_content':json.dumps(answer),
                            '_baseline':{'model':'mock','latency_s':.01},
                            '_finish_reason':'stop','usage':{}}) as api:
                        result=selective_review(data,objects,objects[:1],[0,0,.5],root,[],2,
                                                trusted_categories=True)
                    self.assertEqual(result['status'],status)
                    check=result['checks'][0]
                    self.assertEqual(check['attribute_checks'][0]['visibility'],visibility)
                    self.assertEqual(check['render_provenance']['binding'],'projected_instance')
                    self.assertEqual(check['render_provenance']['source_frames'],['0001.jpg'])
                    self.assertEqual(check['render_provenance']['source_type'],'real_rgb_projected_instance')
                    self.assertFalse(check['render_provenance']['language_frame_changed'])
                    prompt=api.call_args.args[0][0]['content'][-1]['text']
                    self.assertIn('attribute_checks',prompt)

    def test_rgb_generic_positive_keeps_unreported_attribute_missing(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);objects,_=self.fixture(root)
            data=parsed('black window',[],attrs=[{'category':'window','phrase':'black'}])
            with patch('rgb_review.chat',return_value=response(1)):
                result=selective_review(data,objects,objects[:1],[0,0,.5],root,[],2,
                                        trusted_categories=True)
            self.assertEqual(result['status'],'missing')
            self.assertEqual(result['checks'][0]['attribute_checks'][0]['verdict'],'uncertain')

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
