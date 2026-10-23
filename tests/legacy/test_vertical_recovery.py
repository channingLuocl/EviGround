import unittest

from eviground.grounding.diagnostic import diagnose, recover_vertical_relations, refresh_condition_records, validate_structure
from eviground.query.query_rewrite import rewrite_query


class VerticalRecoveryTests(unittest.TestCase):
    def parsed(self, relations):
        return {'Target': ['shelf'], 'Anchor': ['shelf'], 'Relations': relations}

    def test_embedded_reference_above_target_is_inverted(self):
        query = 'The largest shelf with a shelf above it.'
        data = recover_vertical_relations(query, self.parsed([{
            'target_category': 'shelf', 'anchor_category': 'shelf',
            'relation': 'above', 'scope': 'relative_anchor', 'phrase': query,
        }]))
        self.assertEqual(data['Relations'][0]['relation'], 'below')

    def test_embedded_reference_is_recovered_when_relation_was_omitted(self):
        query = 'The largest shelf with a shelf above it.'
        data = recover_vertical_relations(query, self.parsed([]))
        self.assertEqual(data['Relations'][0]['relation'], 'below')
        self.assertEqual(data['Relations'][0]['phrase'], 'shelf above it')

    def test_reference_below_target_is_inverted(self):
        query = 'The largest shelf with a shelf below it.'
        data = recover_vertical_relations(query, self.parsed([]))
        self.assertEqual(data['Relations'][0]['relation'], 'above')

    def test_underneath_is_normalized_to_below(self):
        query = 'the table underneath the shelf'
        raw = {'Target': ['table'], 'Anchor': ['shelf'], 'Relations': []}
        data = recover_vertical_relations(query, raw)
        self.assertEqual(data['Relations'][0]['relation'], 'below')
        self.assertEqual(data['Relations'][0]['phrase'], 'underneath the shelf')

    def test_mixed_underneath_clause_keeps_its_explicit_reference(self):
        query = ('The thin wooden table underneath the television and immediately '
                 'to the left of the trash can.')
        raw = {'Target': ['table'], 'Anchor': ['television', 'trash can'],
               'Relations': [
                   {'target_category': 'table', 'anchor_category': 'television',
                    'relation': 'underneath', 'scope': '', 'phrase': 'underneath the television'},
                   {'target_category': 'table', 'anchor_category': 'trash can',
                    'relation': 'left', 'scope': 'relative_anchor',
                    'phrase': 'immediately to the left of the trash can'},
               ], 'Observer': {}, 'Attributes': [{'category': 'table', 'phrase': 'thin wooden'}],
               'Unhandled': []}
        data = validate_structure(query, raw)
        self.assertEqual([(r['relation'], r['anchor_category']) for r in data['Relations']],
                         [('below', 'television'), ('left', 'trash can')])
        self.assertEqual(data['Relations'][0]['scope'], 'relative_anchor')
        self.assertNotIn('unsupported relation: underneath the television', data['diagnostic_missing'])
        self.assertNotIn('vertical condition not represented', data['diagnostic_missing'])
        refresh_condition_records(query, data)
        immediate = [record for record in data['condition_records']
                     if record['condition_type'] == 'relation:immediate_proximity']
        self.assertEqual(len(immediate), 1)
        self.assertEqual(immediate[0]['source_quote'], 'immediately to the left')
        self.assertEqual(immediate[0]['target_role'], {'category': 'table'})
        self.assertEqual(immediate[0]['reference_role'], {'categories': ['trash can']})
        self.assertEqual(immediate[0]['binding_status'], 'bound')
        self.assertEqual(immediate[0]['evidence_status'], 'missing')
        rewritten = rewrite_query(query, data)
        self.assertTrue(any(c['type'] == 'below' and c['reference'] == 'tv'
                            for c in rewritten['constraints']))
        self.assertIn({'phrase': 'immediately to the left',
                       'reason': 'immediate proximity has no calibrated predicate'},
                      rewritten['unresolved'])

    def test_multiple_references_do_not_fill_an_omitted_vertical_anchor(self):
        query = 'The table underneath the television and to the left of the trash can.'
        raw = {'Target': ['table'], 'Anchor': ['television', 'trash can'],
               'Relations': [{'target_category': 'table', 'relation': 'underneath',
                              'phrase': 'underneath the television'}],
               'Observer': {}, 'Attributes': [], 'Unhandled': []}
        data = validate_structure(query, raw)
        self.assertFalse(any(r['relation'] == 'below' for r in data['Relations']))
        self.assertIn('unsupported relation: underneath the television', data['diagnostic_missing'])

    def test_immediate_left_blocks_acceptance_even_with_a_known_frame(self):
        for modifier, expected_status in (('', 'resolved'), ('immediately ', 'missing')):
            with self.subTest(modifier=modifier):
                query = ('Standing in the middle of the room facing the tv, '
                         'choose the table ' + modifier + 'to the left of the tv.')
                raw = {'Target': ['table'], 'Anchor': ['tv'],
                       'Relations': [{'target_category': 'table', 'anchor_category': 'tv',
                                      'relation': 'left', 'scope': 'relative_anchor',
                                      'phrase': 'to the left of the tv'}],
                       'Observer': {'origin': {'kind': 'room_center',
                                               'phrase': 'Standing in the middle of the room'},
                                    'facing': {'category': 'tv', 'group': False,
                                               'phrase': 'facing the tv'}},
                       'Attributes': [], 'Unhandled': []}
                data = validate_structure(query, raw)
                objects = [
                    {'bbox_id': 1, 'target': 'table', 'bbox_3d': [-1., 2., .5, .4, .4, 1.]},
                    {'bbox_id': 2, 'target': 'table', 'bbox_3d': [1., 2., .5, .4, .4, 1.]},
                    {'bbox_id': 3, 'target': 'tv', 'bbox_3d': [0., 2., .5, .4, .4, 1.]},
                ]
                result = diagnose(data, objects, objects[:2], [0., 0., .5], trusted_categories=True)
                self.assertEqual(result['candidate_ids'], [1])
                self.assertEqual(result['status'], expected_status)
                if modifier:
                    self.assertIn('immediate proximity has no calibrated predicate: immediately to the left',
                                  result['missing'])
                    record = next(record for record in data['condition_records']
                                  if record['condition_type'] == 'relation:immediate_proximity')
                    self.assertEqual(record['binding_status'], 'bound')
                    self.assertEqual(record['reference_role'], {'categories': ['tv']})

    def test_short_pronoun_quote_uses_full_query_direction(self):
        for operator in ('under', 'underneath', 'beneath', 'above'):
            with self.subTest(operator=operator):
                query = 'The table with white chairs ' + operator + ' it'
                raw = {'Target': ['table'], 'Anchor': ['chairs'], 'Relations': [{
                    'target_category': 'table', 'anchor_category': 'chairs',
                    'relation': operator, 'scope': '', 'phrase': operator + ' it'}]}
                data = recover_vertical_relations(query, raw)
                self.assertEqual(data['Relations'][0]['relation'],
                                 'below' if operator == 'above' else 'above')
                self.assertEqual(data['Relations'][0]['phrase'], operator + ' it')
                self.assertEqual(data['Relations'][0]['source_subject'], 'chairs')

    def test_short_pronoun_quote_does_not_reuse_another_operator(self):
        query = 'The cabinet with a shelf above it and a shelf under it.'
        raw = {'Target': ['cabinet'], 'Anchor': ['shelf'], 'Relations': [
            {'target_category': 'cabinet', 'anchor_category': 'shelf',
             'relation': 'above', 'phrase': 'above it'},
            {'target_category': 'cabinet', 'anchor_category': 'shelf',
             'relation': 'below', 'phrase': 'under it'},
        ], 'Observer': {}, 'Attributes': [], 'Unhandled': []}
        data = validate_structure(query, raw)
        self.assertEqual(data['Relations'][0]['relation'], 'below')
        self.assertTrue(any('unsupported relation: under it' in reason
                            for reason in data['diagnostic_missing']))
        objects = [
            {'bbox_id': 1, 'target': 'cabinet', 'bbox_3d': [0., 0., .5, 1., 1., 1.]},
            {'bbox_id': 2, 'target': 'shelf', 'bbox_3d': [0., 0., 2.5, 1., 1., 1.]},
        ]
        result = diagnose(data, objects, objects[:1], [0., 0., .5], trusted_categories=True)
        self.assertEqual(result['status'], 'missing')

    def test_repeated_short_quote_has_no_unique_source_clause(self):
        query = 'The cabinet with a shelf above it and a shelf above it.'
        raw = {'Target': ['cabinet'], 'Anchor': ['shelf'], 'Relations': [{
            'target_category': 'cabinet', 'anchor_category': 'shelf',
            'relation': 'above', 'phrase': 'above it'}],
            'Observer': {}, 'Attributes': [], 'Unhandled': []}
        data = validate_structure(query, raw)
        self.assertEqual(data['Relations'], [])
        self.assertIn('unsupported relation: above it', data['diagnostic_missing'])

    def test_short_quote_without_bound_subject_stays_missing(self):
        query = 'The cabinet is under it.'
        raw = {'Target': ['cabinet'], 'Anchor': ['shelf'], 'Relations': [{
            'target_category': 'cabinet', 'anchor_category': 'shelf',
            'relation': 'below', 'phrase': 'under it'}],
            'Observer': {}, 'Attributes': [], 'Unhandled': []}
        data = validate_structure(query, raw)
        self.assertEqual(data['Relations'], [])
        self.assertIn('unsupported relation: under it', data['diagnostic_missing'])

    def test_duplicate_reference_labels_do_not_block_vertical_scope(self):
        query = 'the cabinets above the big cabinet'
        raw = {'Target': ['cabinet'], 'Anchor': ['cabinet', 'cabinet'], 'Relations': [{
            'target_category': 'cabinet', 'anchor_category': 'cabinet',
            'relation': 'above', 'scope': '', 'phrase': 'above the big cabinet'}]}
        self.assertEqual(recover_vertical_relations(query, raw)['Relations'][0]['scope'],
                         'relative_anchor')

    def test_target_above_reference_keeps_its_direction(self):
        query = 'cabinet above the refrigerator'
        raw = {'Target': ['cabinet'], 'Anchor': ['refrigerator'], 'Relations': []}
        data = recover_vertical_relations(query, raw)
        self.assertEqual(data['Relations'][0]['relation'], 'above')
        self.assertEqual(data['Relations'][0]['phrase'], 'above the refrigerator')

    def test_later_clause_is_not_inverted_into_target_relation(self):
        query = 'Choose the cabinet above the shelf with a painting above it.'
        raw = {'Target': ['cabinet'], 'Anchor': ['shelf'], 'Relations': []}
        data = recover_vertical_relations(query, raw)
        self.assertEqual(data['Relations'][0]['relation'], 'above')

    def test_later_clause_does_not_invert_existing_relation_phrase(self):
        query = 'Choose the cabinet above the shelf with a painting above it.'
        raw = self.parsed([{
            'target_category': 'shelf', 'anchor_category': 'shelf',
            'relation': 'above', 'scope': 'relative_anchor', 'phrase': query,
        }])
        data = recover_vertical_relations(query, raw)
        self.assertEqual(data['Relations'][0]['relation'], 'above')

    def test_later_same_category_clause_does_not_override_main_relation(self):
        query = 'Choose the shelf above the shelf, with a shelf above it.'
        raw = {'Target': ['shelf'], 'Anchor': ['shelf'], 'Relations': []}
        data = recover_vertical_relations(query, raw)
        self.assertEqual(data['Relations'][0]['relation'], 'above')


if __name__ == '__main__':
    unittest.main()
