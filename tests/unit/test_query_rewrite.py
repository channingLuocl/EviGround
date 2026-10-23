import unittest

from eviground.query.query_rewrite import constraints_as_relations, rewrite_query


class QueryRewriteTests(unittest.TestCase):
    def test_compiles_comparative_size_attributes(self):
        for word, expected in (('biggest', 'size_max'), ('larger', 'size_max'),
                               ('smaller', 'size_min'),
                               ('bigger', 'size_max'), ('taller', 'height_max'),
                               ('shorter', 'height_min')):
            with self.subTest(word=word):
                query = f'the {word} cabinet'
                result = rewrite_query(query, {
                    'Target': ['cabinet'], 'Anchor': [], 'Relations': [],
                    'Attributes': [{'category': 'cabinet', 'phrase': word}],
                    'Unhandled': [],
                })
                self.assertEqual([item['type'] for item in result['constraints']], [expected])

    def test_omitted_shape_remains_unresolved_without_invented_binding(self):
        from eviground.grounding.diagnostic import validate_structure
        query = 'Choose the skinny cabinet between the stove and the wall.'
        parsed = validate_structure(query, {
            'Target': ['cabinet'], 'Anchor': ['stove', 'wall'],
            'Relations': [{'target_category': 'cabinet', 'anchor_categories': ['stove', 'wall'],
                           'relation': 'between', 'scope': '',
                           'phrase': 'the skinny cabinet between the stove and the wall'}],
            'Attributes': [], 'Observer': {}, 'Unhandled': [],
        })
        result = rewrite_query(query, parsed)
        self.assertEqual([c['type'] for c in result['constraints']], ['between'])
        self.assertEqual(result['unresolved'][0]['phrase'], 'skinny')
        self.assertIn('role not inferred', result['unresolved'][0]['reason'])

    def test_containment_attribute_stays_unresolved(self):
        query = 'The large L shaped kitchen counter that includes the sink and the dishwasher.'
        result = rewrite_query(query, {
            'Target': ['kitchen counter'], 'Anchor': [], 'Relations': [],
            'Attributes': [{'category': 'kitchen counter', 'phrase': 'large L shaped'},
                           {'category': 'kitchen counter', 'phrase': 'that includes the sink and the dishwasher'}],
        })
        self.assertEqual([c['value'] for c in result['constraints']], ['large L shaped'])
        self.assertEqual(result['unresolved'][0]['phrase'], 'that includes the sink and the dishwasher')
        self.assertIn('unsupported relation predicate', result['unresolved'][0]['reason'])

    def test_omitted_containment_is_still_audited(self):
        query = 'counter containing the sink'
        result = rewrite_query(query, {'Target': ['counter'], 'Anchor': [], 'Relations': [], 'Attributes': []})
        self.assertEqual(result['unresolved'][0]['phrase'], 'containing the sink')

    def test_between_keeps_two_reference_bindings(self):
        query = 'the cabinet between the stove and the wall'
        result = rewrite_query(query, {
            'Target': ['cabinet'], 'Anchor': ['stove', 'wall'],
            'Relations': [{'target_category': 'cabinet',
                           'anchor_categories': ['stove', 'wall'],
                           'relation': 'between', 'scope': 'relative_anchors',
                           'phrase': 'between the stove and the wall'}],
            'Attributes': [], 'Unhandled': [], 'Observer': {},
        })
        self.assertEqual(result['constraints'][0]['type'], 'between')
        self.assertEqual(result['constraints'][0]['references'], ['stove', 'wall'])
        relation = constraints_as_relations(result)[0]
        self.assertEqual(relation['anchor_categories'], ['stove', 'wall'])

    def test_category_alias_is_normalized_and_kept_for_audit(self):
        query = 'the cabinet above the countertop'
        result = rewrite_query(query, {
            'Target': ['cabinet'], 'Anchor': ['countertop'],
            'Relations': [{'target_category': 'cabinet', 'anchor_category': 'countertop',
                           'relation': 'above', 'scope': 'relative_anchor',
                           'phrase': 'above the countertop'}],
            'Attributes': [], 'Unhandled': [], 'Observer': {},
        })
        self.assertEqual(result['references'][0]['category'], 'counter')
        self.assertEqual(result['references'][0]['raw_category'], 'countertop')
        self.assertEqual(result['constraints'][0]['reference'], 'counter')
        self.assertEqual(result['constraints'][0]['raw_reference_category'], 'countertop')

    def test_normalizes_vertical_relation_without_turning_it_into_near(self):
        query = 'brown cabinet with two handles right above the refrigerator'
        parsed = {
            'Target': ['cabinet'], 'Anchor': ['refrigerator'],
            'Relations': [{'target_category': 'cabinet', 'anchor_category': 'refrigerator',
                           'relation': 'above', 'scope': 'relative_anchor',
                           'phrase': 'right above the refrigerator'}],
            'Attributes': [{'category': 'cabinet', 'phrase': 'brown'}],
            'Unhandled': [], 'Observer': {},
        }
        result = rewrite_query(query, parsed)
        self.assertEqual([c['type'] for c in result['constraints']], ['above', 'attribute'])
        self.assertEqual(constraints_as_relations(result)[0]['relation'], 'above')
        self.assertNotIn('near', [c['type'] for c in result['constraints']])

    def test_normalizes_underneath_relation_to_below(self):
        query = 'the table underneath the shelf'
        parsed = {
            'Target': ['table'], 'Anchor': ['shelf'],
            'Relations': [{'target_category': 'table', 'anchor_category': 'shelf',
                           'relation': 'underneath', 'scope': 'relative_anchor',
                           'phrase': 'underneath the shelf'}],
            'Attributes': [], 'Unhandled': [], 'Observer': {},
        }
        result = rewrite_query(query, parsed)
        self.assertEqual(result['constraints'][0]['type'], 'below')
        self.assertEqual(constraints_as_relations(result)[0]['relation'], 'below')

    def test_compiles_size_and_floor_as_explicit_geometry_constraints(self):
        query = 'the smallest cabinet on the floor right of the stove'
        parsed = {
            'Target': ['cabinet'], 'Anchor': ['stove'],
            'Relations': [{'target_category': 'cabinet', 'anchor_category': 'stove',
                           'relation': 'right', 'scope': 'relative_anchor',
                           'phrase': 'right of the stove'}],
            'Attributes': [{'category': 'cabinet', 'phrase': 'smallest'},
                           {'category': 'cabinet', 'phrase': 'on the floor'}],
            'Unhandled': [], 'Observer': {},
        }
        result = rewrite_query(query, parsed)
        self.assertEqual([c['type'] for c in result['constraints']],
                         ['right_of', 'size_min', 'bottom_z_near_floor'])
        self.assertEqual(result['constraints'][1]['metric'], 'volume_m3')

    def test_near_remains_near_and_unhandled_is_audit_only(self):
        query = 'cabinet next to the sink between the stove and wall'
        parsed = {
            'Target': ['cabinet'], 'Anchor': ['sink', 'stove', 'wall'],
            'Relations': [{'target_category': 'cabinet', 'anchor_category': 'sink',
                           'relation': 'near', 'scope': '', 'phrase': 'next to the sink'}],
            'Attributes': [], 'Unhandled': ['between the stove and wall'], 'Observer': {},
        }
        result = rewrite_query(query, parsed)
        self.assertEqual(result['constraints'][0]['type'], 'near')
        self.assertEqual(result['unresolved'][0]['phrase'], 'between the stove and wall')
        self.assertIn('wall', [item['category'] for item in result['references']])

    def test_distance_metric_is_preserved_through_rewrite(self):
        query = 'the chair nearest the table'
        parsed = {
            'Target': ['chair'], 'Anchor': ['table'],
            'Relations': [{'target_category': 'chair', 'anchor_category': 'table',
                           'relation': 'nearest', 'distance_metric': 'horizontal_gap',
                           'scope': '', 'phrase': 'nearest the table'}],
            'Attributes': [], 'Unhandled': [], 'Observer': {},
        }
        rewritten = rewrite_query(query, parsed)
        self.assertEqual(rewritten['constraints'][0]['distance_metric'], 'horizontal_gap')
        restored = constraints_as_relations(rewritten)[0]
        self.assertEqual(restored['distance_metric'], 'horizontal_gap')

    def test_distance_bindings_tolerance_and_metric_alias_round_trip(self):
        query = 'the chair nearest the table'
        metadata = {'metric': 'horizontal', 'distance_scope': 'horizontal',
                    'target_instance_id': 2, 'anchor_id': 10,
                    'tie_tolerance_m': .02}
        relation = {'target_category': 'chair', 'anchor_category': 'table',
                    'relation': 'nearest', 'phrase': 'nearest the table',
                    **metadata}
        rewritten = rewrite_query(query, {
            'Target': ['chair'], 'Anchor': ['table'], 'Relations': [relation],
            'Attributes': [], 'Unhandled': [], 'Observer': {},
        })
        restored = constraints_as_relations(rewritten)[0]
        for key, value in metadata.items():
            self.assertEqual(restored[key], value)

    def test_rejected_relation_is_retained_as_unresolved(self):
        query = 'cabinet between the stove and the wall'
        result = rewrite_query(query, {
            'Target': ['cabinet'], 'Anchor': ['stove', 'wall'], 'Relations': [],
            'Attributes': [], 'Unhandled': [], 'Observer': {},
            'diagnostic_missing': ['unsupported relation: between the stove and the wall'],
        })
        self.assertEqual(result['unresolved'][0]['phrase'], 'between the stove and the wall')

    def test_malformed_observer_fields_are_safe_and_explicitly_empty(self):
        result = rewrite_query('choose a chair', {
            'Target': ['chair'], 'Anchor': [], 'Relations': [], 'Attributes': [],
            'Observer': {'origin': None, 'facing': 'not-a-frame'},
        })
        self.assertEqual(result['language_reference']['origin'], {})
        self.assertEqual(result['language_reference']['facing'], {})

    def test_intrinsic_direction_is_unresolved_in_raw_rewrite(self):
        query = 'the window in front of the chair'
        result = rewrite_query(query, {
            'Target': ['window'], 'Anchor': ['chair'],
            'Relations': [{'target_category': 'window', 'anchor_category': 'chair',
                           'relation': 'front', 'scope': 'relative_anchor',
                           'reference_frame': 'intrinsic',
                           'phrase': 'in front of the chair'}],
            'Attributes': [], 'Observer': {},
        })
        self.assertEqual(result['constraints'], [])
        self.assertEqual(result['unresolved'][0]['phrase'], 'in front of the chair')
        self.assertIn('orientation is not inferred', result['unresolved'][0]['reason'])

    def test_facing_chair_backs_keeps_group_and_unresolved_orientation(self):
        query = 'Facing the back of the chairs, the chair on the left.'
        result = rewrite_query(query, {
            'Target': ['chair'], 'Anchor': [],
            'Relations': [{'target_category': 'chair', 'anchor_category': '',
                           'relation': 'left', 'scope': 'side', 'phrase': 'the chair on the left'}],
            'Attributes': [], 'Observer': {'facing': {
                'category': 'chair', 'group': False, 'phrase': 'Facing the back of the chairs'}},
        })
        self.assertEqual(result['unresolved'][0]['phrase'], 'Facing the back of the chairs')
        self.assertIn('orientation is not inferred', result['unresolved'][0]['reason'])
        facing = result['language_reference']['facing']
        self.assertTrue(facing['group'])
        self.assertEqual(facing['reference_frame'], 'intrinsic')
        self.assertEqual(facing['orientation_part'], 'back')
        self.assertEqual(result['constraints'][0]['scope'], 'side')

    def test_side_direction_without_anchor_roundtrips(self):
        query = 'the window on the right'
        result = rewrite_query(query, {
            'Target': ['window'], 'Anchor': [],
            'Relations': [{'target_category': 'window', 'anchor_category': '',
                           'relation': 'right', 'scope': 'side',
                           'phrase': 'on the right'}],
            'Attributes': [], 'Observer': {},
        })
        self.assertEqual(result['constraints'][0]['scope'], 'side')
        restored = constraints_as_relations(result)[0]
        self.assertEqual(restored['anchor_category'], '')
        self.assertEqual(restored['reference_frame'], 'observer')

    def test_facing_bed_head_and_foot_preserve_intrinsic_part(self):
        for part in ('head', 'foot'):
            with self.subTest(part=part):
                quote = f'Facing the {part} of the bed'
                result = rewrite_query(quote + ', the dresser on the right.', {
                    'Target': ['dresser'], 'Anchor': ['bed'], 'Relations': [],
                    'Attributes': [], 'Observer': {'facing': {
                        'category': 'bed', 'phrase': 'Facing the bed', 'group': False}},
                })
                facing = result['language_reference']['facing']
                self.assertEqual(facing['phrase'], quote)
                self.assertEqual(facing['orientation_part'], part)
                self.assertEqual(facing['reference_frame'], 'intrinsic')
                self.assertEqual(result['unresolved'][0]['phrase'], quote)


if __name__ == '__main__':
    unittest.main()
