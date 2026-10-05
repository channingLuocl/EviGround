import unittest

from query_rewrite import constraints_as_relations, rewrite_query


class QueryRewriteTests(unittest.TestCase):
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

    def test_rejected_relation_is_retained_as_unresolved(self):
        query = 'cabinet between the stove and the wall'
        result = rewrite_query(query, {
            'Target': ['cabinet'], 'Anchor': ['stove', 'wall'], 'Relations': [],
            'Attributes': [], 'Unhandled': [], 'Observer': {},
            'diagnostic_missing': ['unsupported relation: between the stove and the wall'],
        })
        self.assertEqual(result['unresolved'][0]['phrase'], 'between the stove and the wall')


if __name__ == '__main__':
    unittest.main()
