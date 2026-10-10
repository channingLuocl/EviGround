import unittest

from diagnostic import diagnose, validate_structure


def parsed_query(query, phrase, scope):
    return {
        'Target': ['cabinet'], 'Anchor': ['door'],
        'Relations': [{'target_category': 'cabinet', 'anchor_category': '',
                       'relation': 'right', 'scope': scope, 'phrase': phrase}],
        'Attributes': [{'category': 'cabinet', 'role': 'target', 'phrase': 'largest'}],
        'Observer': {
            'origin': {'kind': 'room_center', 'phrase': 'in the middle of the room'},
            'facing': {'category': 'door', 'group': False, 'phrase': 'facing the door'},
        },
        'Unhandled': [],
    }


class DirectionScopeTests(unittest.TestCase):
    def group_side_query(self, phrase='the door on the right', facing='facing both doors',
                         origin='Standing in the middle of the room'):
        return {
            'Target': ['door'], 'Anchor': ['doors'],
            'Relations': [{'target_category': 'door', 'anchor_category': 'doors',
                           'anchor_categories': ['door'], 'relation': 'right',
                           'scope': 'relative_anchor', 'phrase': phrase}],
            'Attributes': [], 'Unhandled': [],
            'Observer': {
                'origin': {'kind': 'room_center', 'phrase': origin},
                'facing': {'category': 'doors', 'group': True, 'phrase': facing},
            },
        }

    def test_complete_group_facing_side_does_not_bind_peer_as_anchor(self):
        from query_rewrite import rewrite_query
        query = "Standing in the middle of the room facing both doors, it's the door on the right"
        data = validate_structure(query, self.group_side_query())
        relation = data['Relations'][0]
        self.assertEqual(relation['scope'], 'side')
        self.assertEqual(relation['anchor_category'], '')
        self.assertEqual(relation['anchor_categories'], [])
        record = next(item for item in data['condition_records']
                      if item['condition_type'] == 'relation:right')
        self.assertIsNone(record['reference_role'])
        objects = [
            {'bbox_id': 9, 'target': 'door', 'bbox_3d': [-1., 2., 1., .4, .2, 2.]},
            {'bbox_id': 20, 'target': 'door', 'bbox_3d': [1., 2., 1., .4, .2, 2.]},
        ]
        result = diagnose(data, objects, objects, [0., 0., 1.], trusted_categories=True)
        self.assertEqual(result['status'], 'resolved')
        self.assertEqual(result['candidate_ids'], [20])
        self.assertEqual([h['references'] for h in result['hypotheses']], [{}])
        self.assertEqual(rewrite_query(query, data)['constraints'][0]['scope'], 'side')

    def test_group_facing_scope_recovery_keeps_relative_extreme_and_incomplete_clauses(self):
        for phrase in ('the door on the right of the other door',
                       'the door farther right than the other door',
                       'the door on the far right'):
            with self.subTest(phrase=phrase):
                query = "Standing in the middle of the room facing both doors, it's " + phrase
                data = validate_structure(query, self.group_side_query(phrase))
                if data['Relations']:
                    self.assertEqual(data['Relations'][0]['scope'], 'relative_anchor')
                    self.assertEqual(data['Relations'][0]['anchor_category'], 'doors')
        query = 'Standing at the door facing both doors, the door on the right'
        raw = self.group_side_query(origin='Standing at the door')
        raw['Observer']['origin'] = {'kind': 'category', 'category': 'door',
                                     'phrase': 'Standing at the door'}
        data = validate_structure(query, raw)
        self.assertEqual(data['Relations'][0]['scope'], 'relative_anchor')
        query = 'Standing in the middle of the room facing both doors, the door on the right, beside the cabinet'
        data = validate_structure(query, self.group_side_query())
        self.assertEqual(data['Relations'][0]['scope'], 'relative_anchor')

    def test_explicit_side_scope_cannot_certify_direction_extreme(self):
        objects = [
            {'bbox_id': 1, 'target': 'cabinet', 'bbox_3d': [1., 2., 1., 2., 2., 2.]},
            {'bbox_id': 2, 'target': 'cabinet', 'bbox_3d': [3., 2., .5, 1., 1., 1.]},
            {'bbox_id': 3, 'target': 'door', 'bbox_3d': [0., 4., 1., 1., .2, 2.]},
        ]
        for phrase in ('on the most right', 'on the far right', 'on the rightmost side',
                       'on the furthest right', 'on the farthest right'):
            for scope in ('side', ''):
                with self.subTest(phrase=phrase, scope=scope):
                    query = ('Choose the largest cabinet ' + phrase +
                             ', standing in the middle of the room facing the door.')
                    data = validate_structure(query, parsed_query(query, phrase, scope))
                    self.assertEqual(data['Relations'], [])
                    self.assertIn('unsupported relation: ' + phrase, data['diagnostic_missing'])
                    result = diagnose(data, objects, objects[:2], [0., 0., .5],
                                      trusted_categories=True)
                    self.assertEqual(result['candidate_ids'], [1])
                    self.assertEqual(result['status'], 'missing')

    def test_plain_side_scope_still_resolves_in_explicit_observer_frame(self):
        phrase = 'on the right'
        query = ('Choose the largest cabinet ' + phrase +
                 ', standing in the middle of the room facing the door.')
        objects = [
            {'bbox_id': 1, 'target': 'cabinet', 'bbox_3d': [1., 2., 1., 2., 2., 2.]},
            {'bbox_id': 2, 'target': 'cabinet', 'bbox_3d': [3., 2., .5, 1., 1., 1.]},
            {'bbox_id': 3, 'target': 'door', 'bbox_3d': [0., 4., 1., 1., .2, 2.]},
        ]
        for scope in ('side', ''):
            with self.subTest(scope=scope):
                data = validate_structure(query, parsed_query(query, phrase, scope))
                self.assertEqual(data['Relations'][0]['scope'], 'side')
                result = diagnose(data, objects, objects[:2], [0., 0., .5],
                                  trusted_categories=True)
                self.assertEqual(result['candidate_ids'], [1])
                self.assertEqual(result['status'], 'resolved')


if __name__ == '__main__':
    unittest.main()
