import unittest

from r8_numeric import execute_choice


class NumericExecutionTests(unittest.TestCase):
    def decision(self, predicate='leftmost', quote='left', pool=(1, 2)):
        return {'predicted_id': 2, 'reason': 'Model proposal', 'evidence': [], 'unresolved': [],
            'numeric_choice': {'predicate': predicate, 'source_quote': quote, 'eligible_ids': list(pool),
                'frame_id': 'reference_9', 'remaining_conditions_checked': True, 'binding_reason': 'Both satisfy appearance.'}}

    def setUp(self):
        self.payload = {'observer_frames': [{'frame_id': 'reference_9', 'rows_left_to_right': [[1, -2, 3], [2, 2, 3], [3, 4, 3]]}],
                        'objects': [{'id': 1, 'height_m': .8}, {'id': 2, 'height_m': 1.2}]}

    def test_executes_direction_in_selected_frame_and_preserves_source_decision(self):
        original = self.decision()
        final, proof = execute_choice(original, 'the left chair', {}, self.payload, [1, 2])
        self.assertEqual(final['predicted_id'], 1)
        self.assertEqual(original['predicted_id'], 2)
        self.assertEqual(final['evidence'][0]['source_quote'], 'the left chair')
        self.assertTrue(proof['applied'])

    def test_refuses_unquoted_excluded_unchecked_and_unbound_claims(self):
        base = self.decision()
        for patch in ({'source_quote': 'invented'}, {'eligible_ids': [1, 9]},
                      {'remaining_conditions_checked': False}, {'frame_id': None},
                      {'eligible_ids': [1, 2, 3]}):
            original = {**base, 'numeric_choice': {**base['numeric_choice'], **patch}}
            final, proof = execute_choice(original, 'the left chair', {}, self.payload, [1, 2, 3])
            self.assertIs(final, original)
            self.assertFalse(proof['applied'])

    def test_frame_id_does_not_depend_on_array_position(self):
        other = {'frame_id': 'reference_10', 'rows_left_to_right': [[2, -9, 0], [1, 9, 0]]}
        for frames in ([other, *self.payload['observer_frames']], [*self.payload['observer_frames'], other]):
            final, proof = execute_choice(self.decision(), 'the left chair', {},
                                          {**self.payload, 'observer_frames': frames}, [1, 2])
            self.assertEqual(final['predicted_id'], 1)
            self.assertEqual(proof['frame_id'], 'reference_9')

    def test_explicit_extreme_can_order_three_but_ties_cannot_choose_by_id(self):
        original = self.decision(quote='far left', pool=(1, 2, 3))
        final, proof = execute_choice(original, 'the far left chair', {}, self.payload, [1, 2, 3])
        self.assertEqual(final['predicted_id'], 1)
        tied = {**self.payload, 'observer_frames': [{'frame_id': 'reference_9', 'rows_left_to_right': [[1, 2, 3], [2, 2, 4]]}]}
        self.assertFalse(execute_choice(self.decision(), 'the left chair', {}, tied, [1, 2])[1]['applied'])

    def test_height_requires_target_attachment_and_never_uses_highest(self):
        parsed = {'Target': ['chair'], 'condition_records': [{'condition_type': 'size',
            'source_quote': 'taller', 'target_role': {'category': 'chair'}, 'reference_role': None}]}
        original = self.decision('tallest', 'taller')
        final, proof = execute_choice(original, 'the taller chair', parsed, self.payload, [1, 2])
        self.assertEqual(final['predicted_id'], 2)
        self.assertTrue(proof['applied'])
        parsed['condition_records'][0]['target_role'] = None
        parsed['condition_records'][0]['reference_role'] = {'category': 'desk'}
        self.assertFalse(execute_choice(original, 'chair by the taller desk', parsed, self.payload, [1, 2])[1]['applied'])
        self.assertFalse(execute_choice(self.decision('tallest', 'highest'), 'the highest chair',
                                       parsed, self.payload, [1, 2])[1]['applied'])


if __name__ == '__main__':
    unittest.main()
