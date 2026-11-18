"""Expanded properties retain their directly quoted target/reference attachment."""
import unittest

from eviground.query.noun_binding import bind_negative_attributes


class AttributeExpansionRoleTests(unittest.TestCase):
    def expand(self, query, phrase, anchor='thing'):
        parsed = dict(Target=['pillow'], Anchor=[anchor], Observer={}, Unhandled=[],
                      Attributes=[dict(category='pillow', role='target', phrase=phrase)])
        bind_negative_attributes(query, parsed, [])
        return {a['phrase']: (a['category'], a['role']) for a in parsed['Attributes']}

    def test_reference_colors_are_not_inherited_from_a_whole_query_attribute(self):
        query = 'The pillow to the right closest to the purple pink thing'
        expanded = self.expand(query, query)
        self.assertEqual(expanded['purple'], ('thing', 'reference'))
        self.assertEqual(expanded['pink'], ('thing', 'reference'))

    def test_distinct_target_and_reference_properties_keep_their_own_nouns(self):
        query = 'The blue pillow closest to the purple pink thing'
        expanded = self.expand(query, query)
        self.assertEqual(expanded['blue'], ('pillow', 'target'))
        self.assertEqual(expanded['purple'], ('thing', 'reference'))
        self.assertEqual(expanded['pink'], ('thing', 'reference'))

    def test_an_explicit_atomic_attribute_keeps_its_reported_role(self):
        query = 'The blue pillow closest to the blue chair'
        self.assertEqual(self.expand(query, 'blue', anchor='chair')['blue'],
                         ('pillow', 'target'))

    def test_resolved_expansion_removes_only_its_stale_missing_role_diagnostic(self):
        query = 'The pillow closest to the purple thing'
        parsed = dict(Target=['pillow'], Anchor=['thing'], Observer={}, Unhandled=[],
                      Attributes=[dict(category='pillow', role='target', phrase=query)],
                      diagnostic_missing=[
                          'appearance condition role not represented: purple (reference thing)',
                          'appearance condition not represented: purple',
                          'direction condition not represented'])
        bind_negative_attributes(query, parsed, [])
        self.assertEqual(parsed['diagnostic_missing'], ['direction condition not represented'])


if __name__ == '__main__':
    unittest.main()
