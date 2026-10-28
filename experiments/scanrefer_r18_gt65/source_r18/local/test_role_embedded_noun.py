"""Quoted noun phrases cannot transfer reference properties to a target."""
import unittest

from diagnostic import (_source_attribute_bindings, attributes_for, diagnose,
                        validate_structure)
from query_rewrite import rewrite_query


def parsed(attrs, unhandled=()):
    return {'Target': ['trash can'], 'Anchor': ['trash can', 'trash can'],
            'Relations': [{'target_category': 'trash can',
                           'anchor_categories': ['trash can', 'trash can'],
                           'relation': 'between',
                           'phrase': 'in between another trash can and a smaller trash can with a blue bag'}],
            'Attributes': attrs, 'Observer': {}, 'Unhandled': list(unhandled)}


QUERY = ('The trash can that is in between another trash can and '
         'a smaller trash can with a blue bag.')


class EmbeddedNounRoleTests(unittest.TestCase):
    def test_live_raw_postnoun_quote_binds_bag_to_reference(self):
        data = validate_structure(QUERY, parsed([
            {'category': 'trash can', 'phrase': 'with a blue bag'},
            {'category': 'trash can', 'phrase': 'smaller'}]))
        self.assertEqual(attributes_for(data, 'trash can', role='target'), [])
        self.assertEqual(attributes_for(data, 'trash can', role='reference'),
                         ['with a blue bag', 'smaller'])

    def test_live_repair_naked_quote_corrects_wrong_target_role(self):
        data = validate_structure(QUERY, parsed([
            {'category': 'trash can', 'phrase': 'smaller'},
            {'category': 'trash can', 'role': 'target', 'phrase': 'blue bag'}]))
        self.assertEqual(attributes_for(data, 'trash can', role='target'), [])
        bag = next(attribute for attribute in data['Attributes'] if attribute['phrase'] == 'blue bag')
        self.assertEqual(bag['role'], 'reference')
        self.assertEqual(bag['reported_role'], 'target')

    def test_direct_connectors_allow_optional_quoted_determiner(self):
        data = {'Target': ['chair'], 'Anchor': [], 'Observer': {}}
        for connector in ('with', 'is', 'has', 'that is', 'that has'):
            for article in ('', 'a ', 'an ', 'the '):
                query = 'the chair ' + connector + ' ' + article + 'blue cushion'
                for phrase in ('blue cushion', connector + ' ' + article + 'blue cushion'):
                    with self.subTest(query=query, phrase=phrase):
                        self.assertEqual(_source_attribute_bindings(query, data, phrase),
                                         {('chair', 'target')})

    def test_postnoun_quote_does_not_cross_a_modeled_head(self):
        data = {'Target': ['table'], 'Anchor': ['chair'], 'Observer': {}}
        query = 'the table with a blue chair'
        self.assertEqual(_source_attribute_bindings(query, data, 'blue chair'),
                         {('chair', 'reference')})
        self.assertEqual(_source_attribute_bindings(query, data, 'with a blue chair'), set())
        self.assertEqual(_source_attribute_bindings(
            'the table near the chair with a blue cushion', data, 'blue cushion'),
            {('chair', 'reference')})
        self.assertEqual(_source_attribute_bindings(
            'the table with a blue cushion near a box', data,
            'with a blue cushion near a box'), set())

    def test_repeated_postnoun_property_keeps_both_language_roles(self):
        data = {'Target': ['chair'], 'Anchor': ['chair'], 'Observer': {}}
        query = 'the chair with a blue cushion above the chair with a blue cushion'
        self.assertEqual(_source_attribute_bindings(query, data, 'blue cushion'),
                         {('chair', 'target'), ('chair', 'reference')})

    def test_postnoun_attachment_respects_observer_and_selection(self):
        data = {'Target': ['chair'], 'Anchor': ['chair'], 'Observer': {
            'facing': {'category': 'chair', 'phrase': 'facing a chair with a blue cushion'}}}
        self.assertEqual(_source_attribute_bindings(
            'Facing a chair with a blue cushion, select the chair.', data, 'blue cushion'),
            {('chair', 'reference')})
        self.assertEqual(_source_attribute_bindings(
            'Facing a chair, select the chair with a blue cushion.', data, 'blue cushion'),
            {('chair', 'target')})

    def test_parser_repair_reference_phrases_keep_their_source_role(self):
        data = validate_structure(QUERY, parsed([
            {'category': 'trash can', 'role': 'target', 'phrase': 'smaller trash can'},
            {'category': 'trash can', 'role': 'reference', 'phrase': 'another trash can'},
            {'category': 'trash can', 'role': 'target',
             'phrase': 'a smaller trash can with a blue bag'}],
            ['a smaller trash can with a blue bag']))

        self.assertEqual(attributes_for(data, 'trash can', role='target'), [])
        self.assertEqual(attributes_for(data, 'trash can', role='reference'),
                         ['smaller trash can', 'another trash can',
                          'a smaller trash can with a blue bag'])
        self.assertFalse(any(constraint['type'] == 'size_min'
                             for constraint in rewrite_query(QUERY, data)['constraints']))
        self.assertIn('unhandled condition: a smaller trash can with a blue bag',
                      data['diagnostic_missing'])

    def test_reference_property_no_cannot_delete_midpoint_target(self):
        data = validate_structure(QUERY, parsed([
            {'category': 'trash can', 'role': 'target',
             'phrase': 'a smaller trash can with a blue bag'}],
            ['a smaller trash can with a blue bag']))
        objects = [{'bbox_id': oid, 'target': 'trash can',
                    'bbox_3d': [x, 0., 1., .5, .5, 1.]}
                   for oid, x in ((31, -2.), (44, 0.), (57, 2.))]
        check = {'id': 44, 'category': 'trash can', 'role': data['Attributes'][0]['role'],
                 'category_verdict': 'yes', 'verdict': 'no', 'visibility': 'visible',
                 'attribute_checks': [{
                     'attribute': 'a smaller trash can with a blue bag',
                     'verdict': 'no', 'visibility': 'visible',
                     'reason': 'The middle bin lacks a blue bag.'}]}

        result = diagnose(data, objects, objects, [0., 0., 0.], [check],
                          trusted_categories=True, apply_size_constraints=False)

        self.assertIn(44, result['candidate_ids'])
        self.assertEqual(result['status'], 'missing')
        self.assertFalse(any(need['role'] == 'target' and need['attributes']
                             for need in result['visual_needs']))

    def test_head_noun_target_phrase_remains_target(self):
        query = 'The smaller trash can in between another trash can and a trash can.'
        data = {'Target': ['trash can'], 'Anchor': ['trash can', 'trash can'],
                'Relations': [], 'Observer': {}, 'Unhandled': [],
                'Attributes': [{'category': 'trash can', 'role': 'reference',
                                'phrase': 'smaller trash can'}]}
        data = validate_structure(query, data)
        self.assertEqual(attributes_for(data, 'trash can', role='target'),
                         ['smaller trash can'])
        self.assertEqual(attributes_for(data, 'trash can', role='reference'), [])

    def test_repeated_phrase_does_not_select_one_language_role(self):
        query = 'The red trash can above the red trash can.'
        data = {'Target': ['trash can'], 'Anchor': ['trash can'],
                'Relations': [], 'Observer': {}}
        self.assertEqual(_source_attribute_bindings(query, data, 'red trash can'),
                         {('trash can', 'target'), ('trash can', 'reference')})

    def test_relation_quote_is_not_a_single_head_noun_attribute(self):
        data = {'Target': ['chair'], 'Anchor': ['table'],
                'Relations': [], 'Observer': {}}
        self.assertEqual(_source_attribute_bindings(
            'the table near a red chair', data, 'near a red chair'), set())
        self.assertEqual(_source_attribute_bindings(
            'the red chair above a table', data, 'red chair above a table'), set())
        self.assertEqual(_source_attribute_bindings(
            'the red chair near a blue box', data, 'red chair near a blue box'), set())


if __name__ == '__main__':
    unittest.main()
