"""Entry regressions based on the first real module acceptance traces."""
import unittest
from diagnostic import attributes_for, diagnose, refresh_condition_records, validate_structure
from query_rewrite import rewrite_query


def obj(oid, category, z, height=1.):
    return {'bbox_id': oid, 'target': category, 'bbox_3d': [0., 0., z, 1., 1., height]}


def raw(target, anchors, relations=(), attrs=()):
    return {'Target': [target], 'Anchor': list(anchors), 'Relations': list(relations),
            'Attributes': list(attrs), 'Observer': {}, 'Unhandled': []}


class SourceRoleBindingTests(unittest.TestCase):
    def test_chairs_under_table_are_reference_and_vertical_direction_is_correct(self):
        query = 'The table with white chairs under it'
        data = validate_structure(query, raw('table', ['chairs'], [{
            'target_category': 'table', 'anchor_category': 'chairs',
            'relation': 'underneath', 'scope': '', 'phrase': 'under it'}],
            [{'category': 'table', 'phrase': 'white'}]))
        self.assertEqual(data['Relations'][0]['relation'], 'above')
        self.assertEqual(attributes_for(data, 'table', role='target'), [])
        self.assertEqual(attributes_for(data, 'chair', role='reference'), ['white'])
        refresh_condition_records(query, data)
        record = next(item for item in data['condition_records']
                      if item['condition_type'] == 'appearance')
        self.assertEqual(record['reference_role'], {'category': 'chair'})
        objects = [obj(1, 'table', 2.), obj(2, 'chair', .5)]
        check = {'id': 2, 'category': 'chair', 'role': 'reference',
                 'category_verdict': 'yes', 'verdict': 'yes', 'visibility': 'visible',
                 'attribute_checks': [{'attribute': 'white', 'verdict': 'yes',
                    'visibility': 'visible', 'reason': 'white chair surface'}]}
        result = diagnose(data, objects, objects[:1], [0., 0., 1.], [check], True)
        self.assertEqual(result['status'], 'resolved')
        self.assertEqual(result['candidate_ids'], [1])
        self.assertTrue(result['geometry']['relation_pairs'][0]['passes'])

    def test_wooden_desk_is_not_wooden_trashcan(self):
        query = 'The trashcan in between printer and wooden desk'
        data = validate_structure(query, raw('trashcan', ['printer', 'wooden desk'], [{
            'target_category': 'trashcan', 'anchor_categories': ['printer', 'wooden desk'],
            'relation': 'between', 'phrase': 'in between printer and wooden desk'}],
            [{'category': 'trashcan', 'phrase': 'wooden'}]))
        self.assertEqual(attributes_for(data, 'trashcan', role='target'), [])
        self.assertEqual(attributes_for(data, 'desk', role='reference'), ['wooden'])
        self.assertFalse(any(c['type'] == 'attribute' for c in rewrite_query(query, data)['constraints']))

    def test_same_category_roles_keep_distinct_properties(self):
        query = 'brown table next to table with rainbow top'
        data = validate_structure(query, raw('table', ['table'], [{
            'target_category': 'table', 'anchor_category': 'table',
            'relation': 'other', 'phrase': 'next to table with rainbow top'}],
            [{'category': 'table', 'phrase': 'brown'},
             {'category': 'table', 'phrase': 'rainbow top'}]))
        self.assertEqual(attributes_for(data, 'table', role='target'), ['brown'])
        self.assertEqual(attributes_for(data, 'table', role='reference'), ['rainbow top'])
        self.assertIn('unsupported relation: next to table with rainbow top', data['diagnostic_missing'])

    def test_unbound_reference_appearance_is_requested_without_binding_or_filtering(self):
        query = 'brown table next to table with rainbow top'
        data = validate_structure(query, raw('table', ['table'], [{
            'target_category': 'table', 'anchor_category': 'table',
            'relation': 'other', 'phrase': 'next to table with rainbow top'}],
            [{'category': 'table', 'phrase': 'brown'},
             {'category': 'table', 'phrase': 'rainbow top'}]))
        objects = [obj(1, 'table', 0.), obj(2, 'table', 1.), obj(3, 'table', 2.)]

        result = diagnose(data, objects, objects, [0., 0., 0.], trusted_categories=True)

        reference_needs = [need for need in result['visual_needs'] if need['role'] == 'reference']
        self.assertEqual({need['id'] for need in reference_needs}, {1, 2, 3})
        self.assertTrue(all(need['attributes'] == ['rainbow top'] for need in reference_needs))
        self.assertTrue(all(need['candidate_ids'] == [1, 2, 3] for need in result['hypotheses']))
        self.assertEqual(result['candidate_ids'], [1, 2, 3])
        self.assertEqual(result['status'], 'missing')
        self.assertIn('unsupported relation: next to table with rainbow top', result['missing'])

    def test_parser_omission_recovers_quoted_reference_surface_attribute(self):
        query = 'brown table next to table with rainbow top'
        data = validate_structure(query, raw('table', ['table'], [], [
            {'category': 'table', 'role': 'target', 'phrase': 'brown'}]))
        self.assertEqual(attributes_for(data, 'table', role='target'), ['brown'])
        self.assertEqual(attributes_for(data, 'table', role='reference'), ['rainbow top'])

        objects = [obj(1, 'table', 0.), obj(2, 'table', 1.), obj(3, 'table', 2.)]
        result = diagnose(data, objects, objects, [0., 0., 0.], trusted_categories=True)
        reference_needs = [need for need in result['visual_needs'] if need['role'] == 'reference']
        self.assertEqual({need['id'] for need in reference_needs}, {1, 2, 3})
        self.assertTrue(all(need['attributes'] == ['rainbow top'] for need in reference_needs))
        self.assertEqual(result['candidate_ids'], [1, 2, 3])
        self.assertEqual(result['status'], 'missing')
        self.assertIn('proximity condition not represented', result['missing'])

    def test_surface_attribute_recovery_does_not_bind_an_unmodeled_object(self):
        query = 'brown table next to table with red box'
        data = validate_structure(query, raw('table', ['table'], [], [
            {'category': 'table', 'role': 'target', 'phrase': 'brown'}]))
        self.assertEqual(attributes_for(data, 'table', role='reference'), [])
        self.assertIn('appearance condition not represented: red', data['diagnostic_missing'])

    def test_reference_size_does_not_become_target_extreme(self):
        query = 'the cabinet above the tallest cabinet'
        data = validate_structure(query, raw('cabinet', ['cabinet'], [{
            'target_category': 'cabinet', 'anchor_category': 'cabinet',
            'relation': 'above', 'phrase': 'above the tallest cabinet'}],
            [{'category': 'cabinet', 'phrase': 'tallest'}]))
        objects = [obj(1, 'cabinet', 3., .5), obj(2, 'cabinet', 1., 1.5)]
        data['query_rewrite'] = rewrite_query(query, data)
        self.assertFalse(any(c['type'] == 'height_max' for c in data['query_rewrite']['constraints']))
        result = diagnose(data, objects, objects, [0., 0., 1.], trusted_categories=True)
        self.assertEqual(result['status'], 'resolved')
        self.assertEqual(result['candidate_ids'], [1])
        self.assertEqual(result['geometry']['size_constraints'][0]['role'], 'reference')

    def test_reference_visual_yes_cannot_certify_target_color(self):
        query = 'red shelf with a shelf above it'
        data = validate_structure(query, raw('shelf', ['shelf'], [{
            'target_category': 'shelf', 'anchor_category': 'shelf',
            'relation': 'above', 'phrase': 'a shelf above it'}],
            [{'category': 'shelf', 'phrase': 'red'}]))
        objects = [obj(1, 'shelf', .5), obj(2, 'shelf', 2.)]
        checks = [{'id': 1, 'category': 'shelf', 'role': 'reference',
                   'verdict': 'yes', 'visibility': 'visible'},
                  {'id': 2, 'category': 'shelf', 'role': 'reference',
                   'verdict': 'yes', 'visibility': 'visible'}]
        result = diagnose(data, objects, objects, [0., 0., 1.], checks, True)
        self.assertEqual(result['status'], 'missing')
        self.assertTrue(any(n['role'] == 'target' and n['attributes'] == ['red']
                            for n in result['visual_needs']))

    def test_unqualified_shared_color_stays_missing(self):
        query = 'shelf with a shelf above it; one is red'
        data = validate_structure(query, raw('shelf', ['shelf'], [],
                    [{'category': 'shelf', 'phrase': 'red'}]))
        self.assertEqual(data['Attributes'], [])
        self.assertIn('attribute target/reference role is ambiguous: red', data['diagnostic_missing'])

    def test_selection_after_observer_does_not_rebind_target_color(self):
        query = ('Standing in the middle of the room facing a chair, '
                 'select the red chair to the right of the table.')
        data = raw('chair', ['chair', 'table'], [{
            'target_category': 'chair', 'anchor_category': 'table',
            'relation': 'right', 'scope': 'relative_anchor',
            'phrase': 'to the right of the table'}],
            [{'category': 'chair', 'role': 'target', 'phrase': 'red'}])
        data['Observer'] = {
            'origin': {'kind': 'room_center', 'phrase': 'Standing in the middle of the room'},
            'facing': {'category': 'chair', 'phrase': 'facing a chair'}}
        data = validate_structure(query, data)
        self.assertEqual(attributes_for(data, 'chair', role='target'), ['red'])
        self.assertEqual(attributes_for(data, 'chair', role='reference'), [])
        objects = [
            {'bbox_id': 1, 'target': 'chair', 'bbox_3d': [-2., 3., .5, 1., 1., 1.]},
            {'bbox_id': 2, 'target': 'chair', 'bbox_3d': [2., 3., .5, 1., 1., 1.]},
            {'bbox_id': 3, 'target': 'table', 'bbox_3d': [0., 3., .5, 1., 1., 1.]}]
        checks = [{'id': oid, 'category': 'chair', 'role': 'reference',
                   'verdict': 'yes' if oid == 1 else 'no', 'visibility': 'visible'}
                  for oid in (1, 2)]
        diagnosis = diagnose(data, objects, objects[:2], [0., 0., 0.], checks, True)
        self.assertEqual(diagnosis['status'], 'missing')
        self.assertTrue(any(need['role'] == 'target' and need['attributes'] == ['red']
                            for need in diagnosis['visual_needs']))

    def test_no_command_after_observer_keeps_target_color_pending(self):
        query = ('Standing in the middle of the room facing a chair, '
                 'the red chair to the right of the table.')
        data = raw('chair', ['chair', 'table'], [{
            'target_category': 'chair', 'anchor_category': 'table',
            'relation': 'right', 'scope': 'relative_anchor',
            'phrase': 'to the right of the table'}],
            [{'category': 'chair', 'role': 'target', 'phrase': 'red'}])
        data['Observer'] = {
            'origin': {'kind': 'room_center', 'phrase': 'in the middle of the room'},
            'facing': {'category': 'chair', 'phrase': 'facing a chair'}}
        data = validate_structure(query, data)
        self.assertEqual(attributes_for(data, 'chair', role='target'), ['red'])
        self.assertEqual(attributes_for(data, 'chair', role='reference'), [])
        objects = [
            {'bbox_id': 1, 'target': 'chair', 'bbox_3d': [-2., 3., .5, .5, .5, 1.]},
            {'bbox_id': 2, 'target': 'chair', 'bbox_3d': [2., 3., .5, .5, .5, 1.]},
            {'bbox_id': 3, 'target': 'table', 'bbox_3d': [0., 3., .5, 1., 1., 1.]}]
        checks = [{'id': oid, 'category': 'chair', 'role': 'reference',
                   'category_verdict': 'yes', 'visibility': 'visible',
                   'attribute_checks': [{'attribute': 'red',
                       'verdict': 'yes' if oid == 1 else 'no',
                       'visibility': 'visible', 'reason': 'chair surface color'}]}
                  for oid in (1, 2)]
        diagnosis = diagnose(data, objects, objects[:2], [0., 0., 0.], checks, True)
        self.assertEqual(diagnosis['status'], 'missing')
        self.assertTrue(any(need['role'] == 'target' and need['attributes'] == ['red']
                            for need in diagnosis['visual_needs']))

    def test_observer_noun_attribute_can_be_corrected_to_reference(self):
        query = ('Standing in the middle of the room facing a red chair, '
                 'select the chair to the right of the table.')
        data = raw('chair', ['chair', 'table'], [{
            'target_category': 'chair', 'anchor_category': 'table',
            'relation': 'right', 'scope': 'relative_anchor',
            'phrase': 'to the right of the table'}],
            [{'category': 'chair', 'role': 'target', 'phrase': 'red'}])
        data['Observer'] = {
            'origin': {'kind': 'room_center', 'phrase': 'in the middle of the room'},
            'facing': {'category': 'chair', 'phrase': 'facing a red chair'}}
        data = validate_structure(query, data)
        self.assertEqual(attributes_for(data, 'chair', role='target'), [])
        self.assertEqual(attributes_for(data, 'chair', role='reference'), ['red'])
        self.assertEqual(data['diagnostic_missing'], [])

    def test_repeated_color_does_not_cover_unchecked_reference(self):
        query = 'The red cabinet above the red shelf.'
        data = validate_structure(query, raw('cabinet', ['shelf'], [{
            'target_category': 'cabinet', 'anchor_category': 'shelf',
            'relation': 'above', 'phrase': 'above the red shelf'}],
            [{'category': 'cabinet', 'role': 'target', 'phrase': 'red'}]))
        objects = [obj(1, 'cabinet', 2.5), obj(2, 'shelf', .5)]
        check = {'id': 1, 'category': 'cabinet', 'role': 'target',
                 'verdict': 'yes', 'visibility': 'visible',
                 'attribute_checks': [{'attribute': 'red', 'verdict': 'yes',
                                       'visibility': 'visible', 'reason': 'red cabinet'}]}
        diagnosis = diagnose(data, objects, objects[:1], [0., 0., 0.], [check], True)
        self.assertEqual(diagnosis['status'], 'missing')
        self.assertIn('appearance condition role not represented: red (reference shelf)',
                      diagnosis['missing'])
        missing_record = next(record for record in data['condition_records']
                              if record['reference_role'] == {'category': 'shelf'}
                              and record['evidence_status'] == 'missing')
        self.assertEqual(missing_record['source_quote'], 'red')

    def test_repeated_color_both_roles_can_be_covered(self):
        query = 'The red cabinet above the red shelf.'
        data = validate_structure(query, raw('cabinet', ['shelf'], [{
            'target_category': 'cabinet', 'anchor_category': 'shelf',
            'relation': 'above', 'phrase': 'above the red shelf'}],
            [{'category': 'cabinet', 'role': 'target', 'phrase': 'red'},
             {'category': 'shelf', 'role': 'reference', 'phrase': 'red'}]))
        objects = [obj(1, 'cabinet', 2.5), obj(2, 'shelf', .5)]
        checks = [{'id': oid, 'category': category, 'role': role,
                   'verdict': 'yes', 'visibility': 'visible',
                   'attribute_checks': [{'attribute': 'red', 'verdict': 'yes',
                                         'visibility': 'visible', 'reason': 'red surface'}]}
                  for oid, category, role in ((1, 'cabinet', 'target'), (2, 'shelf', 'reference'))]
        diagnosis = diagnose(data, objects, objects[:1], [0., 0., 0.], checks, True)
        self.assertEqual(diagnosis['status'], 'resolved')
        self.assertEqual(diagnosis['missing'], [])

    def test_compound_color_phrase_does_not_cover_unchecked_reference(self):
        query = 'The red large cabinet above the red shelf.'
        data = validate_structure(query, raw('cabinet', ['shelf'], [{
            'target_category': 'cabinet', 'anchor_category': 'shelf',
            'relation': 'above', 'phrase': 'above the red shelf'}],
            [{'category': 'cabinet', 'role': 'target', 'phrase': 'red large'}]))
        self.assertEqual(attributes_for(data, 'cabinet', role='target'), ['red large'])
        objects = [obj(1, 'cabinet', 2.5), obj(2, 'shelf', .5)]
        check = {'id': 1, 'category': 'cabinet', 'role': 'target',
                 'category_verdict': 'yes', 'visibility': 'visible',
                 'attribute_checks': [{'attribute': name, 'verdict': 'yes',
                     'visibility': 'visible', 'reason': 'cabinet ' + name}
                    for name in ('red', 'large')]}
        diagnosis = diagnose(data, objects, objects[:1], [0., 0., 0.], [check], True)
        self.assertEqual(diagnosis['status'], 'missing')
        self.assertIn('appearance condition role not represented: red (reference shelf)',
                      diagnosis['missing'])

    def test_compound_color_both_roles_can_be_covered(self):
        query = 'The red large cabinet above the red shelf.'
        data = validate_structure(query, raw('cabinet', ['shelf'], [{
            'target_category': 'cabinet', 'anchor_category': 'shelf',
            'relation': 'above', 'phrase': 'above the red shelf'}],
            [{'category': 'cabinet', 'role': 'target', 'phrase': 'red large'},
             {'category': 'shelf', 'role': 'reference', 'phrase': 'red'}]))
        self.assertEqual(data['diagnostic_missing'], [])
        self.assertEqual(attributes_for(data, 'cabinet', role='target'), ['red large'])

    def test_repeated_color_same_category_requires_both_roles(self):
        query = 'The red shelf above the red shelf.'
        data = validate_structure(query, raw('shelf', ['shelf'], [{
            'target_category': 'shelf', 'anchor_category': 'shelf',
            'relation': 'above', 'phrase': 'above the red shelf'}],
            [{'category': 'shelf', 'role': 'target', 'phrase': 'red'}]))
        self.assertIn('appearance condition role not represented: red (reference shelf)',
                      data['diagnostic_missing'])

    def test_parser_cannot_invent_comparison_scope_metadata(self):
        data = validate_structure('the smaller of the two cabinets', raw('cabinet', [], [], [{
            'category': 'cabinet', 'phrase': 'smaller',
            'geometry_attribute': 'smallest', 'comparison_scope': 'full_category',
            'scope_phrase': 'the smaller of the two cabinets'}]))
        self.assertNotIn('geometry_attribute', data['Attributes'][0])
        self.assertEqual(data['Attributes'][0]['phrase'], 'smaller')
        self.assertEqual(data['Attributes'][0]['comparison_scope'], 'explicit_pair')
        self.assertEqual(data['Attributes'][0]['required_member_count'], 2)
        self.assertEqual(data['diagnostic_missing'], [])

    def test_unquoted_pair_metadata_cannot_restrict_comparison_members(self):
        query = 'the smaller cabinet'
        data = validate_structure(query, raw('cabinet', [], [], [{
            'category': 'cabinet', 'phrase': 'smaller',
            'geometry_attribute': 'smallest', 'comparison_scope': 'explicit_pair',
            'required_member_count': 2, 'scope_phrase': 'the smaller of the two cabinets'}]))
        attribute, = data['Attributes']
        self.assertEqual(attribute['phrase'], 'smaller')
        self.assertNotIn('geometry_attribute', attribute)
        self.assertNotIn('comparison_scope', attribute)
        self.assertNotIn('required_member_count', attribute)
        condition, = rewrite_query(query, data)['constraints'][0]['size_conditions']
        self.assertEqual(condition['scope_kind'], 'full_category')
        self.assertEqual(condition['original_quote'], 'smaller')
        self.assertNotIn('required_member_count', condition)


class RelationScopeTests(unittest.TestCase):
    def test_live_parser_shape_recovers_correct_one_from_explicit_facing_group(self):
        query = ('When standing in the middle of the room facing the windows '
                 'the correct one is on the right.')
        data = raw('', ['room'], [{
            'target_category': 'window', 'anchor_category': 'room',
            'anchor_categories': ['room'], 'relation': 'right',
            'scope': 'relative_anchor', 'phrase': 'the correct one is on the right'}])
        data['Target'] = []
        data['Observer'] = {
            'origin': {'kind': 'room_center', 'category': '',
                       'phrase': 'standing in the middle of the room'},
            'facing': {'category': 'windows', 'group': False,
                       'phrase': 'facing the windows'},
        }
        data['Unhandled'] = [query]

        data = validate_structure(query, data)

        self.assertEqual(data['Target'], ['window'])
        self.assertEqual(data['Anchor'], [])
        self.assertEqual(data['Relations'][0]['scope'], 'side')
        self.assertEqual(data['target_coreferences'], [
            {'phrase': 'the correct one', 'category': 'window'}])
        self.assertEqual(data['Unhandled'], [])
        self.assertFalse(data['diagnostic_missing'])

    def test_complete_sentence_recovery_keeps_extra_condition_missing(self):
        query = ('When standing in the middle of the room facing the windows '
                 'the correct one is on the right, not the left one.')
        data = raw('', ['room'], [{
            'target_category': 'window', 'anchor_category': 'room',
            'anchor_categories': ['room'], 'relation': 'right',
            'scope': 'relative_anchor', 'phrase': 'the correct one is on the right'}])
        data['Target'] = []
        data['Observer'] = {
            'origin': {'kind': 'room_center', 'category': '',
                       'phrase': 'standing in the middle of the room'},
            'facing': {'category': 'windows', 'group': False,
                       'phrase': 'facing the windows'},
        }
        data['Unhandled'] = [query]

        data = validate_structure(query, data)

        self.assertEqual(data['Target'], [])
        self.assertEqual(data['Unhandled'], [query])
        self.assertTrue(any(item.startswith('unhandled condition:')
                            for item in data['diagnostic_missing']))

    def test_plain_side_quote_can_use_explicit_observer(self):
        query = 'When standing in the middle of the room facing the windows the correct one is on the right.'
        data = raw('window', [], [{
            'target_category': 'window', 'anchor_category': '', 'relation': 'right',
            'scope': 'relative_anchor', 'phrase': 'the correct one is on the right'}])
        data['Unhandled'] = ['the correct one']
        data['Observer'] = {'origin': {'kind': 'room_center', 'phrase': 'standing in the middle of the room'},
                            'facing': {'category': 'windows', 'group': False, 'phrase': 'facing the windows'}}
        data = validate_structure(query, data)
        self.assertEqual(data['Relations'][0]['scope'], 'side')
        self.assertTrue(data['Observer']['facing']['group'])
        self.assertEqual(data['target_coreferences'], [{'phrase': 'the correct one', 'category': 'window'}])
        self.assertEqual(data['Unhandled'], [])
        coreference = next(record for record in data['condition_records']
                           if record['condition_type'] == 'target_coreference')
        self.assertEqual(coreference['target_role'], {'category': 'window'})
        self.assertEqual(coreference['binding_status'], 'bound')
        self.assertFalse(data['diagnostic_missing'])

    def test_left_window_keeps_condition_with_missing_observer(self):
        data = validate_structure('The left window.', raw('window', [], [{
            'target_category': 'window', 'anchor_category': '', 'relation': 'left',
            'scope': '', 'phrase': 'The left window'}]))
        self.assertEqual(data['Relations'][0]['scope'], 'side')
        self.assertIn('observer origin or facing direction is unspecified', data['diagnostic_missing'])

    def test_target_coreference_stays_missing_without_observer_frame(self):
        query = 'The correct one is on the right.'
        data = raw('window', [], [{
            'target_category': 'window', 'anchor_category': '', 'relation': 'right',
            'scope': 'side', 'phrase': query}])
        data['Unhandled'] = ['the correct one']
        data = validate_structure(query, data)
        self.assertEqual(data['target_coreferences'], [])
        self.assertEqual(data['Unhandled'], ['the correct one'])
        self.assertTrue(data['diagnostic_missing'])

    def test_missing_relative_anchor_and_extreme_cannot_become_side(self):
        for quote in ('left of the cabinet', 'on the far left', 'on the leftmost side'):
            with self.subTest(quote=quote):
                data = validate_structure('window ' + quote, raw('window', [], [{
                    'target_category': 'window', 'anchor_category': '', 'relation': 'left',
                    'scope': '', 'phrase': quote}]))
                self.assertEqual(data['Relations'], [])
                self.assertTrue(data['diagnostic_missing'])


if __name__ == '__main__':
    unittest.main()
