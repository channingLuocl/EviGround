"""Offline regressions for the audited role/scope and recall failures."""
import unittest

from seeground.grounding.diagnostic import (_source_attribute_bindings, attributes_for,
                        candidate_removal_gate, diagnose, size_condition_scope,
                        size_conditions, validate_structure)
from seeground.query.query_rewrite import rewrite_query


def raw(target, anchors=(), attributes=(), relations=(), observer=None):
    return {'Target': [target], 'Anchor': list(anchors),
            'Attributes': list(attributes), 'Relations': list(relations),
            'Observer': observer or {}, 'Unhandled': []}


def obj(oid, category, extent=1., z=None):
    return {'bbox_id': oid, 'target': category,
            'bbox_3d': [float(oid) * 2., 0., extent / 2. if z is None else z,
                        extent, extent, extent]}


def reliable_measurements(objects):
    return [{'id': item['bbox_id'], 'status': 'known', 'source': 'synthetic_oriented_box',
             'oriented_volume_m3': item['bbox_3d'][3] ** 3,
             'oriented_footprint_m2': item['bbox_3d'][3] ** 2,
             'metric_reliability': {metric: {'status': 'usable'} for metric in
                                    ('oriented_volume_m3', 'oriented_footprint_m2')}}
            for item in objects]


class V2BindingTests(unittest.TestCase):
    def test_joint_short_tall_cannot_discard_unreviewed_reference_hypotheses(self):
        query = 'The short bookshelf between the two tall bookshelves.'
        parsed = validate_structure(query, raw('bookshelf', ['bookshelf'], [
            {'category': 'bookshelf', 'role': 'target', 'phrase': 'short'},
            {'category': 'bookshelf', 'role': 'reference', 'phrase': 'tall'}], [
            {'target_category': 'bookshelf', 'anchor_category': 'bookshelf',
             'anchor_categories': ['bookshelf', 'bookshelf'], 'relation': 'between',
             'phrase': 'between the two tall bookshelves'}]))
        self.assertTrue(parsed.get('relative_height_binding'))
        objects = [obj(1, 'bookshelf'), obj(2, 'bookshelf'), obj(3, 'bookshelf')]
        measurements = [{'id': item['bbox_id'], 'status': 'known', 'height_m': 1.,
                         'source': 'synthetic complete height measurements',
                         'metric_reliability': {'height_m': {'status': 'usable'}}}
                        for item in objects]
        result = diagnose(parsed, objects, objects, [0, 0, 0], trusted_categories=False,
                          size_measurements=measurements)
        self.assertEqual(result['candidate_ids'], [1, 2, 3])
        self.assertEqual(result['status'], 'missing')
        self.assertTrue(result['hypotheses'])
        self.assertFalse(result['geometry'].get('rejected_hypotheses'))
        self.assertTrue(any(need['role'] == 'reference' for need in result['visual_needs']))

    def test_facing_largest_reference_never_executes_as_target_size(self):
        query = 'Facing the largest bookshelf, the bookshelf in the lower right corner.'
        parsed = validate_structure(query, raw('bookshelf', ['bookshelf'], [
            {'category': 'bookshelf', 'role': 'target', 'phrase': 'largest'}],
            observer={'facing': {'category': 'bookshelf', 'group': False,
                                 'phrase': 'Facing the largest bookshelf'}}))
        self.assertEqual(size_conditions(parsed, 'bookshelf', 'target'), [])
        self.assertEqual(size_conditions(parsed, 'bookshelf', 'reference')[0]['word'], 'largest')
        self.assertEqual(attributes_for(parsed, 'bookshelf', 'target'), [])
        parsed['query_rewrite'] = rewrite_query(query, parsed)
        self.assertFalse(any(item['type'] == 'size_max'
                             for item in parsed['query_rewrite']['constraints']))
        objects = [obj(1, 'bookshelf', .5), obj(2, 'bookshelf', 2.)]
        result = diagnose(parsed, objects, objects, [0., 0., 0.], trusted_categories=True)
        self.assertEqual(result['candidate_ids'], [1, 2])
        self.assertFalse(any(item.get('removal_allowed') for item in result['candidate_removal_audit']))

    def test_smaller_white_shelf_noun_phrase_is_not_copied_to_target(self):
        query = 'This floor-length shelf has a smaller white shelf on the floor next to it.'
        data = raw('shelf', ['shelf'], [
            {'category': 'shelf', 'role': 'target', 'phrase': 'smaller'},
            {'category': 'shelf', 'role': 'reference', 'phrase': 'smaller'},
            {'category': 'shelf', 'role': 'target', 'phrase': 'smaller white shelf'}])
        self.assertEqual(_source_attribute_bindings(query, data, 'smaller'), {('shelf', 'reference')})
        parsed = validate_structure(query, data)
        self.assertEqual(size_conditions(parsed, 'shelf', 'target'), [])
        self.assertEqual(len(size_conditions(parsed, 'shelf', 'reference')), 1)
        self.assertEqual(attributes_for(parsed, 'shelf', 'target'), [])
        objects = [obj(1, 'shelf', .5), obj(2, 'shelf', 2.)]
        result = diagnose(parsed, objects, objects, [0., 0., 0.], trusted_categories=True,
                          size_measurements=reliable_measurements(objects))
        self.assertEqual(result['candidate_ids'], [1, 2])
        self.assertEqual(result['status'], 'missing')

    def test_known_observer_frame_cannot_silently_ignore_reference_size(self):
        query = ('Standing in the middle of the room facing the largest bookshelf, '
                 'choose the chair to the right of the table.')
        parsed = validate_structure(query, raw('chair', ['bookshelf', 'table'], attributes=[
            {'category': 'bookshelf', 'role': 'reference', 'phrase': 'largest'}], relations=[
            {'target_category': 'chair', 'anchor_category': 'table', 'relation': 'right',
             'scope': 'relative_anchor', 'phrase': 'to the right of the table'}], observer={
            'origin': {'kind': 'room_center', 'phrase': 'Standing in the middle of the room'},
            'facing': {'category': 'bookshelf', 'group': False,
                       'phrase': 'facing the largest bookshelf'}}))
        objects = [obj(1, 'chair'), obj(2, 'table'),
                   obj(3, 'bookshelf', .5), obj(4, 'bookshelf', 2.)]
        objects[0]['bbox_3d'][:2] = [2., 2.]
        objects[1]['bbox_3d'][:2] = [0., 2.]
        objects[2]['bbox_3d'][:2] = [0., 3.]
        objects[3]['bbox_3d'][:2] = [0., 5.]
        result = diagnose(parsed, objects, objects[:1], [0., 0., 0.], trusted_categories=True)
        self.assertEqual(result['candidate_ids'], [1])
        self.assertEqual(result['status'], 'missing')
        self.assertTrue(any(item['role'] == 'reference' and item.get('observer_role') == 'facing'
                            for item in result['pending_size_conditions']))
        self.assertTrue(any('observer reference size comparison' in reason for reason in result['missing']))

    def test_single_unattached_size_word_role_conflict_keeps_candidates(self):
        query = 'The shelf beside another shelf; one is smaller.'
        parsed = validate_structure(query, raw('shelf', ['shelf'], [
            {'category': 'shelf', 'role': 'target', 'phrase': 'smaller'},
            {'category': 'shelf', 'role': 'reference', 'phrase': 'smaller'}]))
        self.assertEqual(size_conditions(parsed, 'shelf', 'target'), [])
        self.assertEqual(size_conditions(parsed, 'shelf', 'reference'), [])
        self.assertIn('attribute target/reference role is ambiguous: smaller', parsed['diagnostic_missing'])
        objects = [obj(1, 'shelf', .5), obj(2, 'shelf', 2.)]
        result = diagnose(parsed, objects, objects, [0., 0., 0.], trusted_categories=True)
        self.assertEqual(result['candidate_ids'], [1, 2])
        self.assertEqual(result['status'], 'missing')

    def test_relation_dependent_small_does_not_hard_filter_whole_book_category(self):
        query = 'Choose the small book on the table across from the black sofa.'
        parsed = validate_structure(query, raw('book', ['table', 'sofa'], [
            {'category': 'book', 'role': 'target', 'phrase': 'small'},
            {'category': 'sofa', 'role': 'reference', 'phrase': 'black'}], [
            {'target_category': 'book', 'anchor_category': 'table', 'relation': 'other',
             'phrase': 'on the table across from the black sofa'}]))
        books = [obj(1, 'book', .2), obj(2, 'book', .8), obj(3, 'book', 2.)]
        objects = books + [obj(4, 'table'), obj(5, 'sofa')]
        spec, = size_conditions(parsed, 'book')
        self.assertEqual(spec['scope_kind'], 'unbound_explicit_scope')
        pool, scope = size_condition_scope(parsed, objects, 'book', spec)
        self.assertEqual([item['bbox_id'] for item in pool], [1, 2, 3])
        self.assertEqual(scope['status'], 'unknown')
        self.assertFalse(parsed['_size_scope_bindings'])
        result = diagnose(parsed, objects, books, [0., 0., 0.], trusted_categories=True,
                          size_measurements=reliable_measurements(objects))
        self.assertEqual(result['candidate_ids'], [1, 2, 3])
        constraint, = result['geometry']['size_constraints']
        self.assertEqual(constraint['scope_status'], 'unknown')
        self.assertTrue(all(item['status'] == 'unknown' and not item['removal_allowed']
                            for item in constraint['candidate_evidence']))

    def test_empty_compound_category_pool_preserves_scene_fallback(self):
        query = 'The big computer table with a grey chair beside it'
        parsed = validate_structure(query, raw('computer table', ['chair'], [
            {'category': 'computer table', 'role': 'target', 'phrase': 'big'}]))
        objects = [obj(1, 'table', .5), obj(2, 'table', 2.), obj(3, 'chair')]
        result = diagnose(parsed, objects, objects, [0., 0., 0.], trusted_categories=True,
                          size_measurements=reliable_measurements(objects))
        self.assertEqual(result['candidate_ids'], [1, 2, 3])
        self.assertEqual(result['status'], 'missing')
        self.assertIn('target category recall/binding missing: computer table', result['missing'])
        self.assertFalse(any(item['removal_allowed'] for item in result['candidate_removal_audit']))
        self.assertTrue(all(item['status'] == 'unknown'
                            for item in result['geometry']['size_constraints'][0]['candidate_evidence']))

    def test_unverified_broad_fallback_is_not_a_bound_size_scope(self):
        parsed = validate_structure('the smallest book', raw('book', attributes=[
            {'category': 'book', 'role': 'target', 'phrase': 'smallest'}]))
        parsed['candidate_generation'] = {'source': 'broad_fallback', 'status': 'unknown'}
        objects = [obj(1, 'book', .5), obj(2, 'book', 2.)]
        result = diagnose(parsed, objects, objects, [0., 0., 0.], trusted_categories=True)
        self.assertEqual(result['candidate_ids'], [1, 2])
        self.assertEqual(result['geometry']['size_constraints'][0]['scope_status'], 'unknown')
        self.assertFalse(any(item['removal_allowed'] for item in result['candidate_removal_audit']))

    def test_unknown_relation_measurement_cannot_remove_a_candidate(self):
        query = 'the shelf above the table'
        parsed = validate_structure(query, raw('shelf', ['table'], relations=[
            {'target_category': 'shelf', 'anchor_category': 'table',
             'relation': 'above', 'phrase': 'above the table'}]))
        objects = [obj(1, 'shelf', z=.7), obj(2, 'table', z=.5)]
        objects[1]['bbox_3d'][0] = objects[0]['bbox_3d'][0]
        result = diagnose(parsed, objects, objects[:1], [0., 0., 0.], trusted_categories=True)
        self.assertEqual(result['candidate_ids'], [1])
        self.assertEqual(result['geometry']['relation_pairs'][0]['status'], 'unknown')
        self.assertFalse(any(item['removal_allowed'] for item in result['candidate_removal_audit']))

    def test_explicit_superlative_keeps_v1_category_scope_with_separate_relation(self):
        query = 'Choose the largest cabinet above the table.'
        parsed = validate_structure(query, raw('cabinet', ['table'], attributes=[
            {'category': 'cabinet', 'role': 'target', 'phrase': 'largest'}], relations=[
            {'target_category': 'cabinet', 'anchor_category': 'table',
             'relation': 'above', 'phrase': 'above the table'}]))
        cabinets = [obj(1, 'cabinet', .5, z=2.), obj(2, 'cabinet', 2., z=3.)]
        table = obj(3, 'table', z=.5)
        for item in [*cabinets, table]:
            item['bbox_3d'][0] = 0.
        result = diagnose(parsed, [*cabinets, table], cabinets, [0., 0., 0.], trusted_categories=True)
        self.assertEqual(result['candidate_ids'], [2])
        constraint, = result['geometry']['size_constraints']
        self.assertEqual(constraint['scope_status'], 'bound')
        self.assertEqual(constraint['comparison_pool_ids'], [1, 2])
        fail = next(item for item in constraint['candidate_evidence'] if item['candidate_id'] == 1)
        self.assertTrue(fail['removal_allowed'])
        self.assertEqual(fail['source_quote'], 'largest')
        self.assertEqual(fail['measurement_source'], 'annotated_box')
        self.assertEqual(fail['role_binding_status'], 'bound')
        self.assertEqual(fail['comparison_scope_status'], 'bound')
        self.assertTrue(fail['comparison_scope_frozen'])
        self.assertTrue(fail['comparison_scope_id'].startswith('scope-v2-'))
        self.assertEqual(fail['removal_rule_version'], 'relative-size-v2')
        self.assertIsNotNone(fail['value'])
        self.assertTrue(fail['explicit_fail_criterion'])

    def test_bound_reference_identity_exclusion_has_explicit_target_proof(self):
        query = 'the cabinet above the tallest cabinet'
        parsed = validate_structure(query, raw('cabinet', ['cabinet'], attributes=[
            {'category': 'cabinet', 'phrase': 'tallest'}], relations=[
            {'target_category': 'cabinet', 'anchor_category': 'cabinet',
             'relation': 'above', 'phrase': 'above the tallest cabinet'}]))
        objects = [obj(1, 'cabinet', .5, z=3.), obj(2, 'cabinet', 1.5, z=1.)]
        for item in objects:
            item['bbox_3d'][0] = 0.
        result = diagnose(parsed, objects, objects, [0., 0., 0.], trusted_categories=True)
        self.assertEqual(result['candidate_ids'], [1])
        exclusion = next(item for item in result['candidate_removal_audit']
                         if item['candidate_id'] == 2 and item['target_role'] == 'target'
                         and item['removal_allowed'])
        self.assertEqual(exclusion['source_quote'], 'above the tallest cabinet')
        self.assertEqual(exclusion['measurement_source'], 'bound instance identity')
        proof, = exclusion['hypothesis_fail_evidence']
        self.assertEqual(proof['explicit_fail_criterion'],
                         'target and relation reference must be distinct instances')
        self.assertIn(2, proof['comparison_scope_ids'])

    def test_gate_rejects_wrong_role_unknown_scope_or_missing_source(self):
        complete = {'candidate_id': 1, 'status': 'fail', 'source_quote': 'largest shelf',
                    'target_role': 'target', 'role_binding_status': 'bound',
                    'comparison_scope_status': 'bound', 'comparison_scope_ids': [1, 2],
                    'measurement_source': 'synthetic_box',
                    'explicit_fail_criterion': 'volume 1 is below maximum volume 8'}
        self.assertTrue(candidate_removal_gate(complete)['removal_allowed'])
        for update in ({'target_role': 'reference'}, {'comparison_scope_status': 'unknown'},
                       {'source_quote': ''}, {'comparison_scope_ids': [2]},
                       {'measurement_reliability': 'unreliable'},
                       {'measurement_source': 'unspecified'},
                       {'measurement_source': '', 'scope_source': 'complete category pool'},
                       {'explicit_fail_criterion': '', 'exclusion_reason': 'planner finish reason'}):
            with self.subTest(update=update):
                self.assertFalse(candidate_removal_gate({**complete, **update})['removal_allowed'])

    def test_conflicting_parser_reference_ids_do_not_prune_candidates(self):
        query = 'the shelf nearest the chair'
        parsed = validate_structure(query, raw('shelf', ['chair'], relations=[
            {'target_category': 'shelf', 'anchor_category': 'chair', 'relation': 'nearest',
             'phrase': 'nearest the chair', 'anchor_id': anchor_id} for anchor_id in (3, 4)]))
        objects = [obj(1, 'shelf'), obj(2, 'shelf'), obj(3, 'chair'), obj(4, 'chair')]
        result = diagnose(parsed, objects, objects[:2], [0., 0., 0.], trusted_categories=True)
        self.assertEqual(result['candidate_ids'], [1, 2])
        self.assertEqual(result['status'], 'missing')
        self.assertIn('distance reference has incompatible explicit instance bindings: chair', result['missing'])
        binding_audit = [item for item in result['candidate_removal_audit']
                         if item.get('reported_anchor_ids')]
        self.assertEqual({item['candidate_id'] for item in binding_audit}, {3, 4})
        self.assertTrue(all(item['status'] == 'unknown' and not item['removal_allowed']
                            for item in binding_audit))

    def test_irregular_plural_keeps_v1_local_short_tall_role_binding(self):
        query = 'The short bookshelf in between the two tall bookshelves'
        parsed = validate_structure(query, raw('bookshelf', ['bookshelf', 'bookshelf'], attributes=[
            {'category': 'bookshelf', 'role': 'target', 'phrase': 'short'},
            {'category': 'bookshelf', 'role': 'reference', 'phrase': 'tall'}], relations=[
            {'target_category': 'bookshelf', 'anchor_categories': ['bookshelf', 'bookshelf'],
             'relation': 'between', 'phrase': 'in between the two tall bookshelves'}]))
        self.assertEqual(size_conditions(parsed, 'bookshelf', 'target')[0]['word'], 'short')
        self.assertEqual(size_conditions(parsed, 'bookshelf', 'reference')[0]['word'], 'tall')
        self.assertEqual(parsed['relative_height_binding']['interpretation'], 'reference_comparison')


if __name__ == '__main__':
    unittest.main()
