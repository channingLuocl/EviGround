import unittest

from query_rewrite import full_category_size_comparison, rewrite_query
from size_semantics import evaluate_size_condition, relative_height_between, size_specs


def row(oid, volume=1., footprint=1., height=1., **extra):
    return {'id': oid, 'status': 'known', 'measurement_reliability': 'usable',
            'volume_m3': volume, 'footprint_m2': footprint, 'height_m': height,
            'oriented_volume_m3': volume, 'oriented_footprint_m2': footprint,
            'metric_reliability': {'oriented_volume_m3': {'status': 'usable'},
                                   'oriented_footprint_m2': {'status': 'usable'},
                                   'height_m': {'status': 'usable'}}, **extra}


def reliable(row_value, *metrics):
    return {**row_value, 'metric_reliability': {metric: {'status': 'usable'} for metric in metrics}}


def parsed(category, word):
    return {'Target': [category], 'Anchor': [], 'Relations': [], 'Attributes': [
        {'category': category, 'role': 'target', 'phrase': word}], 'Observer': {}, 'Unhandled': []}


def evaluate(word, rows, *, category='box', query='', pool=None, candidates=None, bound=True):
    spec, = size_specs(word, category, query=query)
    return evaluate_size_condition(spec, rows, [item['id'] for item in rows] if pool is None else pool,
                                   scope_bound=bound, candidate_ids=candidates)


def statuses(result):
    return {item['candidate_id']: item['status'] for item in result['candidate_evidence']}


class RelativeSizeSemanticsTests(unittest.TestCase):
    def test_explicit_than_reference_cannot_fall_back_to_full_category_pool(self):
        spec = size_specs('smaller', 'box', query='the box smaller than the cabinet')[0]
        self.assertEqual(spec['scope_kind'], 'unbound_explicit_scope')
        self.assertEqual(spec['scope_phrase'], 'than the cabinet')
        result = evaluate_size_condition(spec, [], [1, 2], scope_bound=True)
        self.assertEqual(statuses(result), {1: 'unknown', 2: 'unknown'})
        self.assertEqual(result['selected_ids'], [])

    def test_original_adjective_is_retained_with_declared_context_interpretation(self):
        result = evaluate('small', [row(91, .2, .3), row(7, 1., 1.)])
        self.assertEqual(result['word'], 'small')
        self.assertEqual(result['original_quote'], 'small')
        self.assertEqual(result['form'], 'adjective')
        self.assertEqual(result['interpretation'], 'contextual_set_comparison')
        self.assertEqual(statuses(result), {91: 'pass', 7: 'fail'})
        self.assertEqual(result['relative_margin'], .20)

    def test_adjective_requires_stronger_difference_than_comparative(self):
        rows = [row(1, .85, .85), row(2)]
        self.assertEqual(statuses(evaluate('smaller', rows)), {1: 'pass', 2: 'fail'})
        self.assertEqual(statuses(evaluate('small', rows)), {1: 'unknown', 2: 'unknown'})

    def test_three_member_bare_comparative_requires_clear_unique_extreme(self):
        result = evaluate('bigger', [row(10, 1., 1.), row(20, 1.1, 1.1), row(30, 2., 2.)])
        self.assertEqual(result['interpretation'], 'set_comparison')
        self.assertEqual(result['selected_ids'], [30])
        near = evaluate('bigger', [row(10, 1., 1.), row(20, 1.95, 1.95), row(30, 2., 2.)])
        self.assertEqual(set(statuses(near).values()), {'unknown'})

    def test_of_the_two_requires_exact_complete_members(self):
        query = 'the smaller of the two boxes'
        rows = [row(1, .2, .2), row(2)]
        self.assertEqual(evaluate('smaller', rows, query=query)['selected_ids'], [1])
        for pool in ([1], [1, 2, 3]):
            result = evaluate('smaller', rows, query=query, pool=pool)
            self.assertEqual(set(statuses(result).values()), {'unknown'})
            self.assertIn('member count', result['reason'])

    def test_unbound_or_unknown_comparison_member_keeps_every_candidate(self):
        rows = [row(1, .1, .1), row(2)]
        for result in (evaluate('small', rows, bound=False),
                       evaluate('small', rows[:1], pool=[1, 2])):
            self.assertEqual(set(statuses(result).values()), {'unknown'})
            self.assertEqual(result['selected_ids'], [])

    def test_near_tied_superlative_and_absolute_height_tolerance_keep_candidates(self):
        result = evaluate('smallest', [row(1, .98, .98), row(2)])
        self.assertEqual(statuses(result), {1: 'unknown', 2: 'unknown'})
        self.assertEqual(result['tie_ids'], [1, 2])
        result = evaluate('shorter', [row(1, height=.09), row(2, height=.10)], category='bookshelf')
        self.assertEqual(statuses(result), {1: 'unknown', 2: 'unknown'})

    def test_size_pool_survives_color_filter_and_never_becomes_singleton_proof(self):
        rows = [row(1, 3., 3.), row(2, 1., 1.)]
        result = evaluate('large', rows, candidates=[2], pool=[1, 2],
                          category='trash can', query='large green trash can')
        self.assertEqual(result['scope_kind'], 'full_category')
        self.assertEqual(result['comparison_pool_ids'], [1, 2])
        self.assertEqual(statuses(result), {2: 'fail'})
        self.assertEqual(statuses(evaluate('large', rows[1:], pool=[2])), {2: 'unknown'})

    def test_only_explicit_of_colored_subset_declares_subset(self):
        spec, = size_specs('largest', 'trash can', query='largest of the green trash cans')
        self.assertEqual(spec['scope_kind'], 'attribute_subset')
        self.assertEqual(spec['scope_modifiers'], 'green')
        spec, = size_specs('large', 'trash can', query='large green trash can')
        self.assertEqual(spec['scope_kind'], 'full_category')

    def test_unrecognized_or_relational_explicit_subset_does_not_fall_back_to_full_pool(self):
        for query, category in [('largest of the green bins', 'trash can'),
                                ('largest of the cabinets near the door', 'cabinet'),
                                ('smaller of the two tables', 'cabinet')]:
            word = 'smaller' if query.startswith('smaller') else 'largest'
            spec, = size_specs(word, category, query=query)
            self.assertEqual(spec['scope_kind'], 'unbound_explicit_scope')
            result = evaluate_size_condition(spec, [row(1, 2., 2.), row(2)], [1, 2], scope_bound=True)
            self.assertEqual(statuses(result), {1: 'unknown', 2: 'unknown'})
            self.assertEqual(result['scope_status'], 'unknown')

    def test_longer_and_shorter_use_independent_correct_dimensions(self):
        rows = [reliable(row(1, height=1., horizontal_oriented_long_extent_m=3.),
                         'height_m', 'horizontal_oriented_long_extent_m'),
                reliable(row(2, height=2., horizontal_oriented_long_extent_m=1.),
                         'height_m', 'horizontal_oriented_long_extent_m')]
        self.assertEqual(evaluate('longer', rows, category='bookshelf')['selected_ids'], [1])
        self.assertEqual(evaluate('shorter', rows, category='bookshelf')['selected_ids'], [1])

    def test_oriented_length_needs_explicit_reliability_not_aabb_x(self):
        rows = [row(1, horizontal_oriented_long_extent_m=3., extent_x_m=3.),
                row(2, horizontal_oriented_long_extent_m=1., extent_x_m=1.)]
        self.assertEqual(set(statuses(evaluate('longer', rows, category='bookshelf')).values()), {'unknown'})

    def test_new_overall_comparative_does_not_rank_unoriented_annotated_boxes(self):
        rows = [row(1, .2, .2), row(2)]
        for item in rows:
            for field in ('oriented_volume_m3', 'oriented_footprint_m2', 'metric_reliability'):
                item.pop(field)
        self.assertEqual(set(statuses(evaluate('smaller', rows)).values()), {'unknown'})
        self.assertEqual(evaluate('smallest', rows)['selected_ids'], [1])

    def test_doors_windows_use_reliable_face_area_despite_opposite_volume_order(self):
        rows = [reliable(row(1, .9, .9, vertical_face_area_m2=1.), 'vertical_face_area_m2'),
                reliable(row(2, .1, .1, vertical_face_area_m2=3.), 'vertical_face_area_m2')]
        for category in ('door', 'window'):
            result = evaluate('smaller', rows, category=category)
            self.assertEqual(result['metric'], 'vertical_face_area_m2')
            self.assertEqual(result['selected_ids'], [1])
        rows[0]['metric_reliability']['vertical_face_area_m2']['status'] = 'unknown'
        self.assertEqual(set(statuses(evaluate('smaller', rows, category='door')).values()), {'unknown'})

    def test_tall_is_descriptive_even_when_one_shelf_is_taller(self):
        for heights in ((2., 2.), (2., 3.)):
            result = evaluate('tall', [row(1, height=heights[0]), row(2, height=heights[1])], category='shelf')
            self.assertEqual(result['interpretation'], 'descriptive_attribute')
            self.assertEqual(statuses(result), {1: 'unknown', 2: 'unknown'})
            self.assertEqual(result['selected_ids'], [])

    def test_local_short_relative_to_two_refs_does_not_require_scene_minimum(self):
        spec, = size_specs('short', 'bookshelf')
        spec.update(interpretation='reference_comparison', reference_ids=[10, 11],
                    required_reference_count=2)
        rows = [row(1, height=1.), row(10, height=2.), row(11, height=2.2), row(90, height=.3)]
        result = evaluate_size_condition(spec, rows, [1, 10, 11], scope_bound=True, candidate_ids=[1])
        self.assertEqual(statuses(result), {1: 'pass'})
        self.assertIsNone(result['reason'])
        self.assertEqual(result['relative_margin'], .10)
        spec['reference_ids'] = [10, 10]
        self.assertEqual(statuses(evaluate_size_condition(
            spec, rows, [1, 10, 11], scope_bound=True, candidate_ids=[1])), {1: 'unknown'})

    def test_local_reference_tie_metadata_distinguishes_tie_from_insufficient_gap(self):
        spec, = size_specs('short', 'bookshelf')
        spec.update(interpretation='reference_comparison', reference_ids=[10, 11],
                    required_reference_count=2)
        for height, tied_ids in ((1.01, [1, 10]), (1.07, [])):
            with self.subTest(reference_height=height):
                rows = [row(1, height=1.), row(10, height=height), row(11, height=2.)]
                result = evaluate_size_condition(spec, rows, [1, 10, 11],
                                                 scope_bound=True, candidate_ids=[1])
                self.assertEqual(statuses(result), {1: 'unknown'})
                self.assertEqual(result['tie_ids'], tied_ids)
                self.assertEqual(result['relative_margin'], .10)

    def test_joint_tall_reference_record_is_complete_and_shared_with_active(self):
        import copy
        import itertools
        import numpy as np
        from active_diagnostic import apply_size_observation, compute_observation
        from diagnostic import diagnose, validate_structure
        from object_geometry import point_cloud_measurements

        query = 'The short bookshelf in between the two tall bookshelves'
        objects = [{'bbox_id': oid, 'target': 'bookshelf',
                    'bbox_3d': [x, 0., height / 2., .4, .4, height]}
                   for oid, x, height in ((1, 0., 1.), (10, -2., 2.),
                                          (11, 2., 2.2), (90, 20., .3))]
        parsed_query = validate_structure(query, {
            'Target': ['bookshelf'], 'Anchor': ['bookshelf', 'bookshelf'],
            'Attributes': [{'category': 'bookshelf', 'role': 'target', 'phrase': 'short'},
                           {'category': 'bookshelf', 'role': 'reference', 'phrase': 'tall'}],
            'Relations': [{'target_category': 'bookshelf',
                           'anchor_categories': ['bookshelf', 'bookshelf'], 'relation': 'between',
                           'scope': 'relative_anchors', 'phrase': 'in between the two tall bookshelves'}],
            'Observer': {}, 'Unhandled': []})
        xyz, ids = [], []
        for obj in objects:
            corners = np.asarray(list(itertools.product((-.5, .5), repeat=3)))
            xyz.extend(corners * obj['bbox_3d'][3:] + obj['bbox_3d'][:3])
            ids.extend([obj['bbox_id']] * len(corners))
        xyz, ids = np.asarray(xyz), np.asarray(ids)
        measurements = point_cloud_measurements(objects, xyz, ids)
        normal = diagnose(parsed_query, objects, objects, [0., 0., 1.],
                          trusted_categories=True, query=query, size_measurements=measurements)
        deferred = diagnose(parsed_query, objects, objects, [0., 0., 1.],
                            trusted_categories=True, query=query, apply_size_constraints=False)
        observation = compute_observation('size', query, parsed_query, deferred, objects, objects, xyz, ids)
        applied = apply_size_observation(copy.deepcopy(deferred), observation)
        self.assertEqual(observation['size_constraints'], normal['geometry']['size_constraints'])
        self.assertEqual(applied['geometry']['size_constraints'], normal['geometry']['size_constraints'])
        record = next(c for c in observation['size_constraints'] if c['role'] == 'reference'
                      and c['target_id'] == 1 and set(c['anchor_ids']) == {10, 11})
        self.assertEqual(record['original_quote'], 'tall')
        self.assertEqual(record['word'], 'tall')
        self.assertEqual(record['form'], 'adjective')
        self.assertEqual(record['interpretation'], 'reference_comparison')
        self.assertEqual(record['scope_kind'], 'bound_relation_hypothesis')
        self.assertEqual(record['scope_status'], 'bound')
        self.assertTrue(record['comparison_pool_frozen'])
        self.assertEqual(record['comparison_pool_ids'], [1, 10, 11])
        self.assertEqual(record['reference_ids'], [1])
        self.assertEqual(record['required_reference_count'], 1)
        self.assertEqual(record['metric'], 'height_m')
        self.assertIn('no absolute tallness claim', record['metric_reason'])
        self.assertTrue(record['source'])
        self.assertEqual(record['selected_ids'], [10, 11])
        self.assertEqual(statuses(record), {10: 'pass', 11: 'pass'})
        for proof in record['candidate_evidence']:
            self.assertEqual(proof['measurement_reliability']['status'], 'usable')
            self.assertEqual(proof['measurement_reliability']['units'], 'm')
            self.assertTrue(proof['measurement_reliability']['algorithm'])

    def test_existing_book_volume_footprint_conflict_protection_remains(self):
        result = evaluate('smallest', [row(1, .5, .1), row(2, .3, .4)], category='book')
        self.assertEqual(statuses(result), {1: 'unknown', 2: 'unknown'})
        self.assertIn('ambiguous', result['reason'])

    def test_oriented_comparison_ignores_rotation_dependent_aabb_conflict(self):
        first = [row(1, 3., 3.), row(2, 4.4, 4.)]
        rotated = [row(1, 8., 8., oriented_volume_m3=3., oriented_footprint_m2=3.),
                   row(2, 8.8, 8., oriented_volume_m3=4.4, oriented_footprint_m2=4.)]
        self.assertEqual(statuses(evaluate('bigger', first, category='trash can')),
                         statuses(evaluate('bigger', rotated, category='trash can')))
        self.assertEqual(evaluate('bigger', rotated, category='trash can')['selected_ids'], [2])
        self.assertEqual(set(statuses(evaluate('biggest', rotated, category='trash can')).values()), {'unknown'})

    def test_oriented_size_does_not_fall_back_when_new_proxy_unreliable(self):
        rows = [reliable(row(1, .1, .1, oriented_volume_m3=.1, oriented_footprint_m2=.1),
                         'oriented_volume_m3', 'oriented_footprint_m2'),
                reliable(row(2, 1., 1., oriented_volume_m3=1., oriented_footprint_m2=1.),
                         'oriented_volume_m3', 'oriented_footprint_m2')]
        result = evaluate('small', rows)
        self.assertEqual(result['metric'], 'oriented_volume_m3')
        self.assertEqual(result['selected_ids'], [1])
        rows[0]['metric_reliability']['oriented_volume_m3']['status'] = 'unknown'
        self.assertEqual(set(statuses(evaluate('small', rows)).values()), {'unknown'})

    def test_id_and_input_order_do_not_select_winner(self):
        first = evaluate('smaller', [row(100, .2, .2), row(1, 1., 1.)])
        second = evaluate('smaller', [row(200, 1., 1.), row(99, .2, .2)])
        self.assertEqual(first['selected_ids'], [100])
        self.assertEqual(second['selected_ids'], [99])
        self.assertEqual(first['margin_to_next'], second['margin_to_next'])

    def test_height_same_top_shifted_bottom_is_unknown_without_completeness_proof(self):
        rows = [row(1, height=1., bottom_z_m=1., top_z_m=2.),
                row(2, height=2., bottom_z_m=0., top_z_m=2.)]
        self.assertEqual(statuses(evaluate('shorter', rows, category='bookshelf')),
                         {1: 'unknown', 2: 'unknown'})
        rows[0].update(bottom_z_m=0., top_z_m=1.)
        self.assertEqual(statuses(evaluate('shorter', rows, category='bookshelf')),
                         {1: 'pass', 2: 'fail'})

    def test_relative_opposition_helper_keeps_quoted_roles(self):
        query = 'The short bookshelf in between the two tall bookshelves'
        result = relative_height_between(query, 'bookshelf')
        self.assertEqual(result['target_phrase'], 'short')
        self.assertEqual(result['reference_phrase'], 'tall')
        self.assertEqual(result['relation_phrase'], 'in between the two tall bookshelves')
        self.assertEqual(result['required_reference_count'], 2)
        self.assertIsNone(relative_height_between('the short bookshelf behind two tall bookshelves', 'bookshelf'))

    def test_whole_request_recovery_is_general_and_preserves_comparative(self):
        for query, category, word in [('The taller of the desks.', 'desk', 'taller'),
                                      ('click the small door', 'door', 'small'),
                                      ('The smaller of the two cabinets.', 'cabinet', 'smaller')]:
            result = full_category_size_comparison(query, parsed(category, word))
            self.assertEqual(result['phrase'], word)
            self.assertNotIn('geometry_attribute', result)
        pair = full_category_size_comparison('The smaller of the two cabinets.', parsed('cabinet', 'smaller'))
        self.assertEqual(pair['comparison_scope'], 'explicit_pair')
        self.assertEqual(pair['required_member_count'], 2)
        self.assertIsNone(full_category_size_comparison('the large green trash can', parsed('trash can', 'large')))
        source = parsed('desk', 'taller')
        source['Anchor'] = ['desks']
        self.assertIsNotNone(full_category_size_comparison('The taller of the desks.', source))

    def test_whole_recovery_preserves_wrong_binding_for_existing_repair(self):
        query = 'The smaller of the cabinets.'
        for attribute in ({'category': 'smaller', 'phrase': 'smaller'},
                          {'category': 'cabinet', 'role': 'reference', 'phrase': 'smaller'}):
            source = {**parsed('cabinet', 'smaller'), 'Attributes': [attribute]}
            self.assertIsNone(full_category_size_comparison(query, source))
        for word in ('skinny', 'narrow', 'thin'):
            self.assertIsNone(full_category_size_comparison(word + ' cabinet', parsed('cabinet', word)))

    def test_legacy_superlative_aabb_compatibility_does_not_extend_to_new_comparative(self):
        rows = [row(1, .2, .2, oriented_volume_m3=None, metric_reliability={}),
                row(2, 1., 1., oriented_volume_m3=None, metric_reliability={})]
        self.assertEqual(evaluate('smallest', rows)['selected_ids'], [1])
        self.assertEqual(set(statuses(evaluate('smaller', rows)).values()), {'unknown'})
        for item in rows:
            item.pop('measurement_reliability')
        self.assertEqual(evaluate('smallest', rows)['selected_ids'], [1])

    def test_rewrite_records_semantics_without_forging_superlative(self):
        result = rewrite_query('small box', parsed('box', 'small'))
        constraint, = result['constraints']
        self.assertEqual(constraint['phrase'], 'small')
        self.assertEqual(constraint['size_conditions'][0]['word'], 'small')
        self.assertEqual(constraint['size_conditions'][0]['form'], 'adjective')


if __name__ == '__main__':
    unittest.main()
