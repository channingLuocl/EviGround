import unittest

from seeground.grounding.appearance_evidence import (aggregate_visual_checks, attribute_contexts, attribute_units,
                                  normalize_visual_check, required_visual_attributes)


def attribute(name, verdict='yes', visibility='visible'):
    return {'attribute': name, 'verdict': verdict, 'visibility': visibility,
            'reason': name + ' surface evidence'}


def view(attributes, **kwargs):
    return {'verdict': 'yes', 'visibility': 'visible', 'reason': 'category visible',
            'attribute_checks': attributes, **kwargs}


class AppearanceEvidenceTests(unittest.TestCase):
    def test_modifiers_are_independent_but_rich_properties_keep_their_quote(self):
        self.assertEqual(attribute_units('tall skinny wooden'), ['tall', 'skinny', 'wooden'])
        self.assertEqual(attribute_units('red details on the door'), ['red details on the door'])
        self.assertEqual(required_visual_attributes(['red largest', 'tall']), ['red', 'tall'])

    def test_qualified_color_retains_full_phrase_and_exact_source_fragments(self):
        phrase = 'Dark BLUE stripes on top and on the bottom shelf'
        units = attribute_units(phrase)
        self.assertEqual(units, [phrase, 'BLUE', 'stripes', 'on top', 'bottom shelf'])
        self.assertTrue(all(unit in phrase for unit in units))
        self.assertEqual(required_visual_attributes(units), units)
        self.assertEqual(attribute_units('very tall'), ['very tall', 'tall'])

    def test_real_stripe_description_exposes_named_properties_and_parts(self):
        phrase = ('has different colored stripes on top, maybe painted, with blue, red, '
                  'yellow, green, and blue, and then again on the bottom shelf')
        self.assertEqual(attribute_units(phrase), [phrase, 'stripes', 'blue', 'red',
                                                  'yellow', 'green', 'on top', 'bottom shelf'])
        # "Maybe painted" is not converted into a definite painted property.
        self.assertNotIn('painted', attribute_units(phrase))

    def test_negation_and_external_references_are_kept_whole(self):
        for phrase in ('not blue but red', 'red without blue stripes',
                       'red rather than blue', 'blue next to the red chair',
                       'red details on the door', 'with a blue chair',
                       'blue stripes on top of the table', 'green note attached to it'):
            with self.subTest(phrase=phrase):
                self.assertEqual(attribute_units(phrase), [phrase])

    def test_component_or_generic_yes_cannot_certify_qualified_property(self):
        phrase = 'dark blue stripes on top'
        expected = required_visual_attributes([phrase])
        components = [attribute(name) for name in expected if name != phrase]
        for checks in ([], components):
            with self.subTest(checks=checks):
                record = normalize_visual_check(view(checks), expected)
                self.assertEqual(record['verdict'], 'uncertain')
                self.assertEqual(record['attribute_checks'][0]['attribute'], phrase)
                self.assertEqual(record['attribute_checks'][0]['verdict'], 'uncertain')
        complete = normalize_visual_check(view([attribute(name) for name in expected]), expected)
        self.assertEqual(complete['verdict'], 'yes')

    def test_component_views_need_independent_complete_phrase_evidence(self):
        phrase = 'dark blue stripes on top'
        expected = required_visual_attributes([phrase])
        components = [view([attribute(name)]) for name in expected if name != phrase]
        self.assertEqual(aggregate_visual_checks(components, expected)['verdict'], 'uncertain')
        complete = aggregate_visual_checks(components + [view([attribute(phrase)])], expected)
        self.assertEqual(complete['verdict'], 'yes')

    def test_qualified_fragment_no_cannot_exclude_a_part_colored_object(self):
        phrase = 'white stripes on top'
        expected = required_visual_attributes([phrase])
        checks = [attribute(name, 'no' if name == 'white' else 'yes') for name in expected]
        record = normalize_visual_check(view(checks), expected)
        self.assertEqual(record['verdict'], 'uncertain')
        white = next(item for item in record['attribute_checks'] if item['attribute'] == 'white')
        self.assertEqual(white['verdict'], 'uncertain')
        self.assertEqual(white['source_quotes'], [phrase])
        self.assertEqual(white['qualified_fragment_verdict'], 'no')
        normalized_again = normalize_visual_check(record, expected)
        self.assertEqual(normalized_again['attribute_checks'], record['attribute_checks'])
        self.assertEqual(aggregate_visual_checks([view(checks)], expected)['verdict'], 'uncertain')

    def test_whole_qualified_property_no_and_independent_modifier_no_still_exclude(self):
        phrase = 'white stripes on top'
        expected = required_visual_attributes([phrase])
        checks = [attribute(name, 'no' if name == phrase else 'yes') for name in expected]
        self.assertEqual(normalize_visual_check(view(checks), expected)['verdict'], 'no')
        self.assertEqual(aggregate_visual_checks([view(checks)], expected)['verdict'], 'no')
        independent = normalize_visual_check(view([attribute('red', 'no'), attribute('wooden')]),
                                            ['red', 'wooden'])
        self.assertEqual(independent['verdict'], 'no')
        self.assertEqual(independent['attribute_checks'][0]['source_quotes'], ['red'])

    def test_fragment_context_retains_every_quoted_qualification(self):
        first, second = 'white stripes on top', 'white bottom shelf'
        expected = required_visual_attributes([first, second])
        contexts = {item['attribute']: item for item in attribute_contexts(expected)}
        self.assertEqual(contexts['white']['source_quotes'], [first, second])
        self.assertTrue(contexts['white']['qualified_fragment'])
        self.assertFalse(contexts[first]['qualified_fragment'])

    def test_explicit_independent_color_no_stays_decisive_with_qualified_color(self):
        phrase = 'white stripes on top'
        sources = ['white', phrase]
        expected = required_visual_attributes(sources)
        checks = [attribute(name, 'no' if name == 'white' else 'yes') for name in expected]
        record = normalize_visual_check(view(checks), expected, source_phrases=sources)
        self.assertEqual(record['verdict'], 'no')
        white = next(item for item in record['attribute_checks'] if item['attribute'] == 'white')
        self.assertEqual(white['source_quotes'], sources)
        self.assertFalse(white['qualified_fragment'])
        self.assertEqual(white['verdict'], 'no')
        self.assertEqual(normalize_visual_check(record, expected)['attribute_checks'],
                         record['attribute_checks'])

    def test_independent_color_no_survives_complementary_qualified_views(self):
        phrase = 'white stripes on top'
        sources = ['white wooden', phrase]
        expected = required_visual_attributes(sources)
        checks = [view([attribute('white', 'no')]),
                  view([attribute(name) for name in expected if name != 'white'])]
        record = aggregate_visual_checks(checks, expected, source_phrases=sources)
        self.assertEqual(record['verdict'], 'no')
        white = next(item for item in record['attribute_checks'] if item['attribute'] == 'white')
        self.assertEqual(white['source_quotes'], sources)
        self.assertFalse(white['qualified_fragment'])
        normalized = [normalize_visual_check(check, expected, source_phrases=sources)
                      for check in checks]
        self.assertEqual(aggregate_visual_checks(normalized, expected)['verdict'], 'no')

    def test_explicit_rich_sources_keep_component_contradictions_uncertain(self):
        phrase = 'white stripes on top'
        expected = required_visual_attributes([phrase])
        check = view([attribute(name, 'no' if name == 'white' else 'yes') for name in expected])
        record = normalize_visual_check(check, expected, source_phrases=[phrase])
        self.assertEqual(record['verdict'], 'uncertain')
        self.assertEqual(normalize_visual_check(record, expected)['attribute_checks'],
                         record['attribute_checks'])
        self.assertEqual(aggregate_visual_checks([record], expected)['verdict'], 'uncertain')

    def test_every_attribute_requires_one_visible_independent_verdict(self):
        for attributes in ([], [attribute('red')], [attribute('red'), attribute('red')],
                           [attribute('red'), attribute('wooden'), attribute('black')]):
            with self.subTest(attributes=attributes):
                record = normalize_visual_check(view(attributes), ['red', 'wooden'])
                self.assertEqual(record['verdict'], 'uncertain')
                self.assertEqual([item['attribute'] for item in record['attribute_checks']],
                                 ['red', 'wooden'])

    def test_occlusion_and_low_detail_cannot_prove_an_attribute(self):
        for visibility in ('occluded', 'low_detail', 'unknown', 'missing_view', 'unmarked'):
            with self.subTest(visibility=visibility):
                record = normalize_visual_check(view([attribute('red', visibility=visibility)]), ['red'])
                self.assertEqual(record['verdict'], 'uncertain')
                self.assertEqual(record['attribute_checks'][0]['verdict'], 'uncertain')

    def test_missing_visibility_is_unknown_for_attribute_checks(self):
        record = normalize_visual_check(view([{'attribute': 'red', 'verdict': 'yes',
                                              'reason': 'surface looks red'}]), ['red'])
        self.assertEqual(record['verdict'], 'uncertain')
        self.assertEqual(record['attribute_checks'][0]['visibility'], 'unknown')

    def test_omitted_attributes_stay_uninspected_across_normalization_and_views(self):
        omitted = normalize_visual_check(view([attribute('red', 'uncertain')]),
                                         ['red', 'wooden'])
        self.assertEqual([item['inspection_status'] for item in omitted['attribute_checks']],
                         ['inspected', 'uninspected'])
        again = normalize_visual_check(omitted, ['red', 'wooden'])
        self.assertEqual(again['attribute_checks'], omitted['attribute_checks'])
        combined = aggregate_visual_checks([again], ['red', 'wooden'])
        self.assertEqual([item['inspection_status'] for item in combined['attribute_checks']],
                         ['inspected', 'uninspected'])
        completed = aggregate_visual_checks([again, view([attribute('wooden', 'uncertain')])],
                                           ['red', 'wooden'])
        self.assertTrue(all(item['inspection_status'] == 'inspected'
                            for item in completed['attribute_checks']))
        self.assertEqual(completed['verdict'], 'uncertain')

    def test_synthetic_category_fallback_stays_uninspected(self):
        fallback = normalize_visual_check(view([], inspection_status='uninspected'), [])
        self.assertEqual(fallback['verdict'], 'uncertain')
        self.assertEqual(normalize_visual_check(fallback, [])['inspection_status'], 'uninspected')
        self.assertEqual(aggregate_visual_checks([fallback], [])['inspection_status'], 'uninspected')

    def test_uninspected_or_invalid_status_cannot_certify_visible_yes(self):
        for status in ('uninspected', 'invalid'):
            with self.subTest(status=status):
                row = attribute('rainbow top')
                row['inspection_status'] = status
                record = normalize_visual_check(view([row]), ['rainbow top'])
                self.assertEqual(record['verdict'], 'uncertain')
                self.assertEqual(record['attribute_checks'][0]['inspection_status'], 'uninspected')
                self.assertEqual(aggregate_visual_checks([record], ['rainbow top'])['verdict'], 'uncertain')

    def test_legacy_generic_yes_cannot_prove_attributes(self):
        check = {'verdict': 'yes', 'reason': 'looks right'}
        self.assertEqual(normalize_visual_check(check, [])['verdict'], 'yes')
        self.assertEqual(normalize_visual_check(check, ['red'])['verdict'], 'uncertain')

    def test_visible_contradiction_can_exclude_with_another_attribute_unknown(self):
        record = normalize_visual_check(view([attribute('red', 'no'),
                                              attribute('wooden', 'yes', 'occluded')]),
                                        ['red', 'wooden'])
        self.assertEqual(record['verdict'], 'no')
        self.assertEqual(record['attribute_checks'][1]['verdict'], 'uncertain')

    def test_complementary_views_accumulate_without_losing_known_attributes(self):
        checks = [view([attribute('red')]), view([attribute('wooden')])]
        record = aggregate_visual_checks(checks, ['red', 'wooden'])
        self.assertEqual(record['verdict'], 'yes')
        self.assertEqual([item['verdict'] for item in record['attribute_checks']], ['yes', 'yes'])
        self.assertEqual(len(record['view_evidence']), 2)
        occluded = aggregate_visual_checks(checks + [view([], visibility='occluded')],
                                            ['red', 'wooden'])
        self.assertEqual(occluded['verdict'], 'yes')

    def test_conflicting_visible_views_remain_uncertain(self):
        record = aggregate_visual_checks([view([attribute('red')]),
                                          view([attribute('red', 'no')])], ['red'])
        self.assertEqual(record['verdict'], 'uncertain')
        self.assertEqual(record['attribute_checks'][0]['verdict'], 'uncertain')


if __name__ == '__main__':
    unittest.main()
