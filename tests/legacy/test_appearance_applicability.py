import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from eviground.grounding.active_diagnostic import _check_payload, evidence_request, execute_evidence
from eviground.grounding.appearance_evidence import aggregate_visual_checks, normalize_visual_check, required_visual_attributes
from eviground.grounding.diagnostic import diagnose
from eviground.rendering.rgb_review import review_need


def attribute(name, verdict='yes', reason='visible surface evidence'):
    return {'attribute': name, 'verdict': verdict, 'visibility': 'visible', 'reason': reason}


def visual(attributes, source_type='colored_point_cloud', context='isolated'):
    return {'category_verdict': 'yes', 'verdict': 'yes', 'visibility': 'visible',
            'reason': 'category visible', 'attribute_checks': attributes,
            'render_provenance': {'source_type': source_type, 'context': context}}


class AppearanceApplicabilityTests(unittest.TestCase):
    def test_missing_category_cannot_turn_unsupported_property_no_into_category_fail(self):
        phrase = 'white with nothing underneath'
        raw = visual([attribute(phrase, 'no', 'The table has legs underneath.'),
                      attribute('white', 'yes', 'The top surface is clearly white.')])
        raw.pop('category_verdict')
        raw.update(id=83, verdict='no', reason='The white table has legs underneath.')
        objects = [{'bbox_id': 83, 'target': 'table', 'bbox_3d': [0, 0, .5, 1, 1, 1]}]
        parsed = {'Target': ['table'], 'Anchor': [], 'Relations': [], 'Observer': {},
                  'Attributes': [{'category': 'table', 'role': 'target', 'phrase': phrase},
                                 {'category': 'table', 'role': 'target', 'phrase': 'white'}],
                  'Unhandled': ['nothing underneath'],
                  'diagnostic_missing': ['unhandled condition: nothing underneath']}
        requirements = [{'id': 83, 'category': 'table', 'role': 'target',
                         'attributes': [phrase, 'white'], 'source_attributes': [phrase, 'white']}]
        records = _check_payload({'checks': [raw]}, [83], objects, parsed, requirements,
                                 {'source_type': 'colored_point_cloud', 'context': 'scene'})
        record = records[0]
        self.assertEqual(record['category_verdict'], 'uncertain')
        self.assertEqual(record['verdict'], 'uncertain')
        self.assertEqual([item['verdict'] for item in record['attribute_checks']], ['uncertain', 'yes'])
        self.assertEqual(record['attribute_checks'][0]['visual_verdict'], 'no')
        replayed = normalize_visual_check(record, [phrase, 'white'])
        self.assertEqual(replayed['attribute_checks'], record['attribute_checks'])
        self.assertEqual(replayed['category_verdict'], record['category_verdict'])
        self.assertEqual(replayed['verdict'], record['verdict'])
        diagnosis = diagnose(parsed, objects, objects, [0, 0, 0], records, trusted_categories=True)
        self.assertEqual(diagnosis['candidate_ids'], [83])
        self.assertEqual(diagnosis['status'], 'missing')

    def test_missing_category_keeps_independent_counterevidence_and_category_only_compatibility(self):
        raw = visual([attribute('green')])
        raw.pop('category_verdict')
        positive = normalize_visual_check(raw, ['green'])
        self.assertEqual(positive['category_verdict'], 'yes')
        self.assertEqual(positive['verdict'], 'yes')
        self.assertEqual(positive['attribute_checks'][0]['verdict'], 'yes')
        raw['verdict'] = 'no'
        raw['attribute_checks'] = [attribute('green', 'no', 'The surface is red, not green.')]
        negative = normalize_visual_check(raw, ['green'])
        self.assertEqual(negative['category_verdict'], 'uncertain')
        self.assertEqual(negative['verdict'], 'no')
        category_only = {'verdict': 'no', 'visibility': 'visible', 'reason': 'The instance is a chair.'}
        self.assertEqual(normalize_visual_check(category_only, [])['verdict'], 'no')
        phrase = 'white with nothing underneath'
        explicit = visual([attribute(phrase, 'no', 'There are legs underneath.')])
        explicit.update(category_verdict='no', reason='This instance is a chair, not a table.')
        self.assertEqual(normalize_visual_check(explicit, [phrase])['verdict'], 'no')

        raw = visual([])
        raw.pop('category_verdict')
        incomplete = normalize_visual_check(raw, ['green'])
        self.assertEqual(incomplete['category_verdict'], 'yes')
        self.assertEqual(incomplete['verdict'], 'uncertain')

    def test_rgb_missing_category_uses_common_unknown_gate(self):
        phrase = 'white with nothing underneath'
        need = {'id': 83, 'category': 'table', 'role': 'target',
                'attributes': [phrase, 'white'], 'source_attributes': [phrase, 'white']}
        crops = {83: [{'image': 'mock.png', 'source_frame': 10}]}
        raw = visual([attribute(phrase, 'no', 'The table has legs underneath.'),
                      attribute('white', 'yes', 'The top surface is clearly white.')])
        raw.pop('category_verdict')
        raw.update(id=83, verdict='no', reason='The white table has legs underneath.')
        response = {'_content': json.dumps(raw), '_finish_reason': 'stop',
                    '_baseline': {'latency_s': .01}}
        with patch('eviground.rendering.rgb_review.chat', return_value=response), patch(
                'eviground.rendering.rgb_review.image_part', return_value={'type': 'text', 'text': 'image'}):
            result = review_need(need, crops, [])
        self.assertEqual(result['category_verdict'], 'uncertain')
        self.assertEqual(result['verdict'], 'uncertain')
        self.assertEqual([item['verdict'] for item in result['attribute_checks']], ['uncertain', 'yes'])

    def test_bare_color_scope_guard_uses_item_and_overall_reasons(self):
        for item_reason, overall_reason in (
                ('The surface is teal.', 'It does not match the pure blue attribute.'),
                ('The surface does not appear predominantly blue.', 'The category is visible.')):
            with self.subTest(item_reason=item_reason):
                raw = visual([attribute('blue', 'no', item_reason)])
                raw['reason'] = overall_reason
                record = normalize_visual_check(raw, ['blue'], ['blue'])
                self.assertEqual(record['verdict'], 'uncertain')
                self.assertEqual(record['attribute_checks'][0]['visual_verdict'], 'no')
                replayed = normalize_visual_check(record, ['blue'])
                self.assertEqual(replayed['attribute_checks'], record['attribute_checks'])
                self.assertEqual(replayed['category_verdict'], record['category_verdict'])
                self.assertEqual(replayed['verdict'], record['verdict'])
                self.assertEqual(aggregate_visual_checks([record], ['blue'])['verdict'], 'uncertain')
        phrase = 'pure blue'
        raw = visual([attribute(phrase, 'no', 'The surface is teal.')])
        raw['reason'] = 'It does not match the pure blue attribute.'
        self.assertEqual(normalize_visual_check(raw, [phrase], [phrase])['verdict'], 'no')

    def test_live_negative_conjunction_cannot_exclude_by_unsupported_predicate(self):
        phrase = 'white with nothing underneath'
        reason = 'The table has visible dark legs underneath the white top, so it is not nothing underneath.'
        raw = visual([attribute(phrase, 'no', reason),
                      attribute('white', 'yes', 'The top surface of the table is clearly white.')])
        record = normalize_visual_check(raw, [phrase, 'white'], [phrase, 'white'])
        self.assertEqual(record['verdict'], 'uncertain')
        self.assertEqual(record['attribute_checks'][0]['verdict'], 'uncertain')
        self.assertEqual(record['attribute_checks'][0]['visual_verdict'], 'no')
        self.assertEqual(record['attribute_checks'][1]['verdict'], 'yes')
        self.assertEqual(aggregate_visual_checks([record], [phrase, 'white'])['verdict'], 'uncertain')
        parsed = {'Target': ['table'], 'Anchor': [], 'Relations': [], 'Observer': {},
                  'Attributes': [{'category': 'table', 'phrase': phrase}],
                  'Unhandled': ['nothing underneath'],
                  'diagnostic_missing': ['unhandled condition: nothing underneath']}
        obj = {'bbox_id': 83, 'target': 'table', 'bbox_3d': [0, 0, .5, 1, 1, 1]}
        self.assertEqual(normalize_visual_check(record, [phrase, 'white'])['attribute_checks'],
                         record['attribute_checks'])
        diagnosis = diagnose(parsed, [obj], [obj], [0, 0, 0],
                             [{**record, 'id': 83, 'category': 'table', 'role': 'target'}],
                             trusted_categories=True)
        self.assertEqual(diagnosis['candidate_ids'], [83])
        self.assertIn('unhandled condition: nothing underneath', diagnosis['missing'])
        self.assertEqual(diagnosis['status'], 'missing')

    def test_unsupported_negation_cannot_gain_support_or_override_independent_fail(self):
        for verdict in ('yes', 'no'):
            for phrase in ('green bin without a bag', 'not red', 'white with nothing underneath'):
                with self.subTest(phrase=phrase, verdict=verdict):
                    record = normalize_visual_check(visual([attribute(phrase, verdict)]), [phrase], [phrase])
                    self.assertEqual(record['verdict'], 'uncertain')
                    self.assertEqual(record['attribute_checks'][0]['visual_verdict'], verdict)
        phrase = 'white with nothing underneath'
        record = normalize_visual_check(visual([
            attribute(phrase, 'yes'), attribute('white', 'no', 'The top is dark brown, not white.')]),
            [phrase, 'white'], [phrase, 'white'])
        self.assertEqual(record['verdict'], 'no')

    def test_independent_views_cannot_prove_or_disprove_physical_size(self):
        for source in ('colored_point_cloud', 'real_rgb_projected_instance'):
            for verdict in ('yes', 'no'):
                for name in ('tall', 'very short', 'large', 'the tall white table'):
                    with self.subTest(source=source, verdict=verdict, name=name):
                        record = normalize_visual_check(visual([attribute(name, verdict)], source), [name])
                        self.assertEqual(record['verdict'], 'uncertain')
                        self.assertEqual(record['attribute_checks'][0]['visual_verdict'], verdict)
                        self.assertEqual(normalize_visual_check(record, [name])['attribute_checks'],
                                         record['attribute_checks'])
                        self.assertEqual(aggregate_visual_checks([record], [name])['verdict'], 'uncertain')

    def test_shape_and_clear_colors_remain_decisive_in_independent_views(self):
        for source in ('colored_point_cloud', 'real_rgb_projected_instance'):
            for name in ('skinny', 'narrow', 'thin', 'round', 'blue', 'green', 'white'):
                for verdict in ('yes', 'no'):
                    with self.subTest(source=source, name=name, verdict=verdict):
                        record = normalize_visual_check(visual([attribute(name, verdict)], source), [name])
                        self.assertEqual(record['verdict'], verdict)

    def test_shared_scene_size_checks_keep_their_existing_visual_scope(self):
        for verdict in ('yes', 'no'):
            record = normalize_visual_check(visual([attribute('tall', verdict)], context='scene'), ['tall'])
            self.assertEqual(record['verdict'], verdict)

    def test_independent_last_view_cannot_erase_earlier_shared_scene_support(self):
        scene = visual([attribute('tall')], context='scene')
        isolated = visual([attribute('tall', 'no')])
        result = aggregate_visual_checks([scene, isolated], ['tall'])
        self.assertEqual(result['verdict'], 'yes')
        self.assertEqual(result['view_evidence'][1]['verdict'], 'uncertain')
        self.assertEqual(normalize_visual_check(result, ['tall'])['verdict'], 'yes')
        self.assertEqual(aggregate_visual_checks([result], ['tall'])['verdict'], 'yes')

    def test_explicit_uncertainty_cannot_be_a_visible_negative(self):
        for reason in ('The lighting makes precise color judgment uncertain.',
                       'The tabletop is not clearly white.',
                       'Cannot determine the surface color from the sampled points.',
                       'The surface might be white, but the image lacks detail.'):
            with self.subTest(reason=reason):
                record = normalize_visual_check(visual([attribute('white', 'no', reason)]), ['white'])
                self.assertEqual(record['verdict'], 'uncertain')
                self.assertEqual(record['attribute_checks'][0]['visual_verdict'], 'no')
        clear = normalize_visual_check(visual([attribute('white', 'no', 'The top surface is dark brown, not white.')]),
                                       ['white'])
        self.assertEqual(clear['verdict'], 'no')

    def test_uncertain_category_negative_also_stays_unknown(self):
        record = normalize_visual_check({
            'category_verdict': 'no', 'verdict': 'no', 'visibility': 'visible',
            'reason': 'Cannot verify table category from these sparse points.'}, [])
        self.assertEqual(record['category_verdict'], 'uncertain')
        self.assertEqual(record['verdict'], 'uncertain')
        self.assertEqual(record['category_visual_verdict'], 'no')

    def test_q27111_live_v1_bare_white_negative_cannot_add_predominant_coverage(self):
        reason = ('Only the top surface is white; the base and legs are dark, '
                  'so the object is not predominantly white.')
        phrases = ['the tall white table', 'tall', 'white']
        expected = required_visual_attributes(phrases)
        raw = visual([
            attribute('the tall white table', 'no',
                      'The table has a white top but a dark base; the full object is not white.'),
            attribute('tall', 'yes',
                      "The table's height relative to its base and overall proportions indicate it is tall."),
            attribute('white', 'no', reason)])
        raw['verdict'] = 'no'
        result = normalize_visual_check(raw, expected, phrases)
        self.assertEqual(result['verdict'], 'uncertain')
        self.assertEqual(result['visual_verdict'], 'no')
        white = result['attribute_checks'][2]
        self.assertEqual(white['verdict'], 'uncertain')
        self.assertEqual(white['visual_verdict'], 'no')
        self.assertIn(reason, white['reason'])
        self.assertEqual(normalize_visual_check(result, expected)['attribute_checks'], result['attribute_checks'])
        self.assertEqual(aggregate_visual_checks([result], expected)['verdict'], 'uncertain')

    def test_bare_color_scope_guard_generalizes_without_disabling_clear_negatives(self):
        for color in ('white', 'blue', 'red', 'green'):
            for reason in (f'Only the top surface is {color}; the frame is dark.',
                           f'The chair has a {color} seat but dark legs; it is not entirely {color}.'):
                with self.subTest(color=color, reason=reason):
                    result = normalize_visual_check(visual([attribute(color, 'no', reason)]), [color])
                    self.assertEqual(result['verdict'], 'uncertain')
        for reason in ('The top surface is beige/tan, not white.',
                       'The top surface is dark brown, not white.',
                       'No parts are white; the body is dark brown.',
                       'The top surface is not white.',
                       'Only the annotation labels are white; the surface is brown.'):
            with self.subTest(reason=reason):
                result = normalize_visual_check(visual([attribute('white', 'no', reason)]), ['white'])
                self.assertEqual(result['verdict'], 'no')

    def test_explicit_color_part_and_coverage_contradictions_stay_decisive(self):
        reason = 'Only the top surface is white; the base is dark, so the object is not entirely white.'
        for phrase in ('entirely white', 'uniformly white', 'predominantly white', 'all white'):
            expected = required_visual_attributes([phrase])
            result = normalize_visual_check(visual([attribute(name, 'no', reason) for name in expected]),
                                            expected, [phrase])
            self.assertEqual(result['verdict'], 'no')
        part = 'white base'
        expected = required_visual_attributes([part])
        result = normalize_visual_check(visual([attribute(name, 'no', reason) for name in expected]),
                                        expected, [part])
        self.assertEqual(result['attribute_checks'][0]['verdict'], 'no')
        self.assertEqual(result['verdict'], 'no')

    def test_q8411_mixed_live_pure_blue_requirement_cannot_exclude_bare_blue(self):
        reason = ('The main body and canopy are teal, which is a bluish-green; '
                  'the object does not appear to be purely blue as required by the source quote.')
        raw = visual([attribute('blue', 'no', reason)])
        result = normalize_visual_check(raw, ['blue'], ['blue'])
        self.assertEqual(result['verdict'], 'uncertain')
        self.assertEqual(result['attribute_checks'][0]['visual_verdict'], 'no')
        self.assertIn(reason, result['attribute_checks'][0]['reason'])
        self.assertEqual(aggregate_visual_checks([result], ['blue'])['verdict'], 'uncertain')
        phrase = 'pure blue'
        expected = required_visual_attributes([phrase])
        explicit = normalize_visual_check(visual([attribute(name, 'no', reason) for name in expected]),
                                           expected, [phrase])
        self.assertEqual(explicit['verdict'], 'no')

    def test_bare_blue_matching_region_cannot_add_majority_or_dominant_coverage(self):
        reason = ("The object has a significant blue region (teal/azure) on its upper-left side, "
                  "but the majority of the object is not blue; the primary visible colors are beige, brown, and gray. "
                  "Since the requirement is for the object to be 'blue' (and no qualifier like 'partially blue' "
                  "is given), and the dominant color is not blue, the attribute fails.")
        result = normalize_visual_check(visual([attribute('blue', 'no', reason)]), ['blue'], ['blue'])
        self.assertEqual(result['verdict'], 'uncertain')
        self.assertEqual(result['attribute_checks'][0]['verdict'], 'uncertain')
        self.assertEqual(result['attribute_checks'][0]['visual_verdict'], 'no')
        self.assertIn(reason, result['attribute_checks'][0]['reason'])
        self.assertEqual(aggregate_visual_checks([result], ['blue'])['verdict'], 'uncertain')
        self.assertEqual(normalize_visual_check(result, ['blue'])['attribute_checks'], result['attribute_checks'])

    def test_explicit_blue_coverage_can_fail_despite_a_matching_region(self):
        reason = ('The object has a significant blue region, but the majority of the object is beige '
                  'and the dominant color is not blue.')
        for phrase in ('mostly blue', 'predominantly blue', 'entirely blue', 'all blue'):
            with self.subTest(phrase=phrase):
                expected = required_visual_attributes([phrase])
                result = normalize_visual_check(visual([attribute(name, 'no', reason) for name in expected]),
                                                expected, [phrase])
                self.assertEqual(result['verdict'], 'no')
                full_phrase = next(item for item in result['attribute_checks'] if item['attribute'] == phrase)
                self.assertEqual(full_phrase['verdict'], 'no')

    def test_dominant_opposite_color_with_no_blue_remains_a_clear_failure(self):
        for reason in ('No blue color is visible on this instance; the dominant colors are neutral earth tones.',
                       'The object shows no discernible blue coloring; all visible regions are in warm earth tones.',
                       'The visible body is brown, not blue; its dominant color is brown.'):
            with self.subTest(reason=reason):
                result = normalize_visual_check(visual([attribute('blue', 'no', reason)]), ['blue'], ['blue'])
                self.assertEqual(result['verdict'], 'no')
                self.assertEqual(aggregate_visual_checks([result], ['blue'])['verdict'], 'no')

    def test_blue_annotation_alone_cannot_become_positive_object_evidence(self):
        reason = 'The blue synthetic label alone satisfies the blue attribute; the object surface is brown.'
        result = normalize_visual_check(visual([attribute('blue', 'yes', reason)]), ['blue'], ['blue'])
        self.assertEqual(result['verdict'], 'uncertain')
        self.assertEqual(result['attribute_checks'][0]['visual_verdict'], 'yes')
        self.assertEqual(aggregate_visual_checks([result], ['blue'])['verdict'], 'uncertain')

    def test_q18112_incidental_black_top_hypothesis_does_not_weaken_clear_blue_no(self):
        reason = ('The object (26) is mostly off-white/cream with black at the top; '
                  'there is no visible blue coloration on any part of the object. '
                  'The black top may be misinterpreted, but it is distinctly black, not blue.')
        result = normalize_visual_check(visual([attribute('blue', 'no', reason)]), ['blue'])
        self.assertEqual(result['verdict'], 'no')
        self.assertEqual(result['attribute_checks'][0]['reason'], reason)
        self.assertEqual(aggregate_visual_checks([result], ['blue'])['verdict'], 'no')
        for reason in ('The surface may be blue, but the color is difficult to judge.',
                       'The surface could be clearly blue in another view.',
                       'The top might be white, but the image lacks detail.'):
            name = 'white' if 'white' in reason else 'blue'
            uncertain = normalize_visual_check(visual([attribute(name, 'no', reason)]), [name])
            self.assertEqual(uncertain['verdict'], 'uncertain')

    def test_actual_low_detail_contradictions_remain_unknown(self):
        raw = visual([attribute('green', 'no', 'The body is brown, not green.')])
        raw['visibility'] = 'low_detail'
        raw['attribute_checks'][0]['visibility'] = 'low_detail'
        self.assertEqual(normalize_visual_check(raw, ['green'])['verdict'], 'uncertain')

    def test_q27111_raw_negative_preserves_white_target_and_other_color_failures(self):
        phrases = ['the tall white table', 'tall', 'white']
        expected = required_visual_attributes(phrases)
        raw = visual([
            attribute('the tall white table', 'no',
                      "The object is white on top, but its height is not clearly 'tall'; the base contradicts 'nothing underneath'."),
            attribute('tall', 'no',
                      'Its height is comparable to or less than the low-profile tables in other tiles.'),
            attribute('white', 'yes', 'The tabletop surface is visibly white.')])
        raw.update(verdict='no', id=6, category='table', role='target')
        objects = [{'bbox_id': 6, 'target': 'table', 'bbox_3d': [0, 0, .5, 1, 1, 1]},
                   {'bbox_id': 9, 'target': 'table', 'bbox_3d': [2, 0, .2, 1, 1, .4]}]
        parsed = {'Target': ['table'], 'Anchor': [], 'Relations': [], 'Viewpoint': '',
                  'Observer': {'origin': {}, 'facing': {}}, 'Unhandled': ['nothing underneath'],
                  'diagnostic_missing': ['unhandled condition: nothing underneath'],
                  'Attributes': [{'category': 'table', 'role': 'target', 'phrase': phrase} for phrase in phrases]}
        requirement = {'id': 6, 'category': 'table', 'role': 'target',
                       'attributes': expected, 'source_attributes': phrases}
        checks = _check_payload({'checks': [raw]}, [6], objects, parsed, [requirement], raw['render_provenance'])
        record, = checks
        self.assertEqual(record['verdict'], 'uncertain')
        self.assertEqual(record['visual_verdict'], 'no')
        self.assertEqual([item['verdict'] for item in record['attribute_checks']], ['uncertain', 'uncertain', 'yes'])
        brown = visual([attribute(name, 'no', 'The top surface is dark brown, not white.') for name in expected])
        brown.update(id=9, category='table', role='target')
        checks.append(normalize_visual_check(brown, expected, phrases))
        diagnosis = diagnose(parsed, objects, objects, [0, 0, 0], checks, trusted_categories=True)
        self.assertEqual(diagnosis['candidate_ids'], [6])
        self.assertEqual(diagnosis['status'], 'missing')
        self.assertIn('unhandled condition: nothing underneath', diagnosis['missing'])

    def test_visual_prompt_keeps_complete_property_quotes_without_other_query_conditions(self):
        phrase = 'white stripes on top'
        expected = required_visual_attributes([phrase])
        objects = [{'bbox_id': 1, 'target': 'table', 'bbox_3d': [0, 0, .5, 1, 1, 1]}]
        parsed = {'Target': ['table'], 'Anchor': [], 'Relations': [], 'Viewpoint': '',
                  'Observer': {'origin': {}, 'facing': {}}, 'Unhandled': ['nothing underneath'],
                  'diagnostic_missing': ['unhandled condition: nothing underneath'],
                  'Attributes': [{'category': 'table', 'role': 'target', 'phrase': phrase}]}
        diagnosis = diagnose(parsed, objects, objects, [0, 0, 0], trusted_categories=True)
        action = evidence_request({'action': 'evidence', 'dimension': 'appearance', 'reason': 'inspect'}, [1], objects)
        response = {'_content': json.dumps({'checks': [{'id': 1, **visual([attribute(name) for name in expected])}]}),
                    '_finish_reason': 'stop', '_baseline': {'latency_s': .01}}
        with patch('eviground.grounding.active_diagnostic.render_action', return_value={
                'image': 'mock.png', 'focus_ids': [1], 'context': 'isolated'}), patch(
                'eviground.grounding.active_diagnostic.image_part', return_value={'type': 'text', 'text': 'image'}), patch(
                'eviground.grounding.active_diagnostic.chat', return_value=response) as api:
            execute_evidence(action, 'white stripes on top with nothing underneath', parsed, diagnosis,
                             objects, objects, np.zeros((1, 3)), None, np.array([1]), Path('/tmp'), 1, [])
        payload = json.loads(api.call_args.args[0][1]['content'][1]['text'])
        self.assertNotIn('query', payload)
        self.assertNotIn('nothing underneath', json.dumps(payload))
        self.assertEqual(payload['review_requirements'][0]['attributes'], expected)
        self.assertEqual(payload['review_requirements'][0]['attribute_contexts'][1]['source_quotes'], [phrase])

    def test_rgb_crop_scale_guard_is_applied_before_normalization(self):
        need = {'id': 1, 'category': 'table', 'role': 'target', 'attributes': ['tall', 'white']}
        crops = {1: [{'image': 'mock.png', 'source_frame': 10}]}
        for verdict in ('yes', 'no'):
            response = {'_content': json.dumps({'id': 1, **visual([
                attribute('tall', verdict), attribute('white')])}),
                '_finish_reason': 'stop', '_baseline': {'latency_s': .01}}
            with patch('eviground.rendering.rgb_review.chat', return_value=response), patch(
                    'eviground.rendering.rgb_review.image_part', return_value={'type': 'text', 'text': 'image'}):
                result = review_need(need, crops, [])
            self.assertEqual(result['verdict'], 'uncertain')
            self.assertEqual([item['verdict'] for item in result['attribute_checks']], ['uncertain', 'yes'])


if __name__ == '__main__':
    unittest.main()
