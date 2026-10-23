import unittest

from seeground.grounding.active_diagnostic import _check_payload
from seeground.grounding.appearance_evidence import aggregate_visual_checks, normalize_visual_check


LIVE_REASON = (
    'Only the lampshade is white; the base and legs are dark gray/black. '
    "A bare color like 'white' does not apply to the whole object unless specified otherwise. "
    'Since the requirement is for the object to be white, and it is not, this fails.')


def visual(reason, color='white'):
    return {'category_verdict': 'yes', 'verdict': 'no', 'visibility': 'visible',
            'reason': 'table structure visible', 'attribute_checks': [
                {'attribute': color, 'verdict': 'no', 'visibility': 'visible', 'reason': reason}]}


class PartialColorCoverageTests(unittest.TestCase):
    def test_live_27111_unlisted_matching_part_cannot_add_whole_object_coverage(self):
        # Exact bare-color reason from final_recheck_8_v1 q27111 step 3.
        raw = {'id': 6, 'category': 'table', 'role': 'target', **visual(LIVE_REASON),
               'attribute_checks': [
                   {'attribute': 'the tall white table', 'verdict': 'no', 'visibility': 'visible',
                    'reason': 'Only the lampshade is white; the full object is not white.'},
                   {'attribute': 'tall', 'verdict': 'yes', 'visibility': 'visible',
                    'reason': 'The object is tall.'},
                   visual(LIVE_REASON)['attribute_checks'][0]]}
        phrases = ['the tall white table', 'tall', 'white']
        need = {'id': 6, 'category': 'table', 'role': 'target',
                'attributes': phrases, 'source_attributes': phrases}
        records = _check_payload({'checks': [raw]}, [6], [{'bbox_id': 6, 'target': 'table'}],
            {'Target': ['table']}, [need], {'source_type': 'colored_point_cloud', 'context': 'isolated'})
        record, = records
        self.assertEqual(record['verdict'], 'uncertain')
        self.assertEqual([item['verdict'] for item in record['attribute_checks']],
                         ['uncertain', 'uncertain', 'uncertain'])
        self.assertEqual(record['attribute_checks'][-1]['visual_verdict'], 'no')
        self.assertEqual(normalize_visual_check(record, phrases)['attribute_checks'],
                         record['attribute_checks'])

    def test_matching_unlisted_parts_generalize_without_color_or_category_aliases(self):
        for color, part in (('blue', 'canopy'), ('green', 'cushion'), ('red', 'handlebar')):
            with self.subTest(color=color, part=part):
                reason = f'Only the {part} is {color}; the whole object must be {color}, so this fails.'
                record = normalize_visual_check(visual(reason, color), [color], [color])
                self.assertEqual(record['verdict'], 'uncertain')
                self.assertEqual(aggregate_visual_checks([record], [color])['verdict'], 'uncertain')

    def test_live_purity_rejection_cannot_strengthen_bare_color_quote(self):
        # Exact attribute reason from mixed_live_v2, independent of instance ID.
        reason = ("The object's primary color is teal or greenish-blue with significant gray/brown tones; "
                  "it does not match the pure 'blue' attribute as stated in the source.")
        record = normalize_visual_check(visual(reason, 'blue'), ['blue'], ['blue'])
        self.assertEqual(record['verdict'], 'uncertain')
        self.assertEqual(record['attribute_checks'][0]['visual_verdict'], 'no')
        self.assertEqual(normalize_visual_check(record, ['blue'])['attribute_checks'],
                         record['attribute_checks'])

    def test_purity_quote_guard_preserves_explicit_coverage_and_clear_counterevidence(self):
        for color in ('blue', 'green', 'red'):
            with self.subTest(color=color):
                reason = f'The surface does not match the pure "{color}" attribute.'
                self.assertEqual(normalize_visual_check(visual(reason, color), [color], [color])['verdict'],
                                 'uncertain')
                phrase = f'pure {color}'
                self.assertEqual(normalize_visual_check(visual(reason, phrase), [phrase],
                                                       [phrase])['verdict'], 'no')
                self.assertEqual(normalize_visual_check(visual('The surface is clearly beige; no matching hue.',
                                                               color), [color], [color])['verdict'], 'no')

    def test_clear_nonmatching_colors_and_annotation_only_parts_remain_negative(self):
        for reason in (
                'The lampshade is beige/yellow, not white; the object is not white overall.',
                'Only the lampshade is black; the whole object is dark, not white.',
                'No parts are white; the entire object is dark brown.',
                'Only the numeric annotation box is white; the whole object is dark brown.',
                'Only the ID overlay is white; the whole object is dark brown.'):
            with self.subTest(reason=reason):
                self.assertEqual(normalize_visual_check(visual(reason), ['white'])['verdict'], 'no')

    def test_explicit_color_coverage_and_part_conditions_remain_mandatory(self):
        for phrase in ('all white', 'entirely white', 'uniformly white', 'predominantly white'):
            with self.subTest(phrase=phrase):
                self.assertEqual(normalize_visual_check(visual(LIVE_REASON, phrase), [phrase], [phrase])['verdict'],
                                 'no')
        self.assertEqual(normalize_visual_check(visual(LIVE_REASON, 'white base'),
                                               ['white base'])['verdict'], 'no')


if __name__ == '__main__':
    unittest.main()
