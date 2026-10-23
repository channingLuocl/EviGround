import unittest

from seeground.grounding.active_diagnostic import _check_payload
from seeground.grounding.appearance_evidence import aggregate_visual_checks, normalize_visual_check


def visual(reason, name='white', verdict='yes', category_reason='object structure visible'):
    return {'id': 9, 'category': 'curtain', 'role': 'target',
            'category_verdict': 'yes', 'verdict': 'yes', 'visibility': 'visible',
            'reason': category_reason,
            'attribute_checks': [{'attribute': name, 'verdict': verdict,
                                  'visibility': 'visible', 'reason': reason}]}


class AnnotationEvidenceTests(unittest.TestCase):
    def test_saved_trial_curtain_label_positive_cannot_certify_surface_white(self):
        # Exact positive reasons from trial q06990, visual check step 4.
        raw = visual(
            "A white rectangular label with red '9' is clearly visible on the object's surface in the image. "
            "The presence of this white area satisfies the 'white' attribute requirement, "
            'even if it is only a small part of the object.', category_reason=(
                'The object is a vertical, textured, elongated structure consistent with a curtain; '
                'the point-cloud rendering shows a dark, mottled surface but also includes a clearly '
                "visible white label with red '9'—the source attribute requires 'white', and the white "
                'label qualifies as a visible white surface component of the target instance.'))
        payload = {'checks': [raw]}
        objects = [{'bbox_id': 9, 'target': 'curtain', 'bbox_3d': [0, 0, 1, 1, .1, 2]}]
        parsed = {'Target': ['curtain'], 'Attributes': [{'category': 'curtain', 'phrase': 'white'}]}
        requirements = [{'id': 9, 'category': 'curtain', 'role': 'target',
                         'attributes': ['white'], 'source_attributes': ['white']}]
        records = _check_payload(payload, [9], objects, parsed,
            requirements, {'image': 'active_view_04_instance_9.png',
                           'source_type': 'colored_point_cloud', 'context': 'isolated'})
        record, = records
        self.assertEqual(record['category_verdict'], 'yes')
        self.assertEqual(record['verdict'], 'uncertain')
        self.assertEqual(record['attribute_checks'][0]['visual_verdict'], 'yes')
        self.assertIn('annotation', record['attribute_checks'][0]['evidence_limit'])
        self.assertEqual(normalize_visual_check(record, ['white'])['attribute_checks'],
                         record['attribute_checks'])

    def test_guard_generalizes_to_other_ids_and_attributes(self):
        for name, reason in (
                ('red', 'The red ID number supplies the red attribute evidence.'),
                ('green', 'The green synthetic overlay satisfies the green requirement.'),
                ('rectangular', 'The rectangular annotation box qualifies as rectangular evidence.')):
            with self.subTest(name=name):
                self.assertEqual(normalize_visual_check(visual(reason, name), [name])['verdict'],
                                 'uncertain')

    def test_overall_reason_explicitly_using_annotation_also_invalidates_attribute(self):
        record = visual('A visible white component matches the requested color.', category_reason=(
            'The white numeric ID label is an annotation; '
            'this white area satisfies the white requirement.'))
        self.assertEqual(normalize_visual_check(record, ['white'])['verdict'], 'uncertain')

    def test_id_location_ignored_annotations_and_real_surface_color_remain_valid(self):
        for reason in (
                'The object labeled 9 has a white surface.',
                'Ignore the white annotation box; the curtain surface is white.',
                'The red ID number is ignored and the white body satisfies the white attribute.',
                'The white label is an annotation, not part of the surface; the curtain fabric is white.',
                'The artificial annotation does not support white; the white upholstery supports white.',
                'The curtain has a printed white care label that satisfies white.',
                'The actual white manufacturer label satisfies white on the fabric.'):
            with self.subTest(reason=reason):
                self.assertEqual(normalize_visual_check(visual(reason), ['white'])['verdict'], 'yes')

    def test_artificial_mark_mentions_do_not_change_the_supporting_surface_subject(self):
        for name, reason in (
                ('green', 'The green chair has an ID label, and the clearly green upholstery supports the green attribute.'),
                ('white', 'The white fabric next to the artificial annotation clearly satisfies the white attribute.'),
                ('green', 'The green physical fabric overlay supports the green attribute.'),
                ('green', 'The synthetic green fabric overlay supports the green description.'),
                ('white', 'The white synthetic fabric supplies the white attribute evidence.')):
            with self.subTest(reason=reason):
                self.assertEqual(normalize_visual_check(visual(reason, name), [name])['verdict'], 'yes')

    def test_exclusion_of_one_mark_cannot_hide_positive_use_of_another_mark(self):
        for reason in (
                'The white numeric ID label supports white, even though it is not part of the actual fabric.',
                'The artificial annotation is ignored but the white ID label qualifies as the white surface component.'):
            with self.subTest(reason=reason):
                self.assertEqual(normalize_visual_check(visual(reason), ['white'])['verdict'], 'uncertain')

    def test_annotation_exclusion_cannot_weaken_a_visible_surface_negative(self):
        reason = ('The annotation label 7 is the only white area, not part of the object; '
                  'the object surface is dark brown, not white.')
        self.assertEqual(normalize_visual_check(visual(reason, verdict='no'), ['white'])['verdict'],
                         'no')

    def test_invalid_positive_cannot_override_valid_evidence_from_another_view(self):
        invalid = visual("A white label with red '9' is visible. "
                         'The presence of this white area satisfies the white attribute.')
        for verdict, reason in (
                ('yes', 'The actual curtain fabric is white.'),
                ('no', 'The curtain fabric is dark brown, not white.')):
            with self.subTest(verdict=verdict):
                valid = visual(reason, verdict=verdict)
                combined = aggregate_visual_checks([valid, invalid], ['white'])
                self.assertEqual(combined['verdict'], verdict)
                self.assertEqual(combined['view_evidence'][1]['verdict'], 'uncertain')
                self.assertEqual(normalize_visual_check(combined, ['white'])['verdict'], verdict)
                self.assertEqual(aggregate_visual_checks([combined], ['white'])['verdict'], verdict)
        self.assertEqual(aggregate_visual_checks([invalid], ['white'])['verdict'], 'uncertain')


if __name__ == '__main__':
    unittest.main()
