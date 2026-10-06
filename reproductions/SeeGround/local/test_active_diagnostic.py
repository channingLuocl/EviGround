import unittest

from active_diagnostic import _check_payload, compute_observation
from diagnostic import diagnose


def obj(oid, category, x, y, z=.5):
    return {'bbox_id': oid, 'target': category,
            'bbox_3d': [x, y, z, .4, .4, 1.]}


def parsed():
    return {
        'Target': ['window'], 'Anchor': [], 'Relations': [],
        'Observer': {'origin': {}, 'facing': {}}, 'Viewpoint': '',
        'Attributes': [], 'Unhandled': [], 'diagnostic_missing': [],
    }


class ActiveDiagnosticTests(unittest.TestCase):
    def test_each_dimension_is_an_explicit_model_selectable_tool(self):
        objects = [obj(1, 'window', 0, 0), obj(2, 'window', 2, 0)]
        diagnosis = {
            'candidate_ids': [1, 2], 'geometry': {
                'relation_pairs': [], 'between_candidate_evidence': [],
                'between_triplets': [], 'candidate_metrics': [],
                'reference_metrics': [],
            }, 'hypotheses': [], 'visual_needs': [],
        }
        for dimension in ('viewpoint', 'position', 'distance', 'appearance', 'size'):
            with self.subTest(dimension=dimension):
                result = compute_observation(dimension, 'window', parsed(), diagnosis,
                                             objects, objects)
                self.assertEqual(result['tool'], 'compute')
                self.assertEqual(result['dimension'], dimension)

    def test_visual_check_only_accepts_rendered_candidate_ids(self):
        objects = [obj(1, 'window', 0, 0), obj(2, 'window', 2, 0)]
        result = _check_payload(
            {'checks': [{'id': 1, 'verdict': 'no', 'reason': 'visible mismatch'}]},
            [1], objects, parsed())
        self.assertEqual(result[0]['id'], 1)
        self.assertEqual(result[0]['verdict'], 'no')
        with self.assertRaises(ValueError):
            _check_payload(
                {'checks': [{'id': 2, 'verdict': 'no', 'reason': 'not rendered'}]},
                [1], objects, parsed())

    def test_visual_no_excludes_candidate_with_annotated_categories(self):
        objects = [obj(1, 'window', 0, 0), obj(2, 'window', 2, 0)]
        initial = diagnose(parsed(), objects, objects, [0, 0, .5], trusted_categories=True)
        self.assertEqual(initial['candidate_ids'], [1, 2])
        checked = diagnose(parsed(), objects, objects, [0, 0, .5],
                           checks=[{'id': 1, 'category': 'window', 'verdict': 'no'}],
                           trusted_categories=True)
        self.assertEqual(checked['candidate_ids'], [2])
        self.assertEqual(checked['status'], 'resolved')

    def test_visual_uncertain_preserves_candidate(self):
        objects = [obj(1, 'window', 0, 0), obj(2, 'window', 2, 0)]
        checked = diagnose(parsed(), objects, objects, [0, 0, .5],
                           checks=[{'id': 1, 'category': 'window', 'verdict': 'uncertain'}],
                           trusted_categories=True)
        self.assertEqual(checked['candidate_ids'], [1, 2])
        self.assertEqual(checked['status'], 'missing')


if __name__ == '__main__':
    unittest.main()
