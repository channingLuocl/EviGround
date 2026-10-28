import unittest

from repair_scan_domain_annotations import unique_groups


class AnnotationTests(unittest.TestCase):
    def test_exact_duplicates_preserve_first_id_and_other_objects(self):
        groups = [{'objectId': 0, 'label': 'chair', 'segments': [1, 2]},
                  {'objectId': 1, 'label': 'chair', 'segments': [3]},
                  {'objectId': 2, 'label': 'chair', 'segments': [2, 1]}]
        kept, removed = unique_groups(groups)
        self.assertEqual(kept, groups[:2])
        self.assertEqual((removed[0]['retained_id'], removed[0]['duplicate_id']), (0, 2))

    def test_conflicting_label_or_partial_overlap_fails(self):
        first = {'objectId': 0, 'label': 'chair', 'segments': [1, 2]}
        for label, segments in [('desk', [1, 2]), ('chair', [2, 3])]:
            with self.assertRaises(ValueError):
                unique_groups([first, {'objectId': 1, 'label': label, 'segments': segments}])


if __name__ == '__main__':
    unittest.main()
