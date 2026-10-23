import unittest

import numpy as np

from seeground.classifiers.mask3d_matching import associate, unique_predictions


class MatchingTest(unittest.TestCase):
    def test_permutation_and_missed_object(self):
        ids = np.array([8, 8, 4, 4, 2, 2])
        masks = np.column_stack([ids == 4, ids == 8])
        self.assertEqual(associate(masks, ids, [8, 4, 2]), {0: (1, 1.), 1: (0, 1.)})
        self.assertEqual(associate(masks[:, ::-1], ids, [4, 2, 8]), {0: (1, 1.), 2: (0, 1.)})

    def test_duplicate_labels_cannot_match_two_objects(self):
        masks = np.ones((4, 2), dtype=bool)
        masks, scores, labels = unique_predictions(masks, [.2, .8], ['chair', 'table'])
        self.assertEqual(labels.tolist(), ['table'])
        self.assertEqual(len(associate(masks, np.array([1, 1, 2, 2]), [1, 2])), 1)

    def test_below_threshold_and_empty_proposals(self):
        ids = np.array([1, 1, 2, 2, 2])
        self.assertEqual(associate(np.ones((5, 1), dtype=bool), ids, [1]), {})
        self.assertEqual(associate(np.empty((5, 0), dtype=bool), ids, [1]), {})


if __name__ == '__main__':
    unittest.main()
