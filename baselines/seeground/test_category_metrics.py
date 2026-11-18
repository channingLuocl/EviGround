import unittest

import numpy as np

from category_metrics import associate, model_class_names, ranked_categories, summarize, unique_mask_indices


class CategoryMetricsTest(unittest.TestCase):
    def test_scannet_mapping_inverts_training_offset(self):
        labels = ['wall', 'chair', 'floor', 'table', *[f'label_{i}' for i in range(196)]]
        names = model_class_names(labels)
        self.assertEqual(len(names), 198)
        self.assertEqual(names[:2], ['chair', 'table'])
        scores = np.zeros((1, 200))
        scores[0, [0, 1, 2, 3, 4]] = [.9, .8, .7, .6, .5]
        scores[0, 199] = .95  # unused output, not a real category
        indices, ranked = ranked_categories(scores, names)
        self.assertEqual(indices.tolist(), [[0, 1, 2, 3, 4]])
        self.assertEqual(ranked[0][:2], ['chair', 'table'])

    def test_duplicate_masks_keep_highest_confidence(self):
        masks = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=bool)
        self.assertEqual(unique_mask_indices(masks, [.2, .8, .9]), [1])

    def test_matching_is_one_to_one_and_keeps_misses(self):
        ids = np.array([3, 3, 7, 7, 9, 9])
        masks = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [0, 0], [0, 0]], dtype=bool)
        self.assertEqual(associate(masks, ids, [3, 7, 9]), {0: (0, 1.), 1: (1, 1.)})
        self.assertEqual(associate(masks[:, :1], ids, [3, 7]), {0: (0, 1.)})
        self.assertEqual(associate(masks[:, :0], ids, [3]), {})

    def test_exact_metrics_do_not_normalize_aliases_or_drop_misses(self):
        rows = [dict(scene_id='a', object_id=0, matched=True, gt_label='chair', top5=['chair']),
                dict(scene_id='a', object_id=1, matched=True, gt_label='sofa', top5=['couch', 'sofa']),
                dict(scene_id='a', object_id=2, matched=False, gt_label='table', top5=[])]
        metrics = summarize(rows)
        self.assertEqual(metrics['objects'], 3)
        self.assertEqual(metrics['unmatched'], 1)
        self.assertEqual(metrics['exact_top1_correct'], 1)
        self.assertEqual(metrics['exact_top5_correct'], 2)

    def test_assignment_prefers_two_valid_matches_over_one(self):
        ids = np.array([1, 1, 2, 2])
        masks = np.array([[1, 1], [1, 1], [1, 0], [1, 0]], dtype=bool)
        self.assertEqual(associate(masks, ids, [1, 2]), {0: (1, 1.), 1: (0, .5)})


if __name__ == '__main__':
    unittest.main()
