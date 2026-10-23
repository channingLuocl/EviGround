import unittest

from eviground.evaluation.metrics import bbox_iou_3d, threshold_accuracy


class MetricTests(unittest.TestCase):
    def test_identical_and_disjoint_boxes(self):
        box = [0, 0, 0, 2, 2, 2]
        self.assertEqual(bbox_iou_3d(box, box), 1.0)
        self.assertEqual(bbox_iou_3d(box, [4, 0, 0, 2, 2, 2]), 0.0)

    def test_threshold_accuracy(self):
        self.assertEqual(threshold_accuracy([0.25, 0.5, 0.8], 0.5), 2 / 3)
        self.assertIsNone(threshold_accuracy([], 0.5))


if __name__ == '__main__':
    unittest.main()
