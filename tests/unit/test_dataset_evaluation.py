"""Predicted proposal IDs and GT IDs belong to different ScanRefer spaces."""

import json
from pathlib import Path
import tempfile
import unittest

from eviground.evaluation.reporting import evaluate, summarize
from eviground.evaluation.metrics import bbox_iou_3d


class DatasetEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "prediction.json"
        self.row = {"dataset": "scanrefer", "target_id": 5, "query": "the chair"}
        self.box = [0, 0, 0, 1, 1, 1]

    def prediction(self, oid, box):
        self.path.write_text(json.dumps({"query": self.row["query"], "category_top_k": 5,
                                        "gt_categories_supplied_to_model": False,
                                        "query_target_id_supplied_to_model": False,
                                        "predicted_id": oid, "bbox_3d": box}))

    def test_scanrefer_different_id_same_box_is_correct(self):
        self.prediction(99, self.box)
        result = evaluate(self.row, self.path, "log", 5, gt_box=self.box)
        self.assertTrue(result["correct"])
        self.assertEqual(result["iou_3d"], 1)
        self.assertEqual(result["evaluation_metric"], "iou25")

    def test_scanrefer_same_id_wrong_geometry_is_incorrect(self):
        self.prediction(5, [10, 10, 10, 1, 1, 1])
        self.assertFalse(evaluate(self.row, self.path, "log", 5, gt_box=self.box)["correct"])

    def test_scanrefer_requires_independent_gt_geometry(self):
        self.prediction(99, self.box)
        with self.assertRaisesRegex(ValueError, "independent target GT box"):
            evaluate(self.row, self.path, "log", 5)

    def test_iou_stays_within_physical_bound_despite_float_roundoff(self):
        box = [.5, .5, .5, .1, .1, .1]
        self.assertEqual(bbox_iou_3d(box, box), 1.0)

    def test_nr3d_keeps_instance_id_evaluation(self):
        self.prediction(5, [10, 10, 10, 1, 1, 1])
        row = {**self.row, "dataset": "nr3d"}
        self.assertTrue(evaluate(row, self.path, "log", 5, gt_box=self.box)["correct"])
        self.prediction(99, self.box)
        self.assertFalse(evaluate(row, self.path, "log", 5, gt_box=self.box)["correct"])

    def test_null_and_error_iou_never_count_as_hits(self):
        rows = [{"predicted_id": None, "iou_3d": 1.0},
                {"predicted_id": 3, "iou_3d": 1.0, "error": "worker failed"}]
        value = summarize(rows, total=4)
        self.assertEqual(value["acc25"], 0)
        self.assertEqual(value["acc50"], 0)
        self.assertEqual(value["pending"], 2)


if __name__ == "__main__":
    unittest.main()
