"""ScanRefer grouping and IoU reporting must not use proposal ID equality."""

import argparse
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "scanrefer", Path(__file__).resolve().parents[2] / "scripts/scanrefer.py"
)
scanrefer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scanrefer)


class ScanReferTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.annotations = self.root / "val.json"
        self.labels = self.root / "labels.tsv"
        self.manifest = self.root / "manifest.json"
        self.annotations.write_text(json.dumps([
            {"scene_id": "scene0000_00", "object_id": str(target),
             "ann_id": str(ann), "object_name": name, "description": query}
            for target, ann, name, query in (
                (1, 0, "office_chair", "chair near the desk"),
                (1, 1, "office_chair", "the same chair"),
                (2, 0, "armchair", "the other chair"),
                (3, 0, "table", "the only table"),
            )
        ]))
        self.labels.write_text("raw_category\tnyu40class\n"
                               "office chair\tchair\narmchair\tchair\ntable\ttable\n")
        with redirect_stdout(io.StringIO()):
            scanrefer.prepare(argparse.Namespace(
                annotations=self.annotations, labels=self.labels, output=self.manifest
            ))
        self.rows = json.loads(self.manifest.read_text())["rows"]

    def report(self, results):
        path = self.root / "results.json"
        output = self.root / "metrics.json"
        path.write_text(json.dumps({"results": results}))
        with redirect_stdout(io.StringIO()):
            scanrefer.report(argparse.Namespace(manifest=self.manifest, results=path,
                                               proposals="mask3d", output=output))
        return json.loads(output.read_text())

    def test_grouping_merges_raw_synonyms_and_counts_each_object_once(self):
        self.assertEqual([r["unique"] for r in self.rows], [False, False, False, True])
        self.assertNotIn("object_name", self.rows[0])

    def test_iou_controls_accuracy_and_pending_queries_stay_in_denominator(self):
        row = {**self.rows[0], "predicted_id": 99, "correct": False, "iou_3d": 0.5}
        value = self.report([row])
        self.assertEqual(value["overall"]["acc50"], 0.25)
        self.assertEqual(value["multiple"]["acc50"], 1 / 3)
        self.assertEqual(value["unique"]["acc50"], 0)
        self.assertEqual(value["pending"], 3)
        self.assertFalse(value["complete"])

    def test_error_and_null_predictions_do_not_count_as_hits(self):
        results = [
            {**r, "predicted_id": None if i == 0 else 5, "iou_3d": 1.0,
             **({"error": "worker failed"} if i == 1 else {})}
            for i, r in enumerate(self.rows)
        ]
        value = self.report(results)
        self.assertTrue(value["complete"])
        self.assertEqual(value["overall"]["acc25"], 0.5)
        self.assertEqual(value["errors"], 1)

    def test_mismatched_or_duplicate_results_are_rejected(self):
        row = {**self.rows[0], "predicted_id": 99, "iou_3d": 0.4}
        with self.assertRaisesRegex(ValueError, "disagree"):
            self.report([{**row, "query": "another query"}])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.report([row, row])

    def test_nan_iou_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid 3D IoU"):
            self.report([{**self.rows[0], "predicted_id": 99, "iou_3d": float("nan")}])

    def test_legacy_float_roundoff_at_one_is_accepted(self):
        value = self.report([{**self.rows[0], "predicted_id": 99, "iou_3d": 1.0000000000000009}])
        self.assertEqual(value["overall"]["acc50"], .25)

    def test_out_of_range_iou_beyond_roundoff_is_rejected(self):
        for value in (-.01, 1.001):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Invalid 3D IoU"):
                self.report([{**self.rows[0], "predicted_id": 99, "iou_3d": value}])

    def test_cli_rejects_placeholder_without_traceback_or_output(self):
        output = self.root / "not_created.json"
        result = subprocess.run([
            sys.executable, str(Path(scanrefer.__file__)), "prepare",
            "--annotations", "/absolute/path/to/ScanRefer_filtered_val.json",
            "--labels", str(self.labels), "--output", str(output),
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("占位路径", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse(output.exists())

    def test_cli_rejects_missing_annotation_without_traceback(self):
        result = subprocess.run([
            sys.executable, str(Path(scanrefer.__file__)), "prepare",
            "--annotations", str(self.root / "missing.json"),
            "--labels", str(self.labels), "--output", str(self.root / "unused.json"),
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("输入文件不存在", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
