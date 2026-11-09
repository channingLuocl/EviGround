"""Validate ScanRefer raw preparation without network or GPU access."""

import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from plyfile import PlyData, PlyElement


spec = importlib.util.spec_from_file_location(
    "prepare_scanrefer_raw", Path(__file__).resolve().parents[2] / "scripts/prepare_scanrefer_raw.py"
)
raw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(raw)


class RawPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        cache = root / "cache"
        cache.mkdir()
        self.scene = "scene0000_00"
        vertices = np.array([(0., 0., 0.), (2., 4., 6.), (9., 9., 9.)],
                            dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        PlyData([PlyElement.describe(vertices, "vertex")]).write(
            cache / (self.scene + "_vh_clean_2.ply"))
        (cache / (self.scene + ".txt")).write_text(
            "axisAlignment = 1 0 0 10 0 1 0 20 0 0 1 30 0 0 0 1\n")
        (cache / (self.scene + raw.SUFFIXES[2])).write_text(
            json.dumps({"segIndices": [3, 3, 4]}))
        (cache / (self.scene + raw.SUFFIXES[3])).write_text(json.dumps({"segGroups": [
            {"objectId": 17, "segments": [3], "label": "chair"},
            {"objectId": 42, "segments": [4], "label": "table"}]}))
        self.args = argparse.Namespace(cache=cache, scannet=root / "unused", output=root / "out")
        self.args.output.mkdir()

    def test_preserves_gt_ids_and_applies_alignment(self):
        record = raw.prepare_scene(self.scene, {17}, self.args)
        path = self.args.output / "evaluation_gt" / self.scene / "objects_gt.json"
        objects = json.loads(path.read_text())
        self.assertEqual([o["bbox_id"] for o in objects], [17, 42])
        np.testing.assert_allclose(objects[0]["bbox_3d"], [11, 22, 33, 2, 4, 6])
        self.assertEqual(record["evaluation_gt_sha256"], raw.digest(path))
        self.assertEqual(record["vertices"], 3)
        self.assertFalse((self.args.output / "scene.npz").exists())

    def test_rejects_missing_referenced_object(self):
        with self.assertRaisesRegex(ValueError, "Missing referenced GT objects"):
            raw.prepare_scene(self.scene, {99}, self.args)


if __name__ == "__main__":
    unittest.main()
