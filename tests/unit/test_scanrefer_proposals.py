"""Predicted masks must retain full geometry when proposals overlap."""

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


scripts = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("prepare_scanrefer_proposals", scripts / "prepare_scanrefer_proposals.py")
proposals = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proposals)


class ProposalGeometryTests(unittest.TestCase):
    def test_overlapping_masks_keep_shared_vertices_and_original_boxes(self):
        xyz = np.array([[0., 0., 0.], [1., 1., 1.], [2., 0., 2.],
                        [3., 1., 3.], [9., 9., 9.]])
        rgb = np.arange(15, dtype=np.uint8).reshape(5, 3)
        masks = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=bool)
        points, colors, ids, objects, rejected = proposals.expand_proposals(xyz, rgb, masks)
        self.assertEqual(rejected, [2])
        self.assertEqual(len(objects), 2)
        for oid in (0, 1):
            np.testing.assert_array_equal(points[ids == oid], xyz[masks[:, oid]])
            np.testing.assert_array_equal(colors[ids == oid], rgb[masks[:, oid]])
        np.testing.assert_array_equal(points[ids == -1], xyz[[4]])
        np.testing.assert_allclose(objects[0]["bbox_3d"], [1, .5, 1, 2, 1, 2])
        np.testing.assert_allclose(objects[1]["bbox_3d"], [2, .5, 2, 2, 1, 2])

    def test_rejects_mask_with_wrong_vertex_count(self):
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            proposals.expand_proposals(np.ones((5, 3)), np.ones((5, 3)), np.ones((4, 2)))


if __name__ == "__main__":
    unittest.main()
