from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from eviground.rendering.overview import scene_views


class RenderingContextTests(unittest.TestCase):
    def test_topdown_retains_supplemental_geometry_without_inventing_categories(self):
        objects = [
            {"bbox_id": 1, "target": "cabinet", "bbox_3d": [0, 0, 1, 1, 1, 2],
             "category_hypotheses": [{"category": "cabinet", "rank": 1}]},
            {"bbox_id": 99, "target": "object", "bbox_3d": [2, 0, 1, .2, 4, 2]},
        ]
        original = deepcopy(objects)
        xyz = np.array([[0., 0., 0.], [.2, .2, 2.], [2., -2., 0.], [2., 2., 2.]])
        ids = np.array([1, 1, 99, 99])
        rgb = np.zeros((4, 3), dtype=np.uint8)
        inputs = []

        def capture(points, colors, path, **kwargs):
            inputs.append((points.copy(), kwargs["ids"].copy()))
            return {"marked_ids": sorted(set(kwargs["ids"].tolist()))}

        with patch("eviground.rendering.overview.render", side_effect=capture):
            views = scene_views({"Anchor": ["wall"]}, objects, [objects[0]],
                                xyz, rgb, ids, Path("unused"))
        self.assertEqual(len(inputs[0][0]), 2)
        self.assertEqual(views[0]["hidden_supplemental_instance_ids"], [99])
        np.testing.assert_array_equal(inputs[2][0], xyz)
        self.assertIn(99, inputs[2][1])
        self.assertEqual(views[2]["hidden_supplemental_instance_ids"], [])
        self.assertEqual(views[2]["supplemental_geometry_instance_ids"], [99])
        self.assertEqual(objects, original)


if __name__ == "__main__":
    unittest.main()
