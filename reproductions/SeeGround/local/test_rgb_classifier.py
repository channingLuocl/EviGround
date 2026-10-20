"""Checks of measured-color rendering, occlusion, invariance and fallback."""
import unittest
import numpy as np
from rgb_classifier import rasterize, render_views, fuse


class RGBTests(unittest.TestCase):
    def test_nearest_point_and_tie(self):
        xy = np.array([[100, 100], [100, 100], [100, 100]])
        rgb = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
        image, fraction = rasterize(xy, np.array([2., 1., 1.]), rgb)
        np.testing.assert_array_equal(image[100, 100], rgb[1])
        self.assertEqual(int(fraction * 224 * 224), 9)
        np.testing.assert_array_equal(image[0, 0], [255, 255, 255])

    def test_translation_scale_and_color_pairing(self):
        rng = np.random.RandomState(123)
        points = rng.uniform(-1, 1, (1000, 3))
        colors = np.tile([17, 83, 209], (len(points), 1))
        a, _ = render_views(points, colors, 42)
        b, _ = render_views(points * 2 + [4., 9., 5.], colors, 42)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (4, 224, 224, 3))
        values = np.unique(a.reshape(-1, 3), axis=0)
        np.testing.assert_array_equal(values, [[17, 83, 209], [255, 255, 255]])

    def test_missing_and_degenerate(self):
        self.assertIsNone(render_views(np.ones((4, 3)), None, 42)[0])
        self.assertIsNone(render_views(np.ones((4, 3)), np.zeros((4, 3)), 42)[0])

    def test_fusion_fallback_and_zero_weight(self):
        base = np.array([[1., 3., 2.], [3., 2., 1.]])
        image = base[:, ::-1]
        valid = np.array([True, False])
        np.testing.assert_array_equal(fuse(base, image, valid, 0).argmax(-1), base.argmax(-1))
        result = fuse(base, image, valid, 1)
        self.assertEqual(result[0].argmax(), image[0].argmax())
        self.assertEqual(result[1].argmax(), base[1].argmax())


if __name__ == '__main__':
    unittest.main()
