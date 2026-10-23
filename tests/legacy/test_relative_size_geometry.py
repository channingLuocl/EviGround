import itertools
import unittest

import numpy as np

from eviground.geometry.object_geometry import point_cloud_measurements, SIZE_GEOMETRY_POLICY


def obj(oid=1, category='bookshelf'):
    return {'bbox_id': oid, 'target': category, 'bbox_3d': [0, 0, 1, 2, .4, 2]}


def corners(width, depth, height):
    return np.asarray(list(itertools.product((0., width), (0., depth), (0., height))))


def rotate(points, angle):
    cosine, sine = np.cos(angle), np.sin(angle)
    result = points.copy()
    result[:, :2] = result[:, :2] @ np.array([[cosine, -sine], [sine, cosine]])
    return result


def measure(points, category='bookshelf', oid=1):
    return point_cloud_measurements([obj(oid, category)], points, np.full(len(points), oid))[0]


class RelativeSizeGeometryTests(unittest.TestCase):
    def test_horizontal_dimensions_and_sorting_are_rotation_stable(self):
        for angle in (0., .21, .67, 1.39, 2.52):
            long_low = measure(rotate(corners(3, .4, 1), angle))
            short_high = measure(rotate(corners(1.5, .6, 2), angle))
            self.assertAlmostEqual(long_low['horizontal_oriented_long_extent_m'], 3.)
            self.assertAlmostEqual(long_low['oriented_footprint_m2'], 1.2)
            self.assertGreater(long_low['horizontal_oriented_long_extent_m'],
                               short_high['horizontal_oriented_long_extent_m'])
            self.assertLess(long_low['height_m'], short_high['height_m'])
            self.assertEqual(long_low['metric_reliability']['height_m']['status'], 'usable')

    def test_vertical_face_area_ignores_rotation_and_depth_proxy(self):
        for angle in (0., .37, 1.18, 2.51):
            door = measure(rotate(corners(1, .04, 2), angle), 'door')
            self.assertAlmostEqual(door['vertical_face_area_m2'], 2.)
            self.assertAlmostEqual(door['vertical_face_width_m'], 1.)
            self.assertAlmostEqual(door['vertical_face_height_m'], 2.)
            self.assertEqual(door['metric_reliability']['vertical_face_area_m2']['status'], 'usable')
            self.assertEqual(door['metric_reliability']['vertical_face_area_m2']['units'], 'm2')

    def test_unsupported_structure_and_thick_or_horizontal_faces_are_unknown(self):
        for category, points in [('box', corners(1, .04, 2)),
                                 ('window', corners(1, 1, 2)),
                                 ('door', corners(2, 2, .03))]:
            row = measure(points, category)
            self.assertIsNone(row['vertical_face_area_m2'])
            self.assertEqual(row['metric_reliability']['vertical_face_area_m2']['status'], 'unknown')

    def test_missing_nonfinite_and_degenerate_instance_support_are_unknown(self):
        missing = point_cloud_measurements([obj()], None, None)[0]
        self.assertEqual(missing['status'], 'unknown')
        invalid = measure(np.array([[0., 0., 0.], [1., np.nan, 1.]]))
        self.assertEqual(invalid['status'], 'unknown')
        collinear = measure(np.array([[0., 0., 0.], [1., 0., 1.], [2., 0., 2.]]))
        self.assertIsNone(collinear['horizontal_oriented_long_extent_m'])
        self.assertEqual(collinear['metric_reliability']['horizontal_oriented_long_extent_m']['status'], 'unknown')

    def test_more_points_do_not_prove_complete_or_stable_bounds(self):
        # A densely sampled triangular fragment cannot establish a rectangular face.
        triangle = np.array([[x, 0., z] for x in np.linspace(0., 1., 30)
                             for z in np.linspace(0., 2. * (1. - x), 30)])
        fragment = measure(triangle, 'window')
        self.assertIsNone(fragment['vertical_face_area_m2'])
        # One distant mislabeled point cannot certify an object's full height.
        rectangle = np.array([[x, y, z] for x in np.linspace(0., 1., 10)
                              for y in (0., .2) for z in np.linspace(0., 1., 10)])
        unstable = measure(np.vstack((rectangle, [0., 0., 10.])))
        self.assertEqual(unstable['metric_reliability']['height_m']['status'], 'unknown')
        self.assertIn('isolated extrema', unstable['metric_reliability']['height_m']['reasons'][0])
        self.assertIn('point count is not completeness evidence',
                      unstable['geometry_details']['scan_completeness'])

    def test_foreign_instances_do_not_add_to_desk_height(self):
        desk = corners(1.5, .8, .75)
        desktop_object = corners(.3, .2, 1.) + [0., 0., .75]
        row = point_cloud_measurements([obj(1, 'desk')], np.vstack((desk, desktop_object)),
                                     np.array([1] * len(desk) + [2] * len(desktop_object)))[0]
        self.assertAlmostEqual(row['height_m'], .75)
        self.assertIn('not a fitted desk tabletop', row['geometry_details']['height_semantics'])

    def test_unknown_depth_does_not_invalidate_supported_length(self):
        body = np.repeat(corners(3., .4, 1.), 50, axis=0)
        row = measure(np.vstack((body, [1.5, .65, .5])))
        self.assertAlmostEqual(row['horizontal_oriented_long_extent_m'], 3.)
        self.assertEqual(row['metric_reliability']['horizontal_oriented_long_extent_m']['status'], 'usable')
        self.assertIsNone(row['oriented_footprint_m2'])
        self.assertEqual(row['metric_reliability']['horizontal_oriented_short_extent_m']['status'], 'unknown')

    def test_ids_order_and_point_density_do_not_define_dimensions(self):
        points = rotate(corners(2, .4, 1.5), .41)
        original = measure(points)
        permuted = measure(points[::-1], oid=97)
        duplicated = measure(np.repeat(points, 20, axis=0))
        for metric in ('height_m', 'horizontal_oriented_long_extent_m',
                       'oriented_volume_m3', 'oriented_footprint_m2'):
            self.assertAlmostEqual(original[metric], permuted[metric])
            self.assertAlmostEqual(original[metric], duplicated[metric])
        self.assertEqual(original['geometry_details']['policy'], SIZE_GEOMETRY_POLICY)


if __name__ == '__main__':
    unittest.main()
