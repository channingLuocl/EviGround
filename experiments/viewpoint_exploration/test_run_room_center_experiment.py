import unittest
from run_room_center_experiment import observer_positions, scan_center


class RoomCenterTest(unittest.TestCase):
    def test_center_is_not_sampling_density_mean(self):
        self.assertEqual(scan_center([(0, 0, 0)] * 20 + [(10, 4, 2)]), (5, 2))

    def test_virtual_center_survives_obstacle_but_free_mode_does_not(self):
        obstacles = [(0, 0, 0.1, 0.1)]
        args = ((0, 0), {(0, 0), (2, 0), (8, 0)}, obstacles, 1.0, 0.25, 0.3)
        self.assertEqual(observer_positions('center', *args), [(0, 0)])
        self.assertEqual(observer_positions('central_free', *args), [(0.5, 0)])

    def test_empty_free_region_does_not_fallback_to_virtual_center(self):
        self.assertEqual(observer_positions('central_free', (0, 0), set(), [], 1, .25, .3), [])


if __name__ == '__main__':
    unittest.main()
