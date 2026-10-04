import unittest

from diagnose_room_floor_cells import classify_cells


class RoomFloorCellsTest(unittest.TestCase):
    def test_wall_ring_keeps_enclosed_floor_and_rejects_outer_floor(self):
        vertices = []
        labels = []
        for x in range(-3, 4):
            for y in range(-3, 4):
                vertices.append((x, y, 0.0))
                labels.append("floor")
        for x in range(-1, 2):
            for y in (-1, 1):
                vertices.append((x, y, 1.0))
                labels.append("wall")
        for y in range(-1, 2):
            for x in (-1, 1):
                vertices.append((x, y, 1.0))
                labels.append("wall")
        result = classify_cells(vertices, labels, 1.0, 0.01, 2.0, 0.01)
        self.assertIn((0, 0), result["interior_cells"])
        self.assertIn((-3, -3), result["outside_floor_cells"])


if __name__ == "__main__":
    unittest.main()
