import unittest

from explore_horizontal_viewpoints import choose_candidate, explore


OBJECTS = [
    {"id": 1, "label": "table", "center": [0, 0, 0]},
    {"id": 2, "label": "chair", "center": [1, 0, 0]},
    {"id": 3, "label": "chair", "center": [-1, 0, 0]},
]


class ViewpointExplorerTest(unittest.TestCase):
    def test_right_and_left_follow_heading(self):
        anchor = OBJECTS[0]
        self.assertEqual(choose_candidate(OBJECTS[1:], anchor, "right", 90)["id"], 2)
        self.assertEqual(choose_candidate(OBJECTS[1:], anchor, "left", 90)["id"], 3)

    def test_opposite_heading_reverses_result(self):
        anchor = OBJECTS[0]
        self.assertEqual(choose_candidate(OBJECTS[1:], anchor, "right", 90)["id"], 2)
        self.assertEqual(choose_candidate(OBJECTS[1:], anchor, "right", 270)["id"], 3)

    def test_multiple_winners_are_marked_unstable(self):
        result = explore(OBJECTS, "chair", 1, "right", [90, 270])
        self.assertFalse(result["is_stable"])
        self.assertEqual(result["distinct_winners"], 2)


if __name__ == "__main__":
    unittest.main()
