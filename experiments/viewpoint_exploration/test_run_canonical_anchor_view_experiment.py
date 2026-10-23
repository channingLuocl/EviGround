import unittest

from run_canonical_anchor_view_experiment import anchor_kind, choose_sides


class CanonicalAnchorViewTest(unittest.TestCase):
    def test_keeps_only_clear_best_side(self):
        scores = [
            {"side": "+x", "free_cell_count": 3},
            {"side": "-x", "free_cell_count": 10},
            {"side": "+y", "free_cell_count": 4},
            {"side": "-y", "free_cell_count": 2},
        ]
        self.assertEqual([item["side"] for item in choose_sides(scores, 0.8, 3)], ["-x"])

    def test_preserves_near_tied_sides(self):
        scores = [{"side": "+x", "free_cell_count": 10}, {"side": "-x", "free_cell_count": 8}]
        self.assertEqual([item["side"] for item in choose_sides(scores, 0.8, 3)], ["+x", "-x"])

    def test_classifies_wall_anchors(self):
        self.assertEqual(anchor_kind("whiteboard"), "wall")
        self.assertEqual(anchor_kind("dining table"), "furniture")


if __name__ == "__main__":
    unittest.main()
