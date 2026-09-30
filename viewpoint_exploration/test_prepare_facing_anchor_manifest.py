import unittest

from prepare_facing_anchor_manifest import infer_anchor, is_pure_direction


class FacingAnchorManifestTest(unittest.TestCase):
    def test_unique_anchor_after_facing_cue(self):
        objects = [{"id": 7, "label": "door"}, {"id": 9, "label": "bed"}]
        self.assertEqual(infer_anchor("When facing the door, the couch to the right.", objects), 7)

    def test_rejects_nonunique_anchor_label(self):
        objects = [{"id": 7, "label": "door"}, {"id": 8, "label": "door"}]
        self.assertIsNone(infer_anchor("When facing the door, the couch to the right.", objects))

    def test_rejects_multiple_matching_anchors(self):
        objects = [{"id": 7, "label": "door"}, {"id": 8, "label": "bed"}]
        self.assertIsNone(infer_anchor("Facing the door beside the bed, choose the lamp on the left.", objects))

    def test_rejects_compositional_direction_query(self):
        self.assertFalse(is_pure_direction("Facing the bed, choose the pillow closest to the headboard on the right."))

    def test_keeps_plain_left_right_query(self):
        self.assertTrue(is_pure_direction("When facing the door, the couch to the right."))

    def test_rejects_second_spatial_reference(self):
        self.assertFalse(is_pure_direction("Facing the doors, the backpack on the left side of the ottoman."))

    def test_rejects_far_and_group_count(self):
        self.assertFalse(is_pure_direction("Facing the bed, choose the far right pillow from the set of two."))


if __name__ == "__main__":
    unittest.main()
