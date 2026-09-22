import unittest

from prepare_facing_anchor_manifest import infer_anchor


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


if __name__ == "__main__":
    unittest.main()
