"""Duplicate masks must not masquerade as separate physical objects."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from eviground.categories.hypotheses import prepare_tables, sha256, load_category_objects
from eviground.geometry.proposals import mask_nms
from eviground.grounding.pipeline import structural_objects
from eviground.geometry.measurements import precision_evidence
from eviground.rendering.renderer import render
from tests.helpers import make_export


class ProposalPolicyTests(unittest.TestCase):
    def test_nms_keeps_highest_confidence_duplicate_and_distinct_instance(self):
        masks = np.asarray([[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1]], dtype=np.uint8)
        keep, groups = mask_nms(np.packbits(masks, axis=1), [.2, .9, .7])
        self.assertEqual(keep, [1, 2])
        self.assertEqual(groups, {1: [1, 0], 2: [2]})

    def test_partial_overlap_does_not_remove_distinct_objects(self):
        masks = np.asarray([[1, 1, 1, 0], [0, 1, 1, 1]], dtype=np.uint8)
        keep, _ = mask_nms(np.packbits(masks, axis=1), [.9, .8])
        self.assertEqual(keep, [0, 1])

    def test_policy_does_not_reintroduce_suppressed_duplicate_as_structure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, scenes, _ = make_export(root)
            (scenes / "protocol.json").write_text("{}")
            masks = scenes / "synthetic/masks.npz"
            masks.write_bytes(b"synthetic predicted-mask source")
            config = json.loads((source / "config.json").read_text())
            policy = root / "policy.json"
            policy.write_text(json.dumps({"version": "mask-nms-proposal-policy-v1",
                "source_protocol_sha256": sha256(scenes / "protocol.json"),
                "scenes": {"synthetic": {"cloud_sha256": config["sources"][0]["cloud_sha256"],
                    "instances_sha256": sha256(scenes / "synthetic/scene_instances.json"),
                    "masks_sha256": sha256(masks),
                    "proposal_scores": {"1": .9, "2": .0001, "3": .2},
                    "kept_ids": [2]}}}))
            prepare_tables(source, scenes, root / "tables", ["synthetic"], 5, proposal_policy=policy)
            table = json.loads((root / "tables/synthetic/objects_top5.json").read_text())
            self.assertEqual([o["bbox_id"] for o in table["objects"]], [2])
            self.assertEqual(table["suppressed_instance_ids"], [1, 3])
            self.assertEqual(table["objects"][0]["proposal_detection_score"], .0001)
            self.assertEqual(load_category_objects(root / "tables/synthetic/objects_top5.json")[0]["proposal_detection_score"], .0001)
            supplemental = structural_objects(scenes / "synthetic", table["objects"], table["suppressed_instance_ids"])
            self.assertNotIn(1, [o["bbox_id"] for o in supplemental])

    def test_duplicate_unmarked_vertex_cannot_erase_focused_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            points = np.asarray([[0., 0., 0.], [0., 0., 0.]])
            rgb = np.asarray([[255, 0, 0], [255, 0, 0]], dtype=np.uint8)
            stats = render(points, rgb, Path(temporary) / "view.png", eye=[0, -2, 1],
                           at=[0, 0, 0], ids=np.asarray([-1, 3]), size=100)
            self.assertEqual(stats["marked_ids"], [3])

    def test_detection_scores_are_exposed_without_changing_category_or_id_pool(self):
        objects = [{"bbox_id": 4, "bbox_3d": [0, 0, 0, 1, 1, 1], "target": "chair",
                    "category_hypotheses": [{"category": "chair", "rank": 1}],
                    "proposal_detection_score": .0001}]
        evidence = precision_evidence("the chair", {"Target": ["chair"], "Anchor": []}, objects, objects)
        self.assertEqual([r["id"] for r in evidence["objects"]], [4])
        self.assertEqual(evidence["objects"][0]["proposal_detection_score"], .0001)

    def test_nearer_unmarked_geometry_still_occludes_focused_instance(self):
        with tempfile.TemporaryDirectory() as temporary:
            points = np.asarray([[0., -1., 0.], [0., 0., 0.]])
            rgb = np.asarray([[255, 0, 0], [0, 0, 255]], dtype=np.uint8)
            stats = render(points, rgb, Path(temporary) / "view.png", eye=[0, -2, 0],
                           at=[0, 0, 0], ids=np.asarray([-1, 3]), size=100)
            self.assertEqual(stats["marked_ids"], [])


if __name__ == "__main__":
    unittest.main()
