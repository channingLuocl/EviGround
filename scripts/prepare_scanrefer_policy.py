"""Build a mask-NMS policy without reading GT boxes, labels, or queries."""

import argparse
import json
from pathlib import Path

import numpy as np

from eviground.categories.hypotheses import sha256
from eviground.geometry.proposals import mask_nms


def prepare(scene_root, output):
    policy = {"version": "mask-nms-proposal-policy-v1", "mask_iou_threshold": 0.8,
              "selection": "highest Mask3D proposal confidence; class-blind, no GT",
              "detection_score_definition": "Mask3D class softmax score times mean foreground mask sigmoid; not calibrated probability",
              "source_protocol_sha256": sha256(scene_root / "protocol.json"), "scenes": {}}
    for directory in sorted(scene_root.glob("scene*")):
        if not directory.is_dir():
            continue
        objects = json.loads((directory / "scene_instances.json").read_text())
        marker = json.loads((directory / "PREPARED.json").read_text())
        assert marker["status"] == "passed"
        for name in ("masks.npz", "scene_instances.json"):
            assert sha256(directory / name) == marker["files"][name], (directory.name, name)
        indices = [o["source_mask_index"] for o in objects]
        with np.load(directory / "masks.npz", allow_pickle=False) as masks:
            packed = masks["packed_masks"][:, indices].T.copy()
            scores = masks["scores"][indices]
        keep, groups = mask_nms(packed, scores, threshold=policy["mask_iou_threshold"])
        policy["scenes"][directory.name] = {
            "cloud_sha256": marker["files"]["scene.npz"],
            "instances_sha256": marker["files"]["scene_instances.json"],
            "masks_sha256": marker["files"]["masks.npz"],
            "proposal_scores": {str(o["bbox_id"]): float(scores[i]) for i, o in enumerate(objects)},
            "kept_ids": [objects[i]["bbox_id"] for i in keep],
            "groups": {str(objects[i]["bbox_id"]): [objects[j]["bbox_id"] for j in group]
                       for i, group in groups.items()},
            "original_count": len(objects), "retained_count": len(keep)}
        if len(policy["scenes"]) % 20 == 0:
            print(json.dumps({"processed_scenes": len(policy["scenes"])}), flush=True)
    if output.exists() and json.loads(output.read_text()) != policy:
        raise ValueError("Proposal policy changed; choose a new output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(policy, indent=2) + "\n")
    return {"scenes": len(policy["scenes"]),
            "original_proposals": sum(s["original_count"] for s in policy["scenes"].values()),
            "retained_proposals": sum(s["retained_count"] for s in policy["scenes"].values())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.scene_root, args.output)), flush=True)
