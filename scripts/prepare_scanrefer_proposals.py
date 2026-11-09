"""Run the frozen Mask3D frontend and preserve overlapping proposal geometry."""

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from plyfile import PlyData

from prepare_scanrefer_raw import digest, save


def expand_proposals(xyz, rgb, masks):
    """Duplicate shared vertices so every retained proposal keeps its full mask."""
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 2 or len(masks) != len(xyz) or rgb.shape != xyz.shape:
        raise ValueError("Mask/mesh dimensions differ")
    indices, objects, rejected = [], [], []
    for source_id in range(masks.shape[1]):
        selected = np.flatnonzero(masks[:, source_id])
        points = xyz[selected]
        if len(points) < 3 or np.any(np.ptp(points, axis=0) <= 0):
            rejected.append(source_id)
            continue
        low, high = points.min(0), points.max(0)
        objects.append({"bbox_id": len(objects), "target": "unclassified proposal",
                        "bbox_3d": np.r_[(low + high) / 2, high - low].tolist(),
                        "source_mask_index": source_id, "point_count": len(points)})
        indices.append(selected)
    if not objects:
        raise ValueError("No nondegenerate Mask3D proposals")
    covered = np.zeros(len(xyz), dtype=bool)
    for selected in indices:
        covered[selected] = True
    background = np.flatnonzero(~covered)
    order = np.concatenate([background, *indices])
    ids = np.concatenate([np.full(len(background), -1, dtype=np.int32), *[
        np.full(len(selected), i, dtype=np.int32) for i, selected in enumerate(indices)]])
    return xyz[order], rgb[order], ids, objects, rejected


def read_mesh(directory, scene):
    vertices = PlyData.read(directory / (scene + "_vh_clean_2.ply"))["vertex"]
    xyz = np.column_stack([vertices[k] for k in ("x", "y", "z")])
    rgb = np.column_stack([vertices[k] for k in ("red", "green", "blue")])
    segments = np.asarray(json.loads(
        (directory / (scene + "_vh_clean_2.0.010000.segs.json")).read_text())["segIndices"])
    if len(segments) != len(xyz) or not np.isfinite(xyz).all():
        raise ValueError(f"Invalid mesh/superpoints: {scene}")
    line = next(line for line in (directory / (scene + ".txt")).read_text().splitlines()
                if line.startswith("axisAlignment"))
    matrix = np.asarray(list(map(float, line.split("=", 1)[1].split()))).reshape(4, 4)
    return xyz, rgb, segments, matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--frozen-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.frozen_source / "local"))
    import torch
    if not torch.cuda.is_available():
        parser.error("Mask3D 预处理需要可用的 NVIDIA CUDA GPU")
    from category_common_v82 import run_lock
    from run_mask3d_comparison import load_model, predict
    data = args.data_root
    inputs = json.loads((data / "raw_inputs.json").read_text())
    if not inputs["complete"]:
        raise ValueError("Raw ScanRefer inputs are incomplete")
    assets = data / "models/mask3d"
    asset_record = json.loads((assets / "assets.json").read_text())
    checkpoint = assets / "scannet200_val.ckpt"
    if digest(checkpoint) != asset_record["assets"][checkpoint.name]["sha256"]:
        raise ValueError("Mask3D validation checkpoint changed")
    protocol = {"version": "scanrefer-mask3d-scannet200-r1", "seed": 42,
                "checkpoint_sha256": digest(checkpoint),
                "source_commit": asset_record["mask3d_commit"],
                "model_source_sha256": {
                    str(path.relative_to(assets / "Mask3D")): digest(path)
                    for directory in ("models", "conf/model")
                    for path in sorted((assets / "Mask3D" / directory).rglob("*"))
                    if path.is_file() and path.suffix in (".py", ".yaml")
                    and not path.name.startswith("._")},
                "script_sha256": digest(Path(__file__)),
                "frontend_sha256": digest(args.frozen_source / "local/run_mask3d_comparison.py"),
                "raw_manifest_sha256": digest(data / "raw_inputs.json"),
                "overlap": "shared vertices duplicated; every proposal retains its complete mask",
                "filter": "drop only masks with <3 points or a nonpositive bbox dimension",
                "class_filter": "none; detector categories do not determine r8 categories"}
    torch.set_num_threads(4)
    with run_lock(args.output):
        config = args.output / "protocol.json"
        if config.exists() and json.loads(config.read_text()) != protocol:
            raise ValueError("Proposal protocol changed; choose a new output directory")
        save(config, protocol)
        model, _ = load_model(assets / "Mask3D", checkpoint)
        for number, record in enumerate(inputs["scenes"], 1):
            scene = record["scene"]
            output = args.output / scene
            marker = output / "PREPARED.json"
            if marker.exists():
                for name, expected in json.loads(marker.read_text())["files"].items():
                    if digest(output / name) != expected:
                        raise ValueError(f"Prepared scene changed: {scene}/{name}")
                continue
            directory = data / "raw_scans" / scene
            for item in record["raw_files"]:
                if item["file"].endswith(".aggregation.json"):
                    continue
                if digest(directory / item["file"]) != item["sha256"]:
                    raise ValueError(f"Raw model input changed: {scene}")
            xyz, rgb, segments, matrix = read_mesh(directory, scene)
            seed = int.from_bytes(scene.encode(), "little") % (2**32 - 42) + 42
            torch.manual_seed(seed)
            np.random.seed(seed)
            masks, scores, classes = predict(model, xyz, rgb, segments)
            aligned = (np.column_stack([xyz, np.ones(len(xyz))]) @ matrix.T)[:, :3]
            points, colors, ids, objects, rejected = expand_proposals(aligned, rgb, masks)
            output.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output / "masks.npz", packed_masks=np.packbits(masks, axis=0),
                                vertices=len(xyz), scores=scores, classes=classes)
            np.savez_compressed(output / "scene.npz", xyz=points, rgb=colors, ids=ids)
            save(output / "scene_instances.json", objects)
            # GT is copied only after detection and never used to filter proposals.
            gt = data / "evaluation_gt" / scene / "objects_gt.json"
            if digest(gt) != record["evaluation_gt_sha256"]:
                raise ValueError(f"Evaluation GT changed: {scene}")
            shutil.copyfile(gt, output / "objects_gt.json")
            save(output / "provenance.json", {**protocol, "scene": scene,
                 "proposals": len(objects), "degenerate_mask_indices": rejected,
                 "gt_usage": "evaluation-only; no GT proposal matching or category input"})
            names = ("masks.npz", "scene.npz", "scene_instances.json", "objects_gt.json", "provenance.json")
            save(marker, {"status": "passed", "files": {n: digest(output / n) for n in names}})
            print(json.dumps({"scene": scene, "progress": [number, len(inputs["scenes"])],
                              "proposals": len(objects), "dropped_degenerate": len(rejected)}), flush=True)
            del masks, points, colors, ids
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
