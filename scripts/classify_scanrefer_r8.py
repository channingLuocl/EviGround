"""Apply the locked r8 B/M classifier to predicted ScanRefer instances."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from prepare_scanrefer_raw import digest, save


VERSION = "pointnext-sampling-vote-r8-export-v1"
SCORE_TYPE = "weighted_standardized_3d_rgb_fusion_not_probability"
PASSES = tuple((seed, 0.) for seed in range(42, 49)) + ((42, -10.), (42, 10.))


def standardize(values):
    return ((values - values.mean(-1, keepdim=True)) /
            values.std(-1, keepdim=True, correction=0).clamp_min(1e-6))


def load_heads(root, device):
    families = []
    for family in ("old_rgb_mlp_ce", "old_rgb_mlp_weighted"):
        heads = []
        for seed in (42, 43, 44):
            record = torch.load(root / f"deployment_models/{family}/seed{seed}/model.pt",
                                map_location="cpu", weights_only=True)
            if record["architecture"] != "rgb_mlp":
                raise ValueError("Unexpected historical RGB head architecture")
            state = {key.removeprefix("head."): value for key, value in record["state_dict"].items()}
            classes, width = state["4.weight"].shape
            dimension = state["0.weight"].numel()
            head = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, width),
                                 nn.GELU(), nn.Dropout(.3), nn.Linear(width, classes))
            head.load_state_dict(state, strict=True)
            heads.append(head.to(device).eval())
        families.append(heads)
    return families


@torch.inference_mode()
def rgb_branch(features, families):
    # Exact r4 locked M: standardize each seed, average seeds, average CE/weighted,
    # then standardize the pair. No ScanRefer labels or fitted calibration.
    ensembles = [torch.stack([standardize(head(features)) for head in heads]).mean(0)
                 for heads in families]
    return standardize(.5 * ensembles[0] + .5 * ensembles[1]).cpu().numpy().astype(np.float64)


def fuse(shape_features, text, rgb_scores):
    features = np.asarray(shape_features, dtype=np.float64)
    if features.ndim != 3 or features.shape[1:] != (9, 512):
        raise ValueError("Expected nine frozen 512-dimensional PointNeXt passes")
    logits = 10 * features.mean(1) @ np.asarray(text, dtype=np.float64).T
    standardized = ((logits - logits.mean(-1, keepdims=True)) /
                    np.maximum(logits.std(-1, keepdims=True), 1e-6))
    result = .4 * standardized + .6 * rgb_scores
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite r8 fusion scores")
    return result


def yaw(points, degrees):
    points = points.copy()
    angle = np.deg2rad(degrees)
    x, y = points[:, 0].copy(), points[:, 1].copy()
    points[:, 0] = x * np.cos(angle) - y * np.sin(angle)
    points[:, 1] = x * np.sin(angle) + y * np.cos(angle)
    return points


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--frozen-source", type=Path, required=True)
    parser.add_argument("--cuda-ops", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("原 r8 PointNeXt 分类需要可用的 NVIDIA CUDA GPU")
    sys.path.insert(0, str(args.frozen_source / "local"))
    if args.cuda_ops:
        sys.path.insert(0, str(args.cuda_ops))
    from category_common_v82 import instance_seed, load_cloud, run_lock
    from pointcloud_classifier_v82 import load_pointnext, prepare_points
    from rgb_classifier import RENDER, load_clip, render_views, verify_text

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model_root = args.data_root / "models/r8"
    model_record = json.loads((model_root / "MODEL_SOURCES.json").read_text())
    for item in model_record["files"]:
        if digest(model_root / item["file"]) != item["sha256"]:
            raise ValueError(f"Historical model changed: {item['file']}")
    selected = json.loads((model_root / "r8_selection.json").read_text())["locked"]["B"]
    if (selected["name"], selected["rgb_weight"], selected["passes"]) != ("sample_yaw9", .6, 9):
        raise ValueError("r8 B/M locked selection changed")
    text = np.load(model_root / "text.npy", allow_pickle=False)
    vocabulary = json.loads((model_root / "vocabulary.json").read_text())
    if text.shape != (len(vocabulary), 512):
        raise ValueError("Text vocabulary shape mismatch")
    raw = json.loads((args.data_root / "raw_inputs.json").read_text())
    if not raw["complete"]:
        raise ValueError("Raw ScanRefer inputs incomplete")
    scenes = [row["scene"] for row in raw["scenes"]]
    sources = [{"scene": scene, "cloud_sha256": digest(args.scene_root / scene / "scene.npz")}
               for scene in scenes]
    config = {"version": VERSION, "method": "r8 locked B/M on Mask3D ScanNet200 proposals",
              "selection": {key: selected[key] for key in ("name", "rgb_weight", "passes")},
              "score_type": SCORE_TYPE, "sources": sources,
              "scene_tables_sha256": {scene: digest(args.scene_root / scene / "scene_instances.json")
                                       for scene in scenes},
              "score_matrix": {"path": "all_scores.npy", "rows": "predictions.json results order",
                               "columns": "vocabulary.json order"},
              "category_gating": "not applied; rank-only Top-5",
              "model_files": {item["file"]: item["sha256"] for item in model_record["files"]},
              "script_sha256": digest(Path(__file__)),
              "frozen_source_sha256": {name: digest(args.frozen_source / "local" / name)
                                       for name in ("pointcloud_classifier_v82.py", "rgb_classifier.py")},
              "proposal_protocol_sha256": digest(args.scene_root / "protocol.json")}
    with run_lock(args.output):
        path = args.output / "config.json"
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError("Classifier inputs changed; choose a new output directory")
        save(path, config)
        encoder, _ = load_pointnext(model_root / "pnext_cls.pth")
        encoder = encoder.cuda().eval()
        clip, tokenizer = load_clip(model_root / "clip_vit_b16.bin", "cuda")
        alignment = verify_text(clip, tokenizer, vocabulary, torch.from_numpy(text), "cuda")
        save(args.output / "text_alignment.json", alignment)
        heads = load_heads(model_root, "cuda")
        mean = torch.tensor([.48145466, .4578275, .40821073], device="cuda")[None, :, None, None]
        std = torch.tensor([.26862954, .26130258, .27577711], device="cuda")[None, :, None, None]
        all_rows, all_scores = [], []
        for number, scene in enumerate(scenes, 1):
            directory = args.scene_root / scene
            shard = args.output / "scenes" / scene
            marker = shard / "COMPLETE.json"
            if not marker.exists():
                xyz, rgb, ids = load_cloud(directory / "scene.npz")
                objects = json.loads((directory / "scene_instances.json").read_text())
                if set(map(int, np.unique(ids))) - {-1} != {obj["bbox_id"] for obj in objects}:
                    raise ValueError(f"Proposal table and cloud IDs differ: {scene}")
                rows, scores = [], []
                for obj in objects:
                    oid = obj["bbox_id"]
                    mask = ids == oid
                    points = xyz[mask]
                    low, high = points.min(0), points.max(0)
                    if not np.allclose(np.r_[(low + high) / 2, high - low],
                                       obj["bbox_3d"], atol=1e-7, rtol=0):
                        raise ValueError(f"Proposal box and point cloud differ: {scene}/{oid}")
                    samples = [yaw(prepare_points(points, None, "pointnext",
                                                  instance_seed(seed, scene, oid)), degrees)
                               for seed, degrees in PASSES]
                    features = F.normalize(encoder(torch.from_numpy(np.stack(samples)).cuda()).float(),
                                           dim=-1).cpu().numpy()
                    images, coverage = render_views(points, rgb[mask], instance_seed(RENDER["seed"], scene, oid))
                    if images is None or min(coverage) <= 0:
                        raise ValueError(f"Invalid RGB views: {scene}/{oid}")
                    tensor = torch.from_numpy(images).permute(0, 3, 1, 2).cuda().float() / 255
                    image_features = clip.encode_image((tensor - mean) / std, normalize=True)
                    merged = F.normalize(image_features.mean(0, keepdim=True), dim=-1)
                    score = fuse(features[None], text, rgb_branch(merged, heads))[0]
                    top = np.argsort(-score, kind="stable")[:5]
                    rows.append({"scene": scene, "bbox_id": oid, "status": "completed",
                                 "predicted_object": vocabulary[int(top[0])],
                                 "top5": [{"category": vocabulary[int(i)], "score": float(score[i]),
                                           "rank": rank} for rank, i in enumerate(top, 1)],
                                 "bbox_3d": obj["bbox_3d"], "point_count": int(mask.sum()),
                                 "score_type": SCORE_TYPE})
                    scores.append(score)
                shard.mkdir(parents=True, exist_ok=True)
                save(shard / "predictions.json", rows)
                np.save(shard / "scores.npy", np.stack(scores))
                save(marker, {"status": "passed", "files": {
                    name: digest(shard / name) for name in ("predictions.json", "scores.npy")}})
            for name, expected in json.loads(marker.read_text())["files"].items():
                if digest(shard / name) != expected:
                    raise ValueError(f"Classifier shard changed: {scene}/{name}")
            all_rows.extend(json.loads((shard / "predictions.json").read_text()))
            all_scores.append(np.load(shard / "scores.npy", allow_pickle=False))
            print(json.dumps({"scene": scene, "progress": [number, len(scenes)]}), flush=True)
        matrix = np.concatenate(all_scores)
        if matrix.shape != (len(all_rows), len(vocabulary)):
            raise ValueError("Score matrix and prediction rows differ")
        save(args.output / "predictions.json", {"version": VERSION, "results": all_rows,
             "summary": {"complete": True, "errors": 0, "count": len(all_rows),
                         "scenes": len(scenes), "score_type": SCORE_TYPE}})
        save(args.output / "vocabulary.json", vocabulary)
        np.save(args.output / "all_scores.npy", matrix)
        save(args.output / "EXPORT_VERIFIED.json", {"status": "passed", "files": {
            name: digest(args.output / name) for name in
            ("predictions.json", "config.json", "vocabulary.json", "all_scores.npy")}})
        from categories_r8 import load_export
        try:
            verified, _, _ = load_export(args.output)
            if len(verified) != len(all_rows):
                raise ValueError("Frozen r18 export adapter rejected object coverage")
        except Exception:
            (args.output / "EXPORT_VERIFIED.json").unlink()
            raise


if __name__ == "__main__":
    main()
