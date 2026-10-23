"""Render a prepared local scene with SeeGround's upstream PyTorch3D renderer."""
from seeground.paths import ROOT

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(ROOT / 'third_party/seeground'))

from inference.projection import render_point_cloud_with_pytorch3d_with_objects


def select(objects, category):
    category = category.lower()
    return [obj for obj in objects if obj["target"].lower() == category]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--target", default="table")
    parser.add_argument("--anchor", default="chair")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=680)
    args = parser.parse_args()

    cloud = np.load(args.scene_dir / "scene.npz", allow_pickle=False)
    xyz = cloud["xyz"].astype(np.float32)
    rgb = cloud["rgb"].astype(np.float32)
    if rgb.max() > 1:
        rgb /= 255
    scan_pc = np.concatenate((xyz, rgb), axis=1)
    objects = json.loads((args.scene_dir / "objects.json").read_text())
    targets = select(objects, args.target)
    anchors = select(objects, args.anchor)
    if not targets:
        raise ValueError(f"No objects with target={args.target!r}")
    if not anchors:
        anchors = targets

    rendered = render_point_cloud_with_pytorch3d_with_objects(
        objects=objects,
        targets=targets,
        anchors=anchors,
        center=xyz.mean(axis=0),
        scan_pc=scan_pc,
        save_dir=str(args.output),
        image_size=args.image_size,
        draw_id=True,
        draw_img=True,
        device="cuda",
    )
    print(rendered)


if __name__ == "__main__":
    main()
