"""Run one SeeGround-style API query with the upstream GPU renderer."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from eviground.paths import ROOT
sys.path.insert(0, str(ROOT / 'third_party/seeground'))

from inference.projection import render_point_cloud_with_pytorch3d_with_objects
from eviground.rendering.run import ground


def select(objects, category):
    category = category.lower()
    return [obj for obj in objects if obj["target"].lower() == category]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--query", required=True)
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
    objects_path = args.scene_dir / "objects.json"
    objects = json.loads(objects_path.read_text())
    targets = select(objects, args.target)
    anchors = select(objects, args.anchor)
    if not targets:
        raise ValueError(f"No objects with target={args.target!r}")
    if not anchors:
        anchors = targets

    image_path = render_point_cloud_with_pytorch3d_with_objects(
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
    ground(
        SimpleNamespace(
            image=image_path,
            objects=str(objects_path),
            query=args.query,
            dry_run=False,
            output=str(args.output / "prediction.json"),
        )
    )


if __name__ == "__main__":
    main()
