"""Depth-tested CPU point-cloud rendering and instance geometry loading."""

import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def project(xyz, eye, at, size):
    forward = np.asarray(at, float) - np.asarray(eye, float)
    norm = np.linalg.norm(forward)
    if norm < 1e-09:
        raise ValueError("相机不能与观察目标重合")
    forward /= norm
    up = np.array([0.0, 0.0, 1.0])
    if abs(forward @ up) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    delta = xyz - np.asarray(eye)
    z = delta @ forward
    safe = np.maximum(z, 1e-06)
    focal = size / (2 * np.tan(np.deg2rad(65) / 2))
    uv = np.column_stack(
        [
            size / 2 + focal * (delta @ right) / safe,
            size / 2 - focal * (delta @ up) / safe,
        ]
    )
    return (uv, z)


def render(xyz, rgb, output, eye=None, at=None, ids=None, size=800, point_radius=0):
    lo, hi = (xyz.min(0), xyz.max(0))
    center = (lo + hi) / 2
    span = max(float(np.max(hi - lo)), 1.0)
    if at is None:
        at = center
    if eye is None:
        eye = center + np.array([0.0, -1.1 * span, 0.8 * span])
    uv, depth = project(xyz, eye, at, size)
    valid = (
        (depth > 0.001)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < size)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < size)
    )
    idx = np.flatnonzero(valid)
    if not len(idx):
        raise ValueError("当前视角没有可见点")
    pix = uv[idx].astype(int)
    if point_radius:
        offsets = np.asarray(
            [
                (x, y)
                for x in range(-point_radius, point_radius + 1)
                for y in range(-point_radius, point_radius + 1)
                if x * x + y * y <= point_radius * point_radius
            ]
        )
        pix = (pix[:, None, :] + offsets).reshape(-1, 2)
        idx = np.repeat(idx, len(offsets))
        inside = (
            (pix[:, 0] >= 0)
            & (pix[:, 0] < size)
            & (pix[:, 1] >= 0)
            & (pix[:, 1] < size)
        )
        pix, idx = (pix[inside], idx[inside])
    flat = pix[:, 1] * size + pix[:, 0]
    order = np.argsort(depth[idx], kind="stable")
    _, first = np.unique(flat[order], return_index=True)
    visible = idx[order[first]]
    image = np.full((size, size, 3), 245, dtype=np.uint8)
    p = pix[order[first]]
    image[p[:, 1], p[:, 0]] = rgb[visible]
    im = Image.fromarray(image)
    drawn = []
    if ids is not None:
        draw = ImageDraw.Draw(im)
        font = ImageFont.load_default(size=23)
        for oid in sorted(set(ids[visible])):
            if oid < 0:
                continue
            positions = p[ids[visible] == oid]
            point = positions[
                np.argmin(np.sum((positions - positions.mean(0)) ** 2, axis=1))
            ]
            x, y = map(int, point)
            text = str(int(oid))
            box = draw.textbbox((0, 0), text, font=font)
            w, h = (box[2] - box[0], box[3] - box[1])
            x = max(3, min(size - w - 7, x - w // 2))
            y = max(3, min(size - h - 7, y - h // 2))
            draw.rectangle((x - 3, y - 3, x + w + 4, y + h + 5), fill="white")
            draw.text((x, y - box[1]), text, fill="red", font=font)
            drawn.append(int(oid))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    im.save(output)
    instance_pixels = (
        {
            str(int(oid)): int(np.sum(ids[visible] == oid))
            for oid in sorted(set(ids[visible]))
            if oid >= 0
        }
        if ids is not None
        else {}
    )
    return {
        "points": len(xyz),
        "visible_pixels": len(visible),
        "marked_ids": drawn,
        "point_radius_px": point_radius,
        "visible_instance_pixels": instance_pixels,
        "eye": np.asarray(eye).tolist(),
        "at": np.asarray(at).tolist(),
    }


def load_objects(path):
    raw = json.loads(Path(path).read_text())
    values = list(raw.values()) if isinstance(raw, dict) else raw
    result = []
    for obj in values:
        box = np.asarray(obj["bbox_3d"], dtype=float)
        if box.shape != (6,) or not np.isfinite(box).all() or (box[3:] <= 0).any():
            raise ValueError("包围框须为中心坐标与三个正尺寸，共六个有限数值")
        result.append(
            {
                "bbox_id": int(obj["bbox_id"]),
                "target": str(obj["target"]),
                "bbox_3d": box.tolist(),
            }
        )
    if not result or len({x["bbox_id"] for x in result}) != len(result):
        raise ValueError("物体表为空或编号重复")
    return result
