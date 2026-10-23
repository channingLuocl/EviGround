"""Model-selected scene and isolated-instance inspection cameras."""

import numpy as np
from PIL import Image
from eviground.rendering.renderer import render


def render_action(action, objects, xyz, rgb, ids, out, index):
    known = {obj["bbox_id"]: obj for obj in objects}
    focus = action.get("focus_ids")
    if (
        not isinstance(focus, list)
        or not focus
        or len(focus) > 6
        or any((type(oid) is not int or oid not in known for oid in focus))
        or (len(set(focus)) != len(focus))
    ):
        raise ValueError(
            "render needs 1-6 distinct known focus_ids; include a few candidates and references"
        )
    azimuth, elevation = (action.get("azimuth_deg"), action.get("elevation_deg"))
    if (
        type(azimuth) not in (int, float)
        or not np.isfinite(azimuth)
        or (not 0 <= azimuth < 360)
        or (type(elevation) not in (int, float))
        or (not np.isfinite(elevation))
        or (not 10 <= elevation <= 75)
    ):
        raise ValueError(
            "camera angles must be finite; azimuth [0,360), elevation [10,75]"
        )
    context = action.get("context")
    if context not in ("scene", "isolated"):
        raise ValueError("context must be scene or isolated")
    boxes = np.asarray([known[oid]["bbox_3d"] for oid in focus])
    lower = (boxes[:, :3] - boxes[:, 3:] / 2).min(0)
    upper = (boxes[:, :3] + boxes[:, 3:] / 2).max(0)
    at = (lower + upper) / 2
    radius = max(float(np.linalg.norm(upper - lower)), 0.5) * 1.5
    azimuth_rad, elevation_rad = np.deg2rad([azimuth, elevation])
    direction = np.array(
        [
            np.cos(elevation_rad) * np.cos(azimuth_rad),
            np.cos(elevation_rad) * np.sin(azimuth_rad),
            np.sin(elevation_rad),
        ]
    )
    eye = at + radius * direction
    ceiling_ids = [
        oid for oid, obj in known.items() if obj["target"].casefold() == "ceiling"
    ]
    keep = np.isin(ids, focus) if context == "isolated" else ~np.isin(ids, ceiling_ids)
    cutaway = context == "scene"
    if cutaway:
        covered = {oid for oid, obj in known.items() if "category_hypotheses" in obj}
        keep = np.isin(ids, list(covered | set(focus)))
        from eviground.grounding.bindings import camera_eye

        eye, at = camera_eye(lower, upper, direction)
    if not keep.any():
        raise ValueError("selected focus has no instance points")
    path = out / f"active_view_{index:02d}.png"
    labels = np.where(np.isin(ids, focus), ids, -1)
    panels = []
    if context == "isolated":
        tile_size = 600
        columns = min(3, len(focus))
        rows = (len(focus) + columns - 1) // columns
        canvas = Image.new(
            "RGB", (columns * tile_size, rows * tile_size), (245, 245, 245)
        )
        for offset, oid in enumerate(focus):
            instance = ids == oid
            if not instance.any():
                continue
            box = np.asarray(known[oid]["bbox_3d"])
            local_at = box[:3]
            local_eye = (
                local_at + max(float(np.linalg.norm(box[3:])), 0.5) * 1.5 * direction
            )
            local_path = out / f"active_view_{index:02d}_instance_{oid}.png"
            local = render(
                xyz[instance],
                rgb[instance],
                local_path,
                eye=local_eye,
                at=local_at,
                ids=ids[instance],
                size=tile_size,
                point_radius=4,
            )
            tile_xy = (offset % columns * tile_size, offset // columns * tile_size)
            with Image.open(local_path) as tile:
                canvas.paste(tile, tile_xy)
            panels.append(
                {
                    "id": oid,
                    "image": str(local_path),
                    "tile_xy_px": list(tile_xy),
                    "tile_size_px": tile_size,
                    **local,
                }
            )
        canvas.save(path)
        meta = {
            "points": sum((panel["points"] for panel in panels)),
            "visible_pixels": sum((panel["visible_pixels"] for panel in panels)),
            "marked_ids": [oid for panel in panels for oid in panel["marked_ids"]],
            "point_radius_px": 4,
            "visible_instance_pixels": {
                key: value
                for panel in panels
                for key, value in panel["visible_instance_pixels"].items()
            },
            "panels": panels,
            "views_rendered": len(panels),
            "layout": "one independently framed tile per instance; no shared spatial frame",
        }
    else:
        meta = render(
            xyz[keep],
            rgb[keep],
            path,
            eye=eye,
            at=at,
            ids=labels[keep],
            size=1000,
            point_radius=4,
        )
    visible = set(meta["marked_ids"])
    return {
        "tool": "render",
        "image": str(path),
        "focus_ids": focus,
        "context": context,
        "azimuth_deg": action["azimuth_deg"],
        "elevation_deg": action["elevation_deg"],
        "unmarked_focus_ids": [oid for oid in focus if oid not in visible],
        "ceiling_ids_hidden": ceiling_ids if context == "scene" else [],
        **(
            {
                "hidden_supplemental_instance_ids": sorted(
                    set(known) - covered - set(focus)
                ),
                "context_policy": "classifier-covered cutaway plus requested instances; no GT shell labels",
            }
            if cutaway
            else {}
        ),
        "provenance": "CPU colored point-cloud inspection with depth-tested 4px point disks; not a real RGB crop",
        **meta,
    }
