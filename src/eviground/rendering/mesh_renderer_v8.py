"""CPU rasterisation of measured ScanNet triangles, without semantic labels."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from eviground.rendering.run import project


def highlight_target(output, mask, bbox_id):
    """Mark only depth-visible target pixels; never draw through occluding objects."""
    output = Path(output)
    Image.fromarray(mask.astype(np.uint8)*255).save(output.with_suffix('.mask.png'))
    if not mask.any():
        return
    with Image.open(output) as image:
        pixels = np.array(image.convert('RGB'))
    Image.fromarray(pixels).save(output.with_suffix('.unmarked.png'))
    inside = np.pad(mask, 1)
    eroded = (inside[1:-1, 1:-1] & inside[:-2, 1:-1] & inside[2:, 1:-1]
              & inside[1:-1, :-2] & inside[1:-1, 2:])
    pixels[mask] = (.8*pixels[mask]+.2*np.array([255, 0, 255])).astype(np.uint8)
    pixels[mask & ~eroded] = [255, 0, 255]
    image = Image.fromarray(pixels)
    positions = np.argwhere(mask)
    y, x = positions[np.argmin(np.sum((positions-positions.mean(0))**2, axis=1))]
    ImageDraw.Draw(image).text((int(x), int(y)), str(bbox_id), fill='red',
                              stroke_width=2, stroke_fill='white',
                              font=ImageFont.load_default(size=23), anchor='mm')
    image.save(output)


def read_scene_faces(scene_root, scene, xyz, rgb):
    """Verify the mesh has the same aligned vertices as the prepared point cloud."""
    from plyfile import PlyData
    directory = scene_root/'downloads'
    path = directory/f'{scene}_vh_clean_2.ply'
    mesh = PlyData.read(path)
    vertices = mesh['vertex']
    raw = np.column_stack([vertices[name] for name in ('x', 'y', 'z')])
    colors = np.column_stack([vertices[name] for name in ('red', 'green', 'blue')])
    line = next(line for line in (directory/f'{scene}.txt').read_text().splitlines()
                if line.startswith('axisAlignment'))
    alignment = np.array([float(value) for value in line.split('=', 1)[1].split()],
                         dtype=np.float32).reshape(4, 4)
    aligned = (np.column_stack([raw, np.ones(len(raw))]) @ alignment.T)[:, :3]
    if aligned.shape != xyz.shape or not np.allclose(aligned, xyz, atol=.002, rtol=0) \
            or not np.array_equal(colors, rgb):
        raise ValueError(f'Mesh vertex order/coordinates do not match scene.npz: {scene}')
    faces = np.stack(mesh['face']['vertex_indices']).astype(np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.min() < 0 or faces.max() >= len(xyz):
        raise ValueError(f'Invalid measured triangle indices: {scene}')
    return faces


def render_mesh(xyz, rgb, faces, output, eye, at, ids=None, size=512, highlight_id=None):
    """Depth-test triangles and interpolate only their measured vertex colours."""
    uv, depth = project(xyz, eye, at, size)
    zbuffer = np.full((size, size), np.inf)
    pixels = np.full((size, size, 3), 245, dtype=np.uint8)
    owners = np.full((size, size), -1, dtype=int)
    for face in faces:
        if np.any(depth[face] <= .001):
            continue
        triangle = uv[face]
        lo = np.maximum(np.floor(triangle.min(0)).astype(int), 0)
        hi = np.minimum(np.ceil(triangle.max(0)).astype(int), size-1)
        if np.any(hi < lo):
            continue
        (ax, ay), (bx, by), (cx, cy) = triangle
        denominator = (by-cy)*(ax-cx)+(cx-bx)*(ay-cy)
        if abs(denominator) < 1e-9:
            continue
        x, y = np.meshgrid(np.arange(lo[0], hi[0]+1)+.5, np.arange(lo[1], hi[1]+1)+.5)
        a = ((by-cy)*(x-cx)+(cx-bx)*(y-cy))/denominator
        b = ((cy-ay)*(x-cx)+(ax-cx)*(y-cy))/denominator
        weights = np.stack([a, b, 1-a-b], axis=-1)
        inside = (weights >= -1e-8).all(axis=-1)
        inverse_depth = (weights/depth[face]).sum(axis=-1)
        distances = 1/np.maximum(inverse_depth, 1e-12)
        region = np.s_[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
        visible = inside & (distances < zbuffer[region])
        if not visible.any():
            continue
        colors = (weights[visible]/depth[face]) @ rgb[face]
        colors /= inverse_depth[visible, None]
        pixels[region][visible] = np.clip(colors, 0, 255).astype(np.uint8)
        zbuffer[region][visible] = distances[visible]
        owners[region][visible] = (int(ids[face[0]]) if ids is not None and
                                   np.all(ids[face] == ids[face[0]]) else -1)
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=23)
    counts = {}
    for oid in sorted(set(owners.ravel()) - {-1}):
        positions = np.argwhere(owners == oid)
        counts[str(oid)] = len(positions)
        y, x = positions[np.argmin(np.sum((positions-positions.mean(0))**2, axis=1))]
        draw.text((int(x), int(y)), str(oid), fill='red', stroke_width=2,
                  stroke_fill='white', font=font, anchor='mm')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    if highlight_id is not None:
        highlight_target(output, owners == highlight_id, highlight_id)
    return {'points': len(xyz), 'triangles': len(faces), 'surface_mode': 'measured_mesh',
            'visible_pixels': int(np.isfinite(zbuffer).sum()),
            'marked_ids': [int(oid) for oid in counts], 'visible_instance_pixels': counts,
            'eye': np.asarray(eye).tolist(), 'at': np.asarray(at).tolist()}
