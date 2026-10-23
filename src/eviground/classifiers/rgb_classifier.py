"""Frozen CLIP on four fixed, label-free RGB point-cloud views."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from eviground.categories.category_common_v82 import file_hash, instance_seed, load_cloud
from eviground.evaluation.run_storage import write_json


RENDER = {'azimuth_degrees': [0, 90, 180, 270], 'elevation_degrees': 20,
          'resolution': 224, 'background': 255, 'radius_pixels': 1,
          'max_points': 32768, 'camera_distance_radii': 3, 'fov_degrees': 60,
          'seed': 20261019, 'centering': 'observed_bbox_center', 'up': 'z'}


def rasterize(xy, depth, rgb, resolution=224):
    """Nearest measured point wins; equal depths use first source point."""
    pixels, distances, sources = [], [], []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            x, y = xy[:, 0] + dx, xy[:, 1] + dy
            valid = (depth > 0) & (x >= 0) & (x < resolution) & (y >= 0) & (y < resolution)
            pixels.append(y[valid] * resolution + x[valid])
            distances.append(depth[valid])
            sources.append(np.flatnonzero(valid))
    pixels, distances, sources = map(np.concatenate, (pixels, distances, sources))
    zbuffer = np.full(resolution * resolution, np.inf)
    np.minimum.at(zbuffer, pixels, distances)
    nearest = distances == zbuffer[pixels]
    owner = np.full(resolution * resolution, len(rgb), dtype=np.int64)
    np.minimum.at(owner, pixels[nearest], sources[nearest])
    covered = owner < len(rgb)
    image = np.full((resolution * resolution, 3), 255, dtype=np.uint8)
    image[covered] = np.rint(rgb[owner[covered]]).astype(np.uint8)
    return image.reshape(resolution, resolution, 3), float(covered.mean())


def render_views(xyz, rgb, seed):
    if rgb is None or len(xyz) < 3:
        return None, [0.] * 4
    xyz = np.asarray(xyz, dtype=np.float64)
    if not np.isfinite(xyz).all() or rgb.shape != xyz.shape:
        raise ValueError('Invalid paired XYZ/RGB')
    center = (xyz.min(0) + xyz.max(0)) / 2
    radius = np.linalg.norm(xyz - center, axis=1).max()
    if radius <= 1e-10:
        return None, [0.] * 4
    if len(xyz) > RENDER['max_points']:
        indices = np.sort(np.random.RandomState(seed).choice(len(xyz), RENDER['max_points'], replace=False))
        xyz, rgb = xyz[indices], rgb[indices]
    points = (xyz - center) / radius
    images, coverage = [], []
    elevation = np.deg2rad(RENDER['elevation_degrees'])
    size = RENDER['resolution']
    focal = (size / 2) / np.tan(np.deg2rad(RENDER['fov_degrees']) / 2)
    for azimuth in np.deg2rad(RENDER['azimuth_degrees']):
        direction = np.array([np.cos(elevation)*np.cos(azimuth), np.cos(elevation)*np.sin(azimuth), np.sin(elevation)])
        right = np.cross([0., 0., 1.], direction)
        right /= np.linalg.norm(right)
        up = np.cross(direction, right)
        depth = RENDER['camera_distance_radii'] - points @ direction
        xy = np.stack([focal * (points @ right) / depth + size/2,
                       -focal * (points @ up) / depth + size/2], axis=1)
        image, fraction = rasterize(np.rint(xy).astype(np.int64), depth, rgb, size)
        images.append(image)
        coverage.append(fraction)
    return np.stack(images), coverage


def load_clip(path, device):
    """Convert the cached HF OpenAI checkpoint, strictly, without downloads."""
    import open_clip
    source = torch.load(path, map_location='cpu', weights_only=True)
    model = open_clip.create_model('ViT-B-16', pretrained=None, force_quick_gelu=True)
    target, used = {}, set()

    def copy(dst, src, transpose=False):
        value = source[src]
        target[dst] = value.T if transpose else value
        used.add(src)

    for dst, src, transpose in (
        ('visual.class_embedding', 'vision_model.embeddings.class_embedding', False),
        ('visual.positional_embedding', 'vision_model.embeddings.position_embedding.weight', False),
        ('visual.conv1.weight', 'vision_model.embeddings.patch_embedding.weight', False),
        ('visual.proj', 'visual_projection.weight', True),
        ('token_embedding.weight', 'text_model.embeddings.token_embedding.weight', False),
        ('positional_embedding', 'text_model.embeddings.position_embedding.weight', False),
        ('text_projection', 'text_projection.weight', True), ('logit_scale', 'logit_scale', False)):
        copy(dst, src, transpose)
    for dst, src in (('visual.ln_pre', 'vision_model.pre_layrnorm'),
                     ('visual.ln_post', 'vision_model.post_layernorm'), ('ln_final', 'text_model.final_layer_norm')):
        for suffix in ('weight', 'bias'):
            copy(dst+'.'+suffix, src+'.'+suffix)
    for tower, prefix in (('visual.transformer', 'vision_model'), ('transformer', 'text_model')):
        for layer in range(12):
            dst, src = f'{tower}.resblocks.{layer}', f'{prefix}.encoder.layers.{layer}'
            for out_name, in_name in (('ln_1', 'layer_norm1'), ('ln_2', 'layer_norm2'),
                                      ('mlp.c_fc', 'mlp.fc1'), ('mlp.c_proj', 'mlp.fc2'),
                                      ('attn.out_proj', 'self_attn.out_proj')):
                for suffix in ('weight', 'bias'):
                    copy(f'{dst}.{out_name}.{suffix}', f'{src}.{in_name}.{suffix}')
            for suffix in ('weight', 'bias'):
                names = [f'{src}.self_attn.{q}_proj.{suffix}' for q in ('q', 'k', 'v')]
                target[f'{dst}.attn.in_proj_{suffix}'] = torch.cat([source[n] for n in names])
                used.update(names)
    extra = set(source) - used
    # Older HF releases persist deterministic position indices as buffers.
    if extra - {'text_model.embeddings.position_ids', 'vision_model.embeddings.position_ids'}:
        raise ValueError(f'Unmapped HF parameters: {sorted(extra)}')
    model.load_state_dict(target, strict=True)
    model.requires_grad_(False).eval().to(device)
    return model, open_clip.get_tokenizer('ViT-B-16')


def verify_text(model, tokenizer, vocabulary, reference, device):
    with torch.inference_mode():
        encoded = model.encode_text(tokenizer([f'a {c} in a scene' for c in vocabulary]).to(device), normalize=True).cpu()
    reference = F.normalize(reference.cpu().float(), dim=-1)
    cosine = (encoded * reference).sum(-1)
    difference = (encoded - reference).abs().max().item()
    if cosine.min() < .99999 or difference > .0001:
        raise ValueError(f'CLIP text space mismatch: cosine={cosine.min().item()}, max_abs={difference}')
    return {'minimum_cosine': cosine.min().item(), 'maximum_absolute_difference': difference,
            'prompt': 'a {} in a scene', 'checkpoint_sha256': None}


def extract(model, manifest, output, device):
    """Scene shards permit restart; every shard is bound to immutable inputs."""
    by_scene = defaultdict(list)
    for index, row in enumerate(manifest['samples']):
        by_scene[row['scene']].append(index)
    sources = {r['scene']: r['cloud_sha256'] for r in manifest['sources']}
    features = np.zeros((len(manifest['samples']), 512), dtype=np.float32)
    coverage = np.zeros((len(features), 4), dtype=np.float32)
    valid = np.zeros(len(features), dtype=bool)
    mean = torch.tensor([.48145466, .4578275, .40821073], device=device)[None, :, None, None]
    std = torch.tensor([.26862954, .26130258, .27577711], device=device)[None, :, None, None]
    for number, (scene, indices) in enumerate(sorted(by_scene.items()), 1):
        path = output / 'scenes' / f'{scene}.npz'
        identities = [manifest['samples'][i]['bbox_id'] for i in indices]
        if path.exists():
            with np.load(path, allow_pickle=False) as archive:
                if archive['ids'].tolist() != identities or str(archive['cloud_sha256']) != sources[scene]:
                    raise ValueError('RGB shard identity mismatch')
                features[indices], coverage[indices], valid[indices] = archive['features'], archive['coverage'], archive['valid']
            continue
        row = manifest['samples'][indices[0]]
        if file_hash(row['cloud']) != sources[scene]:
            raise ValueError(f'Cloud changed: {scene}')
        xyz, rgb, ids = load_cloud(row['cloud'])
        order = np.argsort(ids, kind='stable')
        unique, start, count = np.unique(ids[order], return_index=True, return_counts=True)
        lookup = {int(oid): order[s:s+n] for oid, s, n in zip(unique, start, count)}
        for chunk in range(0, len(indices), 8):
            pending, images = [], []
            for index in indices[chunk:chunk+8]:
                row = manifest['samples'][index]
                selected = lookup[row['bbox_id']]
                views, fraction = render_views(xyz[selected], rgb[selected] if rgb is not None else None,
                    instance_seed(RENDER['seed'], scene, row['bbox_id']))
                coverage[index] = fraction
                if views is not None and min(fraction) > 0:
                    pending.append(index)
                    images.append(views)
            if images:
                tensor = torch.from_numpy(np.concatenate(images)).permute(0, 3, 1, 2).to(device).float() / 255
                with torch.inference_mode():
                    encoded = model.encode_image((tensor - mean) / std, normalize=True)
                    encoded = F.normalize(encoded.reshape(len(pending), 4, 512).mean(1), dim=-1)
                features[pending] = encoded.cpu().numpy()
                valid[pending] = True
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix('.tmp').open('wb') as handle:
            np.savez_compressed(handle, features=features[indices], coverage=coverage[indices], valid=valid[indices],
                                ids=np.array(identities), cloud_sha256=sources[scene])
        path.with_suffix('.tmp').replace(path)
        print(json.dumps({'stage': 'RGB', 'split': manifest['split'], 'scenes': number, 'total': len(by_scene),
                          'objects_completed': sum(len(v) for k, v in sorted(by_scene.items())[:number])}), flush=True)
    if not np.isfinite(features).all() or (np.linalg.norm(features[valid], axis=1) <= 0).any():
        raise ValueError('Invalid RGB features')
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'features.npz', features=features, coverage=coverage, valid=valid)
    return features, valid


def standardize(scores):
    return (scores - scores.mean(-1, keepdims=True)) / np.maximum(scores.std(-1, keepdims=True), 1e-6)


def fuse(base, image, valid, weight, size=None, coefficient=0.):
    scores = (1-weight) * standardize(base) + weight * standardize(image)
    scores[~valid] = standardize(base)[~valid]
    if size is not None:
        scores = scores + coefficient * size
    return scores
