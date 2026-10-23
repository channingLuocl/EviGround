"""Retain the four original CLIP view embeddings on official training rooms."""
from seeground.paths import source_file
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from seeground.categories.category_common_v82 import file_hash, instance_seed, load_cloud, run_lock
from seeground.classifiers.rgb_classifier import RENDER, load_clip, render_views, verify_text
from seeground.evaluation.run_storage import write_json


def main(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = args.output
    old = args.reference
    manifest_path = old / 'manifests/train.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['split'] == 'train' and all(s['label'] >= 0 for s in manifest['samples'])
    expected = np.load(old / 'features/rgb_train/features.npz', allow_pickle=False)
    assert expected['valid'].all()
    config = {'version': 'pair-views-train-r1', 'manifest_sha256': file_hash(manifest_path),
              'old_RGB_sha256': file_hash(old / 'features/rgb_train/features.npz'),
              'clip_sha256': file_hash(args.clip), 'renderer': RENDER,
              'source_sha256': {n: file_hash(source_file(n)) for n in
                  ('cache_pair_views.py', 'rgb_classifier.py', 'category_common_v82.py')},
              'mean_bridge_max_absolute_tolerance': 1e-5}
    old_config = json.loads((old / 'features/config.json').read_text())
    assert config['clip_sha256'] == old_config['clip_sha256'] and RENDER == old_config['renderer']
    with run_lock(root):
        path = root / 'config.json'
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError('Cache inputs or code changed; use a fresh output')
        write_json(path, config)
        model, tokenizer = load_clip(args.clip, 'cuda')
        text = torch.from_numpy(np.load(old / 'features/text.npy'))
        write_json(root / 'text_alignment.json', verify_text(model, tokenizer, manifest['vocabulary'], text, 'cuda'))
        scene_indices = {}
        for i, sample in enumerate(manifest['samples']):
            scene_indices.setdefault(sample['scene'], []).append(i)
        sources = {s['scene']: s['cloud_sha256'] for s in manifest['sources']}
        features = np.zeros((len(manifest['samples']), 4, 512), dtype=np.float32)
        coverage = np.zeros((len(features), 4), dtype=np.float32)
        mean = torch.tensor([.48145466, .4578275, .40821073], device='cuda')[None, :, None, None]
        std = torch.tensor([.26862954, .26130258, .27577711], device='cuda')[None, :, None, None]
        maximum_difference, completed = 0., 0
        for number, (scene, indices) in enumerate(sorted(scene_indices.items()), 1):
            target = root / 'scenes' / f'{scene}.npz'
            ids = [manifest['samples'][i]['bbox_id'] for i in indices]
            if target.exists():
                data = np.load(target, allow_pickle=False)
                assert data['ids'].tolist() == ids and str(data['cloud_sha256']) == sources[scene]
                views, fractions = data['features'], data['coverage']
                difference = float(data['mean_max_absolute_difference'])
            else:
                cloud_path = manifest['samples'][indices[0]]['cloud']
                assert file_hash(cloud_path) == sources[scene]
                xyz, rgb, instance_ids = load_cloud(cloud_path)
                views = np.zeros((len(indices), 4, 512), dtype=np.float32)
                fractions = np.zeros((len(indices), 4), dtype=np.float32)
                difference = 0.
                for start in range(0, len(indices), 8):
                    images = []
                    for offset, i in enumerate(indices[start:start+8], start):
                        row = manifest['samples'][i]
                        mask = instance_ids == row['bbox_id']
                        image, fraction = render_views(xyz[mask], rgb[mask],
                            instance_seed(RENDER['seed'], scene, row['bbox_id']))
                        if image is None or min(fraction) <= 0:
                            raise ValueError('Unexpected missing RGB view')
                        images.append(image)
                        fractions[offset] = fraction
                    tensor = torch.from_numpy(np.concatenate(images)).permute(0, 3, 1, 2).cuda().float() / 255
                    with torch.inference_mode():
                        encoded = model.encode_image((tensor-mean)/std, normalize=True).reshape(-1, 4, 512)
                        merged = F.normalize(encoded.mean(1), dim=-1).cpu().numpy()
                    reference = expected['features'][indices[start:start+8]]
                    current = float(np.max(np.abs(merged-reference)))
                    if current > config['mean_bridge_max_absolute_tolerance']:
                        raise ValueError(f'Historical mean feature bridge failed: {scene}, {current}')
                    difference = max(difference, current)
                    views[start:start+len(images)] = encoded.cpu().numpy()
                target.parent.mkdir(exist_ok=True)
                with target.with_suffix('.tmp').open('wb') as handle:
                    np.savez_compressed(handle, features=views, coverage=fractions,
                        ids=np.array(ids), cloud_sha256=sources[scene], mean_max_absolute_difference=difference)
                target.with_suffix('.tmp').replace(target)
            features[indices], coverage[indices] = views, fractions
            completed += len(indices)
            maximum_difference = max(maximum_difference, difference)
            if number % 10 == 0 or number == 1:
                print(json.dumps({'stage': 'four_views', 'scenes': number, 'total_scenes': len(scene_indices),
                                  'objects_completed': completed,
                                  'maximum_mean_difference': maximum_difference}), flush=True)
        with (root / 'features.tmp').open('wb') as handle:
            np.savez_compressed(handle, features=features, coverage=coverage)
        (root / 'features.tmp').replace(root / 'features.npz')
        write_json(root / 'COMPLETE.json', {'objects': len(features), 'views': 4,
            'features_sha256': file_hash(root / 'features.npz'), 'config_sha256': file_hash(root / 'config.json'),
            'maximum_mean_difference': maximum_difference})
        print(json.dumps({'stage': 'cache_complete', 'objects': len(features), 'max_abs': maximum_difference}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'output', 'clip'):
        parser.add_argument('--'+name, type=Path, required=True)
    main(parser.parse_args())
