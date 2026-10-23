"""Build frozen modality features with revised manifests and sealed holdout access."""
from seeground.paths import source_file
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from seeground.categories.category_common_v82 import file_hash, instance_seed, load_cloud, run_lock
from seeground.classifiers.rgb_classifier import RENDER, extract, load_clip, verify_text
from seeground.experiments.run_rgb_experiment import check_frozen, freeze
from seeground.evaluation.run_storage import write_json
from seeground.geometry.shape_scale import load_global, load_split


def identity(rows):
    return [(r['scene'], r['bbox_id'], r['category']) for r in rows]


def main(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    check_frozen(args.output, 'PROTOCOL_FROZEN.json')
    prior = json.loads((args.prior / 'config.json').read_text())
    text, _ = load_global(args, 42, prior)
    root = args.output / 'features'
    with run_lock(root):
        cfg = {'version': 'scan-domain-modality-cache-r1', 'renderer': RENDER,
               'clip_sha256': file_hash(args.clip), 'protocol_sha256': file_hash(args.output / 'protocol.json'),
               'train_manifest_sha256': file_hash(args.output / 'manifests/train.json'),
               'dev_manifest_sha256': file_hash(args.output / 'manifests/dev.json'),
               'source_sha256': {n: file_hash(source_file(n)) for n in
                  ('cache_scan_domain.py', 'rgb_classifier.py', 'scan_domain_protocol.py')},
               'original_cache_sha256': prior['cache_sha256'], 'prior_sha256': file_hash(args.prior / 'config.json'),
               'dev_RGB_sha256': file_hash(args.old_rgb / 'eval/features.npz')}
        config_path = root / 'config.json'
        if config_path.exists() and json.loads(config_path.read_text()) != cfg:
            raise ValueError('Feature extraction inputs/code changed')
        write_json(config_path, cfg)
        if args.stage == 'train':
            train = json.loads((args.output / 'manifests/train.json').read_text())
            dev = json.loads((args.output / 'manifests/dev.json').read_text())
            old = {s: load_split(args, s, prior) for s in ('train', 'dev', 'eval')}
            lookup = {key: data['features'][i, 0].numpy() for data in (old['train'], old['dev'])
                      for i, key in enumerate(identity(data['manifest']['samples']))}
            shape_train = np.stack([lookup[key] for key in identity(train['samples'])])
            if identity(dev['samples']) != identity(old['eval']['manifest']['samples']):
                raise ValueError('Research-dev identity differs from historical evaluation')
            np.savez_compressed(root / 'shape_train.npz', features=shape_train)
            np.savez_compressed(root / 'shape_dev.npz', features=old['eval']['features'][:, 0].numpy())
            np.save(root / 'text.npy', text.numpy())
            check_frozen(args.old_rgb, 'RESULTS_FROZEN.json')
            old_cfg = json.loads((args.old_rgb / 'config.json').read_text())
            if old_cfg['clip_sha256'] != cfg['clip_sha256'] or old_cfg['renderer'] != RENDER:
                raise ValueError('Historical RGB extraction differs')
            with np.load(args.old_rgb / 'eval/features.npz') as archive:
                np.savez_compressed(root / 'rgb_dev.npz', features=archive['features'], valid=archive['valid'], coverage=archive['coverage'])
            del old, lookup, shape_train
            model, tokenizer = load_clip(args.clip, 'cuda')
            alignment = verify_text(model, tokenizer, train['vocabulary'], text, 'cuda')
            write_json(root / 'text_alignment.json', alignment)
            extract(model, train, root / 'rgb_train', 'cuda')
            freeze(args.output, 'FEATURES_FROZEN.json', ['features/config.json', 'features/shape_train.npz',
                'features/shape_dev.npz', 'features/rgb_train/features.npz', 'features/rgb_dev.npz', 'features/text.npy'])
            print(json.dumps({'stage': 'train_dev_features_complete'}), flush=True)
        else:
            # This guard precedes loading or scoring heldout object features.
            check_frozen(args.output, 'MODEL_SELECTION_FROZEN.json')
            manifest_path = args.output / 'manifests/eval.json'
            seal = json.loads((args.output / 'HOLDOUT_FROZEN.json').read_text())
            if file_hash(manifest_path) != seal['manifest_sha256']:
                raise ValueError('Heldout assets changed after sealing')
            manifest = json.loads(manifest_path.read_text())
            from seeground.classifiers.pointcloud_classifier_v82 import PointCloudClassifier
            encoder = PointCloudClassifier('pointnext', args.pointnext, args.text)
            values, current = [], None
            source = {r['scene']: r['cloud_sha256'] for r in manifest['sources']}
            for i, row in enumerate(manifest['samples']):
                if row['scene'] != current:
                    current = row['scene']
                    if file_hash(row['cloud']) != source[current]:
                        raise ValueError('Heldout cloud changed')
                    xyz, rgb, ids = load_cloud(row['cloud'])
                from seeground.classifiers.pointcloud_classifier_v82 import prepare_points
                seed = instance_seed(42, current, row['bbox_id'])
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                points = prepare_points(xyz[ids == row['bbox_id']], None, 'pointnext', seed)
                with torch.inference_mode():
                    value = F.normalize(encoder.encoder(torch.from_numpy(points[None]).cuda()).float(), dim=-1)
                values.append(value[0].cpu().numpy())
            np.savez_compressed(root / 'shape_eval.npz', features=np.stack(values))
            del encoder
            torch.cuda.empty_cache()
            model, tokenizer = load_clip(args.clip, 'cuda')
            verify_text(model, tokenizer, manifest['vocabulary'], text, 'cuda')
            extract(model, manifest, root / 'rgb_eval', 'cuda')
            freeze(args.output, 'EVAL_FEATURES_FROZEN.json', ['features/shape_eval.npz', 'features/rgb_eval/features.npz', 'manifests/eval.json'])
            print(json.dumps({'stage': 'heldout_features_complete', 'objects': len(values)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('train', 'eval'))
    for name in ('feature-root', 'prior', 'output', 'old-rgb', 'clip', 'pointnext', 'text'):
        parser.add_argument('--' + name, type=Path, required=True)
    main(parser.parse_args())
