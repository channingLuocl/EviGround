"""Prepare ScanNet training data, cache PointNeXt features, train and evaluate v10."""
from seeground.paths import PACKAGE_ROOT, source_files
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from seeground.categories.category_common_v82 import ROOT, category_vocabulary, file_hash, instance_seed, load_cloud, run_lock
from seeground.classifiers.category_v10.data import (VERSION, assert_disjoint, convert_scene, prepare_manifests, scene_names)
from seeground.evaluation.run_storage import write_json


def code_hashes():
    return {p.relative_to(PACKAGE_ROOT).as_posix(): file_hash(p) for p in source_files()
            if not p.name.startswith('test_')}


def read_manifest(path):
    value = json.loads(Path(path).read_text())
    if (value.get('version') != VERSION or value.get('split') not in ('train', 'dev', 'eval')
            or value.get('vocabulary') != list(category_vocabulary()) or not value.get('samples')):
        raise ValueError('Expected a prepared v10 category manifest')
    seen = set()
    from seeground.categories.category_common_v82 import normalized_category
    vocabulary = {normalized_category(c): i for i, c in enumerate(category_vocabulary())}
    for row in value['samples']:
        identity = (row['scene'], row['bbox_id'])
        if (identity in seen or row['scene'] not in value['scenes']
                or type(row['bbox_id']) is not int or row['bbox_id'] < 0
                or type(row['label']) is not int or not -1 <= row['label'] < len(category_vocabulary())
                or vocabulary.get(normalized_category(row['category']), -1) != row['label']
                or (value['split'] == 'train' and row['label'] < 0)):
            raise ValueError('Invalid/duplicate category training sample')
        seen.add(identity)
    return value


def cache(args):
    from seeground.classifiers.pointcloud_classifier_v83 import FrozenFeatures
    manifest = read_manifest(args.manifest)
    with run_lock(args.output):
        extractor = FrozenFeatures(args.checkpoint, args.text_features)
        config = {'version': VERSION, 'manifest_sha256': file_hash(args.manifest),
                  'split': manifest['split'], 'seed': args.seed, 'extractor': extractor.metadata,
                  'text_sha256': file_hash(args.text_features), 'code_sha256': code_hashes(),
                  'augmentation': 'deterministic_20_percent_point_dropout_train_only',
                  'vocabulary': manifest['vocabulary'], 'sources': manifest['sources']}
        path = args.output/'config.json'
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError('Cache inputs/code changed; choose a fresh --output')
        write_json(path, config)
        sources = {s['scene']: s for s in manifest['sources']}
        current, cloud, values, records = None, None, [], []
        for index, sample in enumerate(manifest['samples']):
            if sample['scene'] != current:
                current = sample['scene']
                if file_hash(sample['cloud']) != sources[current]['cloud_sha256']:
                    raise ValueError(f'Cloud changed since preparation: {current}')
                cloud = load_cloud(sample['cloud'])
            xyz, _, ids = cloud
            points = xyz[ids == sample['bbox_id']]
            if len(points) < 3:
                raise ValueError(f'Instance has fewer than three points: {current}/{sample["bbox_id"]}')
            seed = instance_seed(args.seed, current, sample['bbox_id'])
            entry = args.output/'instances'/current/f'{sample["bbox_id"]:04d}.npz'
            if not entry.exists():
                views = [points]
                if manifest['split'] == 'train':
                    keep = np.random.RandomState(seed).choice(len(points), max(3, int(.8*len(points))),
                                                               replace=False)
                    views.append(points[np.sort(keep)])
                features = [extractor.extract(p, (seed+i) % 2**32) for i, p in enumerate(views)]
                entry.parent.mkdir(parents=True, exist_ok=True)
                with entry.with_suffix('.tmp').open('wb') as handle:
                    np.savez_compressed(handle, **{k: np.stack([v[k] for v in features]) for k in
                                                 ('global_features', 'local_features', 'geometry')})
                entry.with_suffix('.tmp').replace(entry)
            with np.load(entry, allow_pickle=False) as item:
                values.append({k: item[k] for k in ('global_features', 'local_features', 'geometry')})
            records.append({'scene': current, 'bbox_id': sample['bbox_id'],
                            'bbox_3d': np.r_[(points.min(0)+points.max(0))/2, np.ptp(points, axis=0)].tolist(),
                            'category': sample['category']})
            print(f'Cached {index+1}/{len(manifest["samples"])} instances', flush=True)
        if file_hash(args.manifest) != config['manifest_sha256']:
            raise ValueError('Manifest changed during extraction')
        metadata = {**config, 'records': records}
        with (args.output/'features.tmp').open('wb') as handle:
            np.savez_compressed(handle, **{k: np.stack([v[k] for v in values]) for k in values[0]},
                                labels=np.array([s['label'] for s in manifest['samples']], dtype=np.int64),
                                metadata=json.dumps(metadata))
        (args.output/'features.tmp').replace(args.output/'features.npz')
        # Validate the complete cache before declaring the GPU stage finished.
        read_cache(args.output/'features.npz')


def read_cache(path):
    import torch
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['metadata']))
        arrays = {k: data[k] for k in ('global_features', 'local_features', 'geometry', 'labels')}
    if (meta.get('version') != VERSION or meta.get('split') not in ('train', 'dev', 'eval')
            or meta.get('vocabulary') != list(category_vocabulary())):
        raise ValueError('Invalid v10 cache metadata')
    count, views = len(arrays['labels']), 2 if meta['split'] == 'train' else 1
    shapes = {'global_features': (count, views, 512), 'local_features': (count, views, 8, 512),
              'geometry': (count, views, 8, 7), 'labels': (count,)}
    if (not count or len(meta['records']) != count
            or any(v.shape != shapes[k] or not np.isfinite(v).all() for k, v in arrays.items())
            or arrays['labels'].dtype.kind not in 'iu'
            or np.any(arrays['labels'] < (0 if meta['split'] == 'train' else -1))
            or np.any(arrays['labels'] >= len(category_vocabulary()))
            or any(np.any(np.linalg.norm(arrays[k], axis=-1) <= 0)
                   for k in ('global_features', 'local_features'))):
        raise ValueError('Invalid numeric v10 feature cache')
    keys = [(r['scene'], r['bbox_id']) for r in meta['records']]
    if len(set(keys)) != count or set(s for s, _ in keys) != {s['scene'] for s in meta['sources']}:
        raise ValueError('Duplicate instances or inconsistent scene provenance in cache')
    return {k: torch.from_numpy(v).long() if k == 'labels' else torch.from_numpy(v).float()
            for k, v in arrays.items()}, meta


def forward(head, data, indices, device, view=0):
    return head(*(data[k][indices, view].to(device) for k in
                  ('global_features', 'local_features', 'geometry')))


def metrics(logits, labels):
    import torch
    from torch.nn import functional as F
    order = logits.argsort(dim=-1, descending=True, stable=True)
    correct = order[:, 0] == labels
    known = labels >= 0
    return {'count': len(labels), 'correct': int(correct.sum()), 'accuracy': float(correct.float().mean()),
            'top5_accuracy': float((order[:, :5] == labels[:, None]).any(-1).float().mean()),
            'class_mean_accuracy_in_vocabulary': float(torch.stack([correct[labels == c].float().mean()
                                         for c in labels[known].unique()]).mean()) if known.any() else None,
            'out_of_vocabulary': int((~known).sum()),
            'category_loss': float(F.cross_entropy(logits[known], labels[known])) if known.any() else None}


def evaluate_head(head, data, batch_size, device):
    import torch
    head.eval()
    with torch.inference_mode():
        return torch.cat([forward(head, data, indices, device)['logits'].cpu()
                          for indices in torch.arange(len(data['labels'])).split(batch_size)])


def check_sources(first, second):
    assert_disjoint([s['scene'] for s in first], [s['scene'] for s in second])
    if {s['cloud_sha256'] for s in first} & {s['cloud_sha256'] for s in second}:
        raise ValueError('Identical point-cloud contents across splits')


def train(args):
    import torch
    from seeground.classifiers.category_v10.model import CategoryHead, adaptation_loss
    from seeground.classifiers.pointcloud_classifier_v82 import load_text_features
    train_data, train_meta = read_cache(args.train_cache)
    dev_data, dev_meta = read_cache(args.dev_cache)
    if train_meta['split'] != 'train' or dev_meta['split'] != 'dev':
        raise ValueError('Training requires train and internal-dev caches; never validation/eval')
    check_sources(train_meta['sources'], dev_meta['sources'])
    for key in ('extractor', 'text_sha256', 'code_sha256', 'seed', 'augmentation'):
        if train_meta[key] != dev_meta[key]:
            raise ValueError(f'Train/dev extraction differs: {key}')
    if file_hash(args.text_features) != train_meta['text_sha256'] or train_meta['code_sha256'] != code_hashes():
        raise ValueError('Text features or experiment code changed; regenerate caches')
    if not (dev_data['labels'] >= 0).any():
        raise ValueError('Development cache has no in-vocabulary instances for model selection')
    text, _ = load_text_features(args.text_features)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; cached-feature head training can use --device cpu')
    head = CategoryHead(torch.from_numpy(text), args.mode).to(args.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=.01)
    with run_lock(args.output):
        if any(p.name != '.run.lock' for p in args.output.iterdir()):
            raise ValueError('Choose a fresh training output directory')
        config = {'version': VERSION, 'mode': args.mode, 'seed': args.seed, 'epochs': args.epochs,
                  'lr': args.lr, 'batch_size': args.batch_size, 'consistency_weight': args.consistency_weight,
                  'device': args.device, 'torch': str(torch.__version__), 'code_sha256': code_hashes(),
                  'parameters': sum(p.numel() for p in head.parameters()), 'backbone_frozen': True,
                  'train_cache_sha256': file_hash(args.train_cache), 'dev_cache_sha256': file_hash(args.dev_cache),
                  'text_sha256': train_meta['text_sha256'], 'extractor': train_meta['extractor'],
                  'train_sources': train_meta['sources'], 'dev_sources': dev_meta['sources'],
                  'learning_regime': 'supervised_category_adaptation_not_zero_shot',
                  'selection': 'minimum_internal_dev_cross_entropy; official validation not used'}
        write_json(args.output/'config.json', config)
        history, best = [], float('inf')
        try:
            for epoch in range(args.epochs):
                head.train()
                total = 0.
                for indices in torch.randperm(len(train_data['labels'])).split(args.batch_size):
                    optimizer.zero_grad()
                    loss = adaptation_loss(forward(head, train_data, indices, args.device, 0),
                                           forward(head, train_data, indices, args.device, 1),
                                           train_data['labels'][indices].to(args.device),
                                           args.consistency_weight)
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite training loss')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 5.)
                    optimizer.step()
                    total += float(loss.detach()) * len(indices)
                logits = evaluate_head(head, dev_data, args.batch_size, args.device)
                row = {'epoch': epoch+1, 'train_loss': total/len(train_data['labels']),
                       'dev': metrics(logits, dev_data['labels'])}
                history.append(row)
                write_json(args.output/'history.json', history)
                print(json.dumps(row), flush=True)
                if row['dev']['category_loss'] < best:
                    best = row['dev']['category_loss']
                    torch.save({'config': config, 'best_epoch': epoch+1,
                                'state_dict': {k: v.detach().cpu() for k, v in head.state_dict().items()}},
                               args.output/'head.tmp')
                    (args.output/'head.tmp').replace(args.output/'head.pt')
        except KeyboardInterrupt:
            print('Stopped; completed epoch history and best checkpoint preserved.', file=sys.stderr)
            return 130
    return 0


def predict(args):
    import torch
    from seeground.classifiers.category_v10.model import CategoryHead
    data, meta = read_cache(args.cache)
    checkpoint = torch.load(args.head, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    if meta['split'] != 'eval':
        raise ValueError('Use the held-out evaluation cache for reported predictions')
    check_sources(config['train_sources'] + config['dev_sources'], meta['sources'])
    if (config['version'] != VERSION or config['code_sha256'] != code_hashes()
            or config['extractor'] != meta['extractor'] or config['text_sha256'] != meta['text_sha256']):
        raise ValueError('Checkpoint/evaluation feature contracts differ')
    head = CategoryHead(checkpoint['state_dict']['category_text'], config['mode'])
    head.load_state_dict(checkpoint['state_dict'], strict=True)
    head.to(args.device)
    torch.set_num_threads(args.threads)
    with run_lock(args.output):
        if any(p.name != '.run.lock' for p in args.output.iterdir()):
            raise ValueError('Choose a fresh prediction output directory')
        logits = evaluate_head(head, data, args.batch_size, args.device)
        frozen = CategoryHead(head.category_text.detach().cpu(), 'frozen').to(args.device)
        baseline_logits = evaluate_head(frozen, data, args.batch_size, args.device)
        probabilities = logits.softmax(-1)
        vocabulary, predictions = list(category_vocabulary()), []
        for row, scores in zip(meta['records'], probabilities.tolist()):
            order = sorted(range(len(scores)), key=lambda c: -scores[c])[:5]
            predictions.append({k: row[k] for k in ('scene', 'bbox_id', 'bbox_3d')})
            predictions[-1].update(status='completed', predicted_object=vocabulary[order[0]],
                top5=[{'category': vocabulary[c], 'score': scores[c]} for c in order], scores=scores,
                score_type='uncalibrated_softmax_not_v83_similarity')
        summary = {'version': VERSION, 'completed': len(predictions), 'errors': 0, 'complete': True,
                   'learning_regime': config['learning_regime'],
                   'evaluation_scope': 'scene_disjoint_supervised_adaptation; previously_inspected_validation'}
        write_json(args.output/'config.json', {'version': VERSION, 'head_sha256': file_hash(args.head),
                   'cache_sha256': file_hash(args.cache), 'code_sha256': code_hashes(),
                   'sources': meta['sources'], 'vocabulary': vocabulary, 'training': config,
                   'score_type': 'uncalibrated_softmax_not_v83_similarity'})
        # Predictions contain no GT names, target labels or language queries.
        write_json(args.output/'predictions.json', {'summary': summary, 'results': predictions})
        write_json(args.output/'results.json', {'summary': {**summary, **metrics(logits, data['labels'])},
                   'frozen_single_view_global_baseline': metrics(baseline_logits, data['labels']),
                   'results': [{**p, 'target': row['category'],
                                'frozen_global_predicted_object': vocabulary[int(baseline_logits[i].argmax())],
                                'frozen_global_correct': int(baseline_logits[i].argmax()) == int(data['labels'][i]),
                                'correct': int(logits[i].argmax()) == int(data['labels'][i])}
                               for i, (p, row) in enumerate(zip(predictions, meta['records']))]})
        write_json(args.output/'FROZEN.json', {'version': VERSION, 'files': {
                   name: file_hash(args.output/name) for name in ('config.json', 'predictions.json', 'results.json')}})
        print(json.dumps(metrics(logits, data['labels'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    convert = sub.add_parser('convert', help='Convert officially acquired ScanNet raw scans, without author OLT')
    convert.add_argument('--scenes', type=Path, required=True)
    convert.add_argument('--raw-root', type=Path, required=True, help='Contains sceneXXXX_XX/ subdirectories')
    convert.add_argument('--output', type=Path, required=True)
    prepare = sub.add_parser('prepare', help='Official train scenes -> room-disjoint train/dev manifests')
    prepare.add_argument('--train-scenes', type=Path, required=True)
    prepare.add_argument('--validation-scenes', type=Path, default=ROOT/'data/scannet/scannetv2_val.txt')
    prepare.add_argument('--eval-scenes', type=Path, default=ROOT/'data/validation/nr3d_val_full.json')
    prepare.add_argument('--scene-root', type=Path, required=True)
    prepare.add_argument('--dev-fraction', type=float, default=.1)
    prepare.add_argument('--seed', type=int, default=42)
    prepare.add_argument('--output', type=Path, required=True)
    feature = sub.add_parser('cache', help='GPU stage: one original view and training-only dropout view')
    feature.add_argument('--manifest', type=Path, required=True)
    feature.add_argument('--checkpoint', type=Path, default=ROOT/'weights/pnext_cls.pth')
    feature.add_argument('--text-features', type=Path, default=ROOT/'weights/v82_clip_text_features.npz')
    feature.add_argument('--seed', type=int, default=42)
    feature.add_argument('--output', type=Path, required=True)
    fitting = sub.add_parser('train', help='Train a small head on cached frozen features; CPU is supported')
    fitting.add_argument('--train-cache', type=Path, required=True)
    fitting.add_argument('--dev-cache', type=Path, required=True)
    fitting.add_argument('--text-features', type=Path, default=ROOT/'weights/v82_clip_text_features.npz')
    fitting.add_argument('--mode', choices=('global', 'attention', 'conditioned'), default='conditioned')
    fitting.add_argument('--epochs', type=int, default=40)
    fitting.add_argument('--batch-size', type=int, default=64)
    fitting.add_argument('--lr', type=float, default=.003)
    fitting.add_argument('--consistency-weight', type=float, default=.1)
    fitting.add_argument('--seed', type=int, default=42)
    fitting.add_argument('--threads', type=int, default=10)
    fitting.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    fitting.add_argument('--output', type=Path, required=True)
    inference = sub.add_parser('predict', help='Evaluate a trained head on scene-disjoint cached features')
    inference.add_argument('--cache', type=Path, required=True)
    inference.add_argument('--head', type=Path, required=True)
    inference.add_argument('--batch-size', type=int, default=64)
    inference.add_argument('--threads', type=int, default=10)
    inference.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    inference.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command in ('train', 'predict'):
        if min(args.batch_size, args.threads, getattr(args, 'epochs', 1)) < 1:
            parser.error('epochs, batch size and threads must be positive')
    if args.command == 'train' and (not np.isfinite([args.lr, args.consistency_weight]).all()
                                     or args.lr <= 0 or args.consistency_weight < 0):
        parser.error('Learning rate must be positive; consistency weight must be nonnegative and finite')
    if args.command == 'convert':
        for scene in scene_names(args.scenes):
            convert_scene(args.raw_root, scene, args.output/scene)
            print(f'Converted {scene}', flush=True)
        return 0
    return {'prepare': prepare_manifests, 'cache': cache, 'train': train, 'predict': predict}[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
