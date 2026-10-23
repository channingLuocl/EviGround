"""Prepare public part text, cache frozen point features, then train small v8.3 heads."""
import argparse
import json
from pathlib import Path

import numpy as np

from category_common_v82 import (ROOT, category_vocabulary, file_hash, instance_seed, load_cloud,
                                 normalized_category, run_lock)
from evidence_data_v83 import (VERSION, PATCHES, check_disjoint, crop_view, load_manifest,
                               load_part_masks, load_parts, manifest_sources)
from pointcloud_classifier_v82 import CLIP_REVISION, load_text_features
from pointcloud_classifier_v83 import FrozenFeatures, load_part_text
from run_storage import ensure_run_writable, write_json


def make_manifest(args):
    """Explicit training supervision only; never fill part labels from object classes."""
    ensure_run_writable(args.output.parent)
    if args.output.exists():
        raise ValueError('Manifest exists; choose a new output')
    scenes = sorted({r['scene'] for r in json.loads(args.scenes.read_text())['rows']})
    vocabulary = {normalized_category(c) for c in category_vocabulary()}
    samples, excluded = [], 0
    for scene in scenes:
        if Path(scene).name != scene or scene in ('.', '..'):
            raise ValueError('Invalid scene identity')
        directory = (args.scene_root/scene).resolve()
        _, _, ids = load_cloud(directory/'scene.npz')
        available = {int(i) for i in np.unique(ids) if i >= 0}
        for row in json.loads((directory/'objects_gt.json').read_text()):
            if normalized_category(row['target']) not in vocabulary:
                excluded += 1
                continue
            if row['bbox_id'] not in available:
                raise ValueError('Annotated training instance missing in scene cloud')
            samples.append({'scene': scene, 'bbox_id': row['bbox_id'],
                            'cloud': str(directory/'scene.npz'), 'category': row['target'], 'part_states': {}})
    if not samples:
        raise ValueError('No in-vocabulary training instances')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump({'format': 'v8.3-training-manifest-v1', 'split': args.split,
                   'annotation_protocol': 'Category labels from designated supervised training/dev scenes. '
                                          'Part states UNLABELLED: requires verified manual/external annotations.',
                   'samples': samples}, handle, ensure_ascii=False, indent=2)
    print(f'Created {len(samples)} {args.split} rows; excluded {excluded} out-of-vocabulary labels. '
          'No part absence labels inferred. Evidence mode requires additional annotations.')


def prepare_text(args):
    import torch
    from transformers import AutoTokenizer, CLIPTextModelWithProjection
    schema = load_parts(args.parts)
    ensure_run_writable(args.output.parent)
    if args.output.exists():
        raise ValueError('Output exists; choose a new path')
    name = 'openai/clip-vit-base-patch16'
    tokenizer = AutoTokenizer.from_pretrained(name, revision=CLIP_REVISION)
    model = CLIPTextModelWithProjection.from_pretrained(name, revision=CLIP_REVISION).eval()
    tokens = tokenizer([schema['prompt'].format(p) for p in schema['parts']],
                       padding=True, return_tensors='pt')
    with torch.inference_mode():
        values = model(**tokens).text_embeds.float().numpy()
    metadata = {'parts': schema['parts'], 'prompt': schema['prompt'],
                'text_model': name, 'text_revision': CLIP_REVISION}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as handle:
        np.savez_compressed(handle, embeddings=values, metadata=json.dumps(metadata))
    print(f'Public part text saved: {args.output}; no instance labels.')


def cache_features(args):
    schema = load_parts(args.parts)
    manifest, samples = load_manifest(args.manifest, schema['parts'])
    manifest_hash = file_hash(args.manifest)
    if args.crop_views and manifest['split'] != 'train':
        raise ValueError('Crop augmentation is training-only; dev must use --crop-views 0')
    sources = manifest_sources(samples)
    output = args.output.resolve()
    with run_lock(output):
        if any(p.name != '.run.lock' for p in output.iterdir()):
            raise ValueError('Feature cache output must be empty; choose a new --output')
        assets = {'pointnext_sha256': file_hash(args.checkpoint),
                  'category_text_sha256': file_hash(args.text_features)}
        extractor = FrozenFeatures(args.checkpoint, args.text_features)
        records, arrays = [], {k: [] for k in ('global_features', 'local_features', 'geometry', 'labels', 'states')}
        # One scene at a time without retaining every scene cloud in RAM.
        current, cloud = None, None
        for index, sample in enumerate(samples):
            if sample['cloud'] != current:
                current = sample['cloud']
                cloud = load_cloud(current)
            xyz, _, ids = cloud
            points = xyz[ids == sample['bbox_id']]
            if len(points) < 3:
                raise ValueError(f'Instance missing or too small: {sample["scene"]}/{sample["bbox_id"]}')
            seed = instance_seed(args.seed, sample['scene'], sample['bbox_id'])
            views = [(points, sample['states'], 'original')]
            if args.crop_views:
                if not sample['part_masks']:
                    raise ValueError('--crop-views requires verified observed part_masks for every sample')
                masks = load_part_masks(sample['part_masks'], len(points), schema['parts'], sample['states'])
                for view in range(args.crop_views):
                    keep, states = crop_view(points, sample['states'], masks, (seed+view+1) % 2**32)
                    views.append((points[keep], states, f'crop_{view}'))
            for view, (visible, states, name) in enumerate(views):
                features = extractor.extract(visible, (seed+view) % 2**32)
                for key in ('global_features', 'local_features', 'geometry'):
                    arrays[key].append(features[key])
                arrays['labels'].append(sample['label'])
                arrays['states'].append(states)
                records.append({'scene': sample['scene'], 'bbox_id': sample['bbox_id'], 'view': name})
            print(f'Cached {index+1}/{len(samples)} instances', flush=True)
        if sources != manifest_sources(samples) or manifest_hash != file_hash(args.manifest):
            raise ValueError('Source files changed during feature extraction')
        metadata = {'version': VERSION, 'split': manifest['split'], 'parts': schema['parts'],
                    'parts_schema': schema, 'manifest_sha256': manifest_hash,
                    'annotation_protocol': manifest['annotation_protocol'], 'sources': sources,
                    'extractor': extractor.metadata, 'extractor_assets': assets,
                    'seed': args.seed, 'crop_views': args.crop_views, 'records': records,
                    'code_sha256': training_code_hashes()}
        temporary = output/'features.tmp'
        with temporary.open('xb') as handle:
            np.savez_compressed(handle, **{k: np.asarray(v) for k, v in arrays.items()},
                                metadata=json.dumps(metadata))
        temporary.replace(output/'features.npz')
        write_json(output/'config.json', metadata)
        print(f'Saved {len(records)} feature rows: {output/"features.npz"}')


def training_code_hashes():
    return {name: file_hash(Path(__file__).parent/name) for name in
            ('train_categories_v83.py', 'evidence_head_v83.py', 'evidence_data_v83.py',
             'pointcloud_classifier_v83.py', 'pointcloud_classifier_v82.py', 'category_common_v82.py')}


def read_cache(path):
    import torch
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata']))
        values = {k: data[k] for k in ('global_features', 'local_features', 'geometry', 'labels', 'states')}
    count, parts = len(values['labels']), len(metadata['parts'])
    expected = {'global_features': (count, 512), 'local_features': (count, PATCHES, 512),
                'geometry': (count, PATCHES, 7), 'states': (count, parts), 'labels': (count,)}
    if (metadata.get('version') != VERSION or not count
            or any(a.shape != expected[k] or not np.isfinite(a).all() for k, a in values.items())
            or values['labels'].dtype.kind not in 'iu' or values['states'].dtype.kind not in 'iu'
            or (values['labels'] < 0).any() or (values['labels'] >= len(category_vocabulary())).any()
            or (values['states'] < -1).any() or (values['states'] > 2).any()
            or (np.linalg.norm(values['global_features'], axis=-1) <= 0).any()
            or (np.linalg.norm(values['local_features'], axis=-1) <= 0).any()
            or len(metadata['records']) != count):
        raise ValueError('Invalid numeric feature cache')
    tensors = {k: torch.from_numpy(v).long() if k in ('labels', 'states') else
               torch.from_numpy(v).float() for k, v in values.items()}
    return tensors, metadata


def train(args):
    import torch
    from evidence_head_v83 import EvidenceHead, training_loss
    train_data, train_meta = read_cache(args.train_cache)
    dev_data, dev_meta = read_cache(args.dev_cache)
    if train_meta['split'] != 'train' or dev_meta['split'] != 'dev' or dev_meta['crop_views']:
        raise ValueError('Require train and unaugmented dev caches')
    check_disjoint(train_meta['sources'], dev_meta['sources'])
    for key in ('parts_schema', 'extractor', 'extractor_assets', 'code_sha256'):
        if train_meta[key] != dev_meta[key]:
            raise ValueError(f'Train/dev extraction contracts differ: {key}')
    if train_meta['code_sha256'] != training_code_hashes():
        raise ValueError('Feature/training code changed; regenerate caches')
    if file_hash(args.text_features) != train_meta['extractor_assets']['category_text_sha256']:
        raise ValueError('Category text differs from frozen feature extraction')
    if args.mode == 'evidence':
        # A part with no state supervision cannot legitimately be called an evidence estimator.
        missing = [p for i, p in enumerate(train_meta['parts'])
                   if set(train_data['states'][:, i].tolist()) - {-1} != {0, 1, 2}]
        if missing:
            raise ValueError('Evidence training needs present/absent/unknown annotations per part. '
                             f'Missing state coverage: {missing}; curate annotations or reduce public part schema.')
    category_text, _ = load_text_features(args.text_features)
    part_text = load_part_text(args.part_text_features, train_meta['parts_schema'])
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    head = EvidenceHead(torch.from_numpy(category_text), torch.from_numpy(part_text), args.mode)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=.01)
    def forward(data, indices):
        return head(*(data[k][indices] for k in ('global_features', 'local_features', 'geometry')))
    output = args.output.resolve()
    with run_lock(output):
        if any(p.name != '.run.lock' for p in output.iterdir()):
            raise ValueError('Training output must be empty; choose a new --output')
        provenance = {'version': VERSION, 'mode': args.mode, 'trained': True,
                      'parts': train_meta['parts'], 'parts_schema': train_meta['parts_schema'],
                      'vocabulary': list(category_vocabulary()), 'extractor': train_meta['extractor'],
                      'extractor_assets': train_meta['extractor_assets'],
                      'train_sources': train_meta['sources'], 'dev_sources': dev_meta['sources'],
                      'train_cache_sha256': file_hash(args.train_cache), 'dev_cache_sha256': file_hash(args.dev_cache),
                      'part_text_sha256': file_hash(args.part_text_features),
                      'code_sha256': training_code_hashes(), 'seed': args.seed, 'epochs': args.epochs,
                      'learning_rate': args.lr, 'batch_size': args.batch_size,
                      'supervision': 'supervised_adaptation_not_claimed_unseen_class_zero_shot',
                      'parameters': sum(p.numel() for p in head.parameters())}
        write_json(output/'config.json', provenance)
        best, history = float('inf'), []
        for epoch in range(args.epochs):
            head.train()
            total = 0.
            for indices in torch.randperm(len(train_data['labels'])).split(args.batch_size):
                optimizer.zero_grad()
                loss, _, _ = training_loss(forward(train_data, indices), train_data['labels'][indices],
                                            train_data['states'][indices])
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 5.)
                optimizer.step()
                total += float(loss.detach()) * len(indices)
            head.eval()
            dev_loss, correct, state_correct, state_count, state_brier = 0., 0, 0, 0, 0.
            with torch.inference_mode():
                for indices in torch.arange(len(dev_data['labels'])).split(args.batch_size):
                    result = forward(dev_data, indices)
                    dev_loss += float(torch.nn.functional.cross_entropy(result['logits'],
                                      dev_data['labels'][indices], reduction='sum'))
                    correct += int((result['logits'].argmax(-1) == dev_data['labels'][indices]).sum())
                    valid = dev_data['states'][indices] >= 0
                    if 'state_probabilities' in result and valid.any():
                        probs = result['state_probabilities'][valid]
                        targets = dev_data['states'][indices][valid]
                        state_count += len(targets)
                        state_correct += int((probs.argmax(-1) == targets).sum())
                        one_hot = torch.nn.functional.one_hot(targets, 3).float()
                        state_brier += float(((probs-one_hot)**2).sum())
            dev_loss /= len(dev_data['labels'])
            row = {'epoch': epoch+1, 'train_loss': total/len(train_data['labels']),
                   'dev_category_loss': dev_loss, 'dev_accuracy': correct/len(dev_data['labels']),
                   'dev_state_accuracy': state_correct/state_count if state_count else None,
                   'dev_state_brier': state_brier/state_count if state_count else None}
            history.append(row)
            write_json(output/'history.json', history)
            print(json.dumps(row), flush=True)
            if dev_loss < best:
                best = dev_loss
                temporary = output/'head.tmp'
                torch.save({**provenance, 'best_epoch': epoch+1, 'state_dict': head.state_dict()}, temporary)
                temporary.replace(output/'head.pt')
        print(f'Trained head saved: {output/"head.pt"}; dev-selected, no test evaluation performed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    manifest = sub.add_parser('make-manifest')
    manifest.add_argument('--scenes', type=Path, required=True, help='JSON {rows:[{scene:...}]} of train OR dev scenes')
    manifest.add_argument('--scene-root', type=Path, required=True)
    manifest.add_argument('--split', choices=('train', 'dev'), required=True)
    manifest.add_argument('--output', type=Path, required=True)
    text = sub.add_parser('prepare-text')
    text.add_argument('--parts', type=Path, default=Path(__file__).with_name('parts_v83.json'))
    text.add_argument('--output', type=Path, required=True)
    cache = sub.add_parser('cache')
    cache.add_argument('--parts', type=Path, default=Path(__file__).with_name('parts_v83.json'))
    cache.add_argument('--manifest', type=Path, required=True)
    cache.add_argument('--checkpoint', type=Path, default=ROOT/'weights/pnext_cls.pth')
    cache.add_argument('--text-features', type=Path, default=ROOT/'weights/v82_clip_text_features.npz')
    cache.add_argument('--output', type=Path, required=True)
    cache.add_argument('--crop-views', type=int, default=0)
    cache.add_argument('--seed', type=int, default=42)
    trainer = sub.add_parser('train')
    trainer.add_argument('--mode', choices=('global', 'attention', 'voting', 'evidence'), default='evidence')
    trainer.add_argument('--train-cache', type=Path, required=True)
    trainer.add_argument('--dev-cache', type=Path, required=True)
    trainer.add_argument('--part-text-features', type=Path, required=True)
    trainer.add_argument('--text-features', type=Path, default=ROOT/'weights/v82_clip_text_features.npz')
    trainer.add_argument('--output', type=Path, required=True)
    trainer.add_argument('--epochs', type=int, default=50)
    trainer.add_argument('--batch-size', type=int, default=32)
    trainer.add_argument('--lr', type=float, default=.001)
    trainer.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.command == 'cache' and args.crop_views < 0:
        parser.error('--crop-views must be nonnegative')
    if args.command == 'train' and (args.epochs < 1 or args.batch_size < 1 or not 0 < args.lr < 1):
        parser.error('Require positive epochs/batch-size and 0 < lr < 1')
    try:
        {'make-manifest': make_manifest, 'prepare-text': prepare_text,
         'cache': cache_features, 'train': train}[args.command](args)
    except (ValueError, OSError, ImportError, RuntimeError, KeyError) as exc:
        parser.exit(1, f'{type(exc).__name__}: {exc}\nSee local/README_V8_3.md\n')


if __name__ == '__main__':
    main()
