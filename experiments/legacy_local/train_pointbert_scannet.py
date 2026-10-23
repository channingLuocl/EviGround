"""Supervised ScanNet adaptation of the released OpenShape PointBERT encoder."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import shutil
import time
import traceback

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from category_common_v82 import (ROOT, category_vocabulary, file_hash, instance_seed,
                                 load_cloud, normalized_category, run_lock)
from category_v10.data import assert_disjoint, scene_names
from pointcloud_classifier_v82 import prepare_points
from run_storage import write_json


def load_manifests(directory):
    manifests = {s: json.loads((directory/f'{s}.json').read_text())
                 for s in ('train', 'dev', 'eval')}
    official_train = set(scene_names(ROOT/'data/scannet/scannetv2_train.txt'))
    official_val = set(scene_names(ROOT/'data/scannet/scannetv2_val.txt'))
    assert_disjoint(official_train, official_val)
    train, dev, evaluation = (manifests[s] for s in ('train', 'dev', 'eval'))
    assert_disjoint(train['scenes'], dev['scenes'])
    if set(train['scenes']) | set(dev['scenes']) != official_train:
        raise ValueError('Train and internal-dev must cover the official training split')
    if not set(evaluation['scenes']) <= official_val:
        raise ValueError('Evaluation scenes must belong to official validation')
    vocabulary = list(category_vocabulary())
    mapping = {normalized_category(c): i for i, c in enumerate(vocabulary)}
    for split, manifest in manifests.items():
        if manifest['split'] != split or manifest['vocabulary'] != vocabulary:
            raise ValueError('Manifest split or vocabulary mismatch')
        if {r['scene'] for r in manifest['samples']} != set(manifest['scenes']):
            raise ValueError('Manifest scene membership mismatch')
        keys = [(r['scene'], r['bbox_id']) for r in manifest['samples']]
        if len(set(keys)) != len(keys):
            raise ValueError('Duplicate scene-instance key')
        for row in manifest['samples']:
            if row['label'] != mapping.get(normalized_category(row['category']), -1):
                raise ValueError('Category label index differs from the shared vocabulary')
            if split == 'train' and row['label'] < 0:
                raise ValueError('Training contains an unmapped category')
    # Different scan IDs of the same room and copied cloud content must stay together.
    for first, second in ((train, dev), (train, evaluation), (dev, evaluation)):
        assert_disjoint(first['scenes'], second['scenes'])
        if {s['cloud_sha256'] for s in first['sources']} & {
                s['cloud_sha256'] for s in second['sources']}:
            raise ValueError('Identical cloud contents occur in different splits')
    if len(evaluation['samples']) != 3413:
        raise ValueError('Expected the fixed 3413-object evaluation set')
    return manifests


def report(output, stage, **values):
    row = {'stage': stage, 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), **values}
    write_json(output/'status.json', row)
    print(json.dumps(row), flush=True)


def save_checkpoint(path, value):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_encoder(checkpoint):
    import openshape
    return openshape.G14(torch.load(checkpoint, map_location='cpu', weights_only=True)).cuda().eval()


def configure_tail(encoder):
    encoder.requires_grad_(False)
    for layer in encoder.ppat.transformer.layers[-4:]:
        layer.requires_grad_(True)
    encoder.proj.requires_grad_(True)
    # Keep frozen PointNet BatchNorm statistics fixed even during optimization.
    encoder.eval()


def object_points(row, seed):
    xyz, rgb, ids = load_cloud(row['cloud'])
    mask = ids == row['bbox_id']
    return prepare_points(xyz[mask], None if rgb is None else rgb[mask],
                          'openshape', instance_seed(seed, row['scene'], row['bbox_id']))


def smoke(args, manifests):
    seed_all(args.seed)
    encoder = load_encoder(args.checkpoint)
    configure_tail(encoder)
    head = nn.Linear(getattr(args, 'feature_dim', 1280), len(category_vocabulary())).cuda()
    rows = manifests['train']['samples'][:args.batch_size]
    x = torch.from_numpy(np.stack([object_points(r, args.seed) for r in rows])).cuda()
    y = torch.tensor([r['label'] for r in rows], device='cuda')
    frozen = encoder.ppat.sa.mlp_convs[0].weight.detach().clone()
    tail = encoder.ppat.transformer.layers[-1][1].fn.net[0].weight.detach().clone()
    params = [p for p in encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{'params': params, 'lr': 1e-4},
                                  {'params': head.parameters(), 'lr': 1e-3}])
    losses, gradients = [], []
    for step in range(10):
        torch.manual_seed(args.seed)  # Identical FPS anchors for this learning check.
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(F.normalize(encoder(x), dim=-1)), y)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite smoke loss')
        loss.backward()
        gradients.append(float(torch.nn.utils.clip_grad_norm_(params, 5.)))
        optimizer.step()
        losses.append(float(loss.detach()))
        print(json.dumps({'smoke_step': step+1, 'loss': losses[-1]}), flush=True)
    result = {'losses': losses, 'backbone_gradient_norms': gradients,
              'frozen_weights_unchanged': torch.equal(frozen, encoder.ppat.sa.mlp_convs[0].weight),
              'tail_weights_changed': not torch.equal(
                  tail, encoder.ppat.transformer.layers[-1][1].fn.net[0].weight),
              'train_instances': [{'scene': r['scene'], 'bbox_id': r['bbox_id']} for r in rows],
              'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated(),
              'evaluation_instances_used': 0}
    if not (losses[-1] < losses[0] and result['frozen_weights_unchanged']
            and result['tail_weights_changed'] and all(np.isfinite(gradients))
            and max(gradients) > 0):
        raise ValueError(f'Smoke learning/gradient check failed: {result}')
    write_json(args.output/'smoke.json', result)
    report(args.output, 'smoke_passed', initial_loss=losses[0], final_loss=losses[-1])


def make_bank(args, manifest):
    split = manifest['split']
    path = args.output/f'{split}_points.npy'
    done = args.output/f'{split}_points.json'
    shape = (len(manifest['samples']), 6, 10000)
    if done.exists():
        bank = np.load(path, mmap_mode='r')
        if bank.shape != shape or bank.dtype != np.float32:
            raise ValueError('Point cache has unexpected shape/dtype')
        return path
    required = int(np.prod(shape))*4
    if shutil.disk_usage(args.output).free < required + 3*1024**3:
        raise ValueError('Insufficient disk space for point bank plus 3 GiB reserve')
    bank = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=shape)
    groups = defaultdict(list)
    for i, row in enumerate(manifest['samples']):
        groups[row['scene']].append((i, row))
    sources = {r['scene']: r['cloud_sha256'] for r in manifest['sources']}
    for count, (scene, rows) in enumerate(groups.items(), 1):
        if file_hash(rows[0][1]['cloud']) != sources[scene]:
            raise ValueError(f'Source cloud changed: {scene}')
        xyz, rgb, ids = load_cloud(rows[0][1]['cloud'])
        for i, row in rows:
            mask = ids == row['bbox_id']
            bank[i] = prepare_points(xyz[mask], None if rgb is None else rgb[mask],
                                     'openshape', instance_seed(args.seed, scene, row['bbox_id']))
        if count % 20 == 0 or count == len(groups):
            bank.flush()
            report(args.output, f'prepare_{split}', scenes_done=count, scenes_total=len(groups))
    bank.flush()
    del bank
    write_json(done, {'shape': shape, 'dtype': 'float32', 'seed': args.seed,
                     'sha256': file_hash(path), 'geometry_only_input': True})
    return path


class PointBank(Dataset):
    def __init__(self, path, labels, augment=False):
        self.path, self.labels, self.augment = path, labels, augment
        self.bank = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if self.bank is None:
            self.bank = np.load(self.path, mmap_mode='r')
        points = np.array(self.bank[i], copy=True)
        if self.augment:
            angle = np.random.uniform(-np.pi, np.pi)
            c, s = np.cos(angle), np.sin(angle)
            points[:2] = np.array([[c, -s], [s, c]]) @ points[:2]
            points[:3] += np.clip(np.random.normal(0, .005, points[:3].shape), -.02, .02)
        return torch.from_numpy(points), self.labels[i]


def loader(args, path, manifest, training=False):
    dataset = PointBank(path, [r['label'] for r in manifest['samples']], training)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=training,
                      num_workers=2, pin_memory=True, persistent_workers=True)


def cache_features(args, encoder, path, manifest):
    saved = args.output/f"{manifest['split']}_features.npy"
    if saved.exists():
        return torch.from_numpy(np.load(saved)).cuda()
    values = []
    batches = loader(args, path, manifest)
    with torch.inference_mode():
        for i, (points, _) in enumerate(batches):
            features = encoder(points.cuda(non_blocking=True))
            if not torch.isfinite(features).all():
                raise ValueError('Nonfinite frozen features')
            values.append(F.normalize(features, dim=-1).cpu())
            if (i+1) % 50 == 0 or i+1 == len(batches):
                report(args.output, f"features_{manifest['split']}",
                       batches_done=i+1, batches_total=len(batches))
    features = torch.cat(values).numpy()
    temporary = saved.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        np.save(handle, features, allow_pickle=False)
    temporary.replace(saved)
    return torch.from_numpy(features).cuda()


def metrics(logits, manifest):
    labels = torch.tensor([r['label'] for r in manifest['samples']])
    logits = logits.cpu()
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite logits')
    order = logits.argsort(dim=-1, descending=True, stable=True)
    known = labels >= 0
    correct = order[:, 0] == labels
    by_class = defaultdict(list)
    for i, row in enumerate(manifest['samples']):
        by_class[normalized_category(row['category'])].append(bool(correct[i]))
    predicted = [category_vocabulary()[i] for i in order[:, 0].tolist()]
    return {'count': len(labels), 'correct': int(correct.sum()),
            'accuracy': float(correct.sum()/len(labels)),
            'exact_accuracy': sum(p == r['category'] for p, r in
                                  zip(predicted, manifest['samples']))/len(labels),
            'top5_accuracy': float((order[:, :5] == labels[:, None]).any(-1).float().mean()),
            'class_mean_accuracy': float(np.mean([np.mean(v) for v in by_class.values()])),
            'class_mean_accuracy_in_vocabulary': float(torch.stack([
                correct[labels == c].float().mean() for c in labels[known].unique()]).mean()),
            'targets_outside_vocabulary': int((~known).sum()),
            'category_loss': float(F.cross_entropy(logits[known], labels[known])),
            'errors': 0}


def fit_head(args, encoder, banks, manifests):
    path = args.output/'linear_best.pt'
    if (args.output/'linear_done.json').exists():
        return
    features = {s: cache_features(args, encoder, banks[s], manifests[s])
                for s in ('train', 'dev')}
    labels = torch.tensor([r['label'] for r in manifests['train']['samples']], device='cuda')
    head = nn.Linear(getattr(args, 'feature_dim', 1280), len(category_vocabulary())).cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=.003, weight_decay=.01)
    best, history = float('inf'), []
    for epoch in range(40):
        head.train()
        total = 0.
        for indices in torch.randperm(len(labels), device='cuda').split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(features['train'][indices]), labels[indices])
            loss.backward()
            optimizer.step()
            total += float(loss.detach())*len(indices)
        head.eval()
        with torch.inference_mode():
            dev = metrics(head(features['dev']), manifests['dev'])
        row = {'epoch': epoch+1, 'train_loss': total/len(labels), 'dev': dev}
        history.append(row)
        write_json(args.output/'linear_history.json', history)
        report(args.output, 'train_linear', **row)
        if dev['category_loss'] < best:
            best = dev['category_loss']
            save_checkpoint(path, {'head': head.state_dict(), 'epoch': epoch+1, 'dev': dev})
    write_json(args.output/'linear_done.json', {'epochs': 40, 'best_dev_loss': best})


def network_logits(args, encoder, head, bank, manifest):
    values = []
    encoder.eval()
    head.eval()
    torch.manual_seed(args.seed)
    with torch.inference_mode():
        for points, _ in loader(args, bank, manifest):
            values.append(head(F.normalize(encoder(points.cuda(non_blocking=True)), dim=-1)).cpu())
    return torch.cat(values)


def fit_tail(args, encoder, banks, manifests, tail_configurator=None):
    if (args.output/'finetune_done.json').exists():
        return
    (tail_configurator or configure_tail)(encoder)
    head = nn.Linear(getattr(args, 'feature_dim', 1280), len(category_vocabulary())).cuda()
    head.load_state_dict(torch.load(args.output/'linear_best.pt', weights_only=True)['head'])
    parameters = [p for p in encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{'params': parameters, 'lr': 1e-5},
                                  {'params': head.parameters(), 'lr': 1e-3}], weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    history, best, start, stale = [], float('inf'), 0, 0
    last = args.output/'finetune_last.pt'
    if last.exists():
        saved = torch.load(last, map_location='cuda', weights_only=False)
        encoder.load_state_dict(saved['encoder'])
        head.load_state_dict(saved['head'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        history, best, start, stale = saved['history'], saved['best'], saved['epoch'], saved['stale']
    batches = loader(args, banks['train'], manifests['train'], training=True)
    for epoch in range(start, args.epochs):
        seed_all(args.seed+epoch)
        head.train()
        encoder.eval()
        optimizer.zero_grad(set_to_none=True)
        total, count = 0., 0
        for step, (points, labels) in enumerate(batches):
            logits = head(F.normalize(encoder(points.cuda(non_blocking=True)), dim=-1))
            loss = F.cross_entropy(logits, labels.cuda(non_blocking=True))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite finetuning loss')
            group_start = (step//args.accumulation)*args.accumulation
            group_size = min(args.accumulation, len(batches)-group_start)
            (loss/group_size).backward()
            if (step+1) % args.accumulation == 0 or step+1 == len(batches):
                torch.nn.utils.clip_grad_norm_(parameters+list(head.parameters()), 5.)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach())*len(labels)
            count += len(labels)
            if (step+1) % 100 == 0:
                report(args.output, 'train_finetune', epoch=epoch+1,
                       batches_done=step+1, batches_total=len(batches), train_loss=total/count)
        dev = metrics(network_logits(args, encoder, head, banks['dev'], manifests['dev']), manifests['dev'])
        row = {'epoch': epoch+1, 'train_loss': total/count, 'dev': dev}
        history.append(row)
        write_json(args.output/'finetune_history.json', history)
        report(args.output, 'validate_finetune', **row)
        if dev['category_loss'] < best:
            best, stale = dev['category_loss'], 0
            save_checkpoint(args.output/'finetune_best.pt',
                            {'encoder': encoder.state_dict(), 'head': head.state_dict(),
                             'epoch': epoch+1, 'dev': dev})
        else:
            stale += 1
        scheduler.step()
        save_checkpoint(last, {'encoder': encoder.state_dict(), 'head': head.state_dict(),
                               'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                               'epoch': epoch+1, 'history': history, 'best': best, 'stale': stale})
        if stale >= 5:
            break
    write_json(args.output/'finetune_done.json', {'epochs_completed': len(history), 'best_dev_loss': best})


def evaluate(args, banks, manifests, encoder_factory=None):
    manifest = manifests['eval']
    if 'eval' not in banks:
        banks['eval'] = make_bank(args, manifest)
    comparison = {}
    for mode in ('linear', 'finetune'):
        encoder = (encoder_factory or load_encoder)(args.checkpoint)
        saved = torch.load(args.output/f'{mode}_best.pt', map_location='cpu', weights_only=True)
        if mode == 'finetune':
            encoder.load_state_dict(saved['encoder'])
        head = nn.Linear(getattr(args, 'feature_dim', 1280), len(category_vocabulary())).cuda()
        head.load_state_dict(saved['head'])
        logits = network_logits(args, encoder, head, banks['eval'], manifest)
        summary = metrics(logits, manifest)
        order = logits.argsort(dim=-1, descending=True, stable=True)[:, :5].tolist()
        predictions = [{'scene': r['scene'], 'bbox_id': r['bbox_id'],
                        'predicted_object': category_vocabulary()[o[0]],
                        'top5': [category_vocabulary()[i] for i in o], 'status': 'completed'}
                       for r, o in zip(manifest['samples'], order)]
        write_json(args.output/f'{mode}_predictions.json', {'results': predictions})
        np.save(args.output/f'{mode}_logits.npy', logits.numpy(), allow_pickle=False)
        comparison[mode] = {'summary': summary, 'best_internal_dev_epoch': saved['epoch']}
        write_json(args.output/f'{mode}_results.json', comparison[mode])
        del encoder, head, saved
        torch.cuda.empty_cache()
    write_json(args.output/'comparison.json', comparison)
    report(args.output, 'completed', comparison=comparison)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifests', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(8)
    seed_all(args.seed)
    if not torch.cuda.is_available():
        raise ValueError('NVIDIA CUDA is required')
    manifests = load_manifests(args.manifests)
    import openshape
    config = {'version': 'pointbert-scannet-supervised-r1',
              'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'code_sha256': file_hash(__file__), 'checkpoint_sha256': file_hash(args.checkpoint),
              'manifest_sha256': {s: file_hash(args.manifests/f'{s}.json') for s in manifests},
              'support_sha256': {p.name: file_hash(p) for p in Path(openshape.__file__).parent.glob('*.py')},
              'preprocessing_sha256': file_hash(ROOT/'local/pointcloud_classifier_v82.py'),
              'vocabulary': list(category_vocabulary()),
              'counts': {s: len(m['samples']) for s, m in manifests.items()},
              'regime': 'supervised ScanNet adaptation; not zero-shot',
              'input': '10000 points, normalized XYZ + RGB/255, float32; no text/query inputs',
              'linear': 'normalized frozen 1280-D OpenShape embedding + learned 197-way linear head',
              'finetune': 'last four Transformer blocks, semantic projection and head; earlier layers and BN frozen',
              'augmentation': 'finetune train only: random Z rotation and clipped XYZ jitter',
              'selection': 'lowest internal-dev in-vocabulary cross-entropy; patience 5 in finetuning',
              'evaluation': 'fixed previously inspected 3413 NR3D objects from official ScanNet val; OOV counted wrong',
              'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name(0)}
    with run_lock(args.output):
        stored = args.output/'config.json'
        if stored.exists() and json.loads(stored.read_text()) != config:
            raise ValueError('Run configuration changed; choose a new output directory')
        write_json(stored, config)
        try:
            if args.smoke_only:
                smoke(args, manifests)
                return
            if (args.output/'comparison.json').exists():
                return
            banks = {s: make_bank(args, manifests[s]) for s in ('train', 'dev')}
            encoder = load_encoder(args.checkpoint)
            encoder.requires_grad_(False)
            fit_head(args, encoder, banks, manifests)
            fit_tail(args, encoder, banks, manifests)
            del encoder
            torch.cuda.empty_cache()
            evaluate(args, banks, manifests)
        except BaseException as exc:
            report(args.output, 'failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
