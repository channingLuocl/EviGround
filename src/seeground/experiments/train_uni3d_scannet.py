"""Adapt Uni3D-B on the same ScanNet splits used for PointBERT."""
from seeground.paths import source_file
import argparse
import importlib.metadata
import json
from pathlib import Path
import traceback

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from seeground.categories.category_common_v82 import ROOT, category_vocabulary, file_hash, run_lock
from seeground.evaluation.run_storage import write_json
import seeground.experiments.train_pointbert_scannet as training
from seeground.classifiers.uni3d_classifier import Uni3DClassifier


def to_uni3d_frame(points):
    """Convert cached B,6,N ScanNet points into B,N,6 Uni3D inputs."""
    points = points.transpose(1, 2)
    return torch.cat((points[:, :, 0:1], points[:, :, 2:3],
                      -points[:, :, 1:2], points[:, :, 3:]), dim=-1)


class Uni3DEncoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.point_encoder = encoder

    def forward(self, points):
        points = to_uni3d_frame(points)
        return self.point_encoder(points[:, :, :3].contiguous(),
                                  points[:, :, 3:].contiguous())


def load_encoder(args):
    # Reuse the strict source/checkpoint validation and exact zero-shot backbone.
    classifier = Uni3DClassifier(args.checkpoint, args.text_features, args.source_dir)
    encoder = Uni3DEncoder(classifier.encoder).cuda().eval()
    del classifier
    return encoder


def configure_tail(encoder):
    encoder.requires_grad_(False)
    point_encoder = encoder.point_encoder
    for block in point_encoder.visual.blocks[-4:]:
        block.requires_grad_(True)
    for module in (point_encoder.visual.norm, point_encoder.visual.fc_norm,
                   point_encoder.trans2embed):
        module.requires_grad_(True)
    # Keep frozen local PointNet BatchNorm statistics fixed.
    encoder.eval()


def validated_banks(args, manifests):
    """Reuse immutable, hash-checked Z-up point caches from the previous run."""
    config = json.loads((args.point_bank_dir/'config.json').read_text())
    if (config['args']['seed'] != args.seed
            or config['vocabulary'] != list(category_vocabulary())
            or config['preprocessing_sha256'] != file_hash(source_file('pointcloud_classifier_v82.py'))):
        raise ValueError('Point-bank seed, vocabulary or preprocessing mismatch')
    banks = {}
    for split, manifest in manifests.items():
        if config['manifest_sha256'][split] != file_hash(args.manifests/f'{split}.json'):
            raise ValueError(f'Point-bank {split} manifest mismatch')
        path = args.point_bank_dir/f'{split}_points.npy'
        metadata = json.loads(path.with_suffix('.json').read_text())
        expected = (len(manifest['samples']), 6, 10000)
        bank = np.load(path, mmap_mode='r')
        if (tuple(metadata['shape']) != expected or bank.shape != expected
                or bank.dtype != np.float32 or metadata['seed'] != args.seed
                or metadata['sha256'] != file_hash(path)):
            raise ValueError(f'Point-bank {split} integrity mismatch')
        banks[split] = path
        training.report(args.output, 'validate_point_cache', split=split, count=len(bank))
    return banks


def smoke(args, encoder, banks, manifests):
    configure_tail(encoder)
    head = nn.Linear(args.feature_dim, len(category_vocabulary())).cuda()
    bank = np.load(banks['train'], mmap_mode='r')
    points = torch.from_numpy(np.array(bank[:args.batch_size], copy=True)).cuda()
    labels = torch.tensor([r['label'] for r in manifests['train']['samples'][:args.batch_size]],
                          device='cuda')
    local = encoder.point_encoder.encoder.first_conv[0].weight
    tail = encoder.point_encoder.visual.blocks[-1].mlp.fc2.weight
    frozen_before, tail_before = local.detach().clone(), tail.detach().clone()
    bn = encoder.point_encoder.encoder.first_conv[1]
    bn_before = bn.running_mean.detach().clone()
    parameters = [p for p in encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{'params': parameters, 'lr': 1e-4},
                                  {'params': head.parameters(), 'lr': 1e-3}])
    losses, gradients = [], []
    for step in range(10):
        training.seed_all(args.seed)
        optimizer.zero_grad(set_to_none=True)
        features = encoder(points)
        if features.shape != (len(labels), args.feature_dim) or not torch.isfinite(features).all():
            raise ValueError('Invalid Uni3D training embedding')
        loss = F.cross_entropy(head(F.normalize(features, dim=-1)), labels)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite smoke loss')
        loss.backward()
        gradients.append(float(torch.nn.utils.clip_grad_norm_(parameters, 5.)))
        optimizer.step()
        losses.append(float(loss.detach()))
        training.report(args.output, 'smoke_step', step=step+1, loss=losses[-1])
    result = {'losses': losses, 'backbone_gradient_norms': gradients,
              'frozen_weights_unchanged': torch.equal(frozen_before, local),
              'frozen_bn_statistics_unchanged': torch.equal(bn_before, bn.running_mean),
              'tail_weights_changed': not torch.equal(tail_before, tail),
              'trainable_backbone_parameters': sum(p.numel() for p in parameters),
              'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated(),
              'evaluation_instances_used_for_optimization': 0}
    if not (losses[-1] < losses[0] and result['frozen_weights_unchanged']
            and result['frozen_bn_statistics_unchanged'] and result['tail_weights_changed']
            and all(np.isfinite(gradients)) and max(gradients) > 0):
        raise ValueError(f'Uni3D smoke learning/freeze check failed: {result}')
    write_json(args.output/'smoke.json', result)
    training.report(args.output, 'smoke_passed', initial_loss=losses[0], final_loss=losses[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifests', type=Path, required=True)
    parser.add_argument('--point-bank-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--text-features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    args.feature_dim = 1024
    torch.set_num_threads(8)
    training.seed_all(args.seed)
    if not torch.cuda.is_available():
        raise ValueError('NVIDIA CUDA is required')
    manifests = training.load_manifests(args.manifests)
    config = {
        'version': 'uni3d-b-scannet-supervised-r1',
        'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'code_sha256': file_hash(__file__),
        'dependencies_sha256': {name: file_hash(source_file(name)) for name in (
            'train_pointbert_scannet.py', 'uni3d_classifier.py', 'pointcloud_classifier_v82.py',
            'category_common_v82.py', 'category_v10/data.py', 'run_storage.py')},
        'checkpoint_sha256': file_hash(args.checkpoint),
        'source_sha256': file_hash(args.source_dir/'point_encoder.py'),
        'text_features_sha256': file_hash(args.text_features),
        'manifest_sha256': {s: file_hash(args.manifests/f'{s}.json') for s in manifests},
        'vocabulary': list(category_vocabulary()),
        'counts': {s: len(m['samples']) for s, m in manifests.items()},
        'regime': 'supervised ScanNet adaptation; not zero-shot',
        'input': '10000 points, XYZ+RGB; Z-up to Y-up: (x,y,z)->(x,z,-y)',
        'linear': 'normalized frozen 1024-D Uni3D-B embedding + learned 197-way linear head; 40 epochs',
        'finetune': 'last 4 visual blocks, final norms, 768-to-1024 projection and linear head',
        'augmentation': 'finetune train only: random Z rotation and XYZ jitter before Y-up conversion',
        'selection': 'lowest internal-dev in-vocabulary cross-entropy; finetune patience 5, max 20 epochs',
        'evaluation': 'same fixed 3413 previously inspected objects; 157 OOV retained in denominator',
        'text': 'cached text features loaded for provenance validation only; learned head supplies scores',
        'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name(0),
        'packages': {p: importlib.metadata.version(p) for p in ('timm', 'open_clip_torch', 'pointnet2_ops')}}
    with run_lock(args.output):
        stored = args.output/'config.json'
        if stored.exists() and json.loads(stored.read_text()) != config:
            raise ValueError('Run configuration changed; choose a new output directory')
        write_json(stored, config)
        try:
            if (args.output/'comparison.json').exists():
                return
            banks = validated_banks(args, manifests)
            encoder = load_encoder(args)
            if args.smoke_only:
                smoke(args, encoder, banks, manifests)
                return
            encoder.requires_grad_(False)
            training.fit_head(args, encoder, banks, manifests)
            training.fit_tail(args, encoder, banks, manifests, tail_configurator=configure_tail)
            del encoder
            torch.cuda.empty_cache()
            training.evaluate(args, banks, manifests, encoder_factory=lambda _: load_encoder(args))
        except BaseException as exc:
            training.report(args.output, 'failed', error=f'{type(exc).__name__}: {exc}',
                            traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
