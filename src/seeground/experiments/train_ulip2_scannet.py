"""Adapt official ULIP-2 on the same ScanNet manifests and immutable point banks."""
from seeground.paths import source_file
import argparse
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
from seeground.experiments.train_uni3d_scannet import validated_banks
from seeground.classifiers.ulip2_classifier import load_encoder, load_ulip2_text


def configure_tail(encoder):
    encoder.requires_grad_(False)
    for block in encoder.point_encoder.blocks.blocks[-4:]:
        block.requires_grad_(True)
    encoder.point_encoder.norm.requires_grad_(True)
    encoder.pc_projection.requires_grad_(True)
    encoder.eval()  # Keep frozen local PointNet BN statistics fixed.


def smoke(args, encoder, banks, manifests):
    configure_tail(encoder)
    head = nn.Linear(args.feature_dim, len(category_vocabulary())).cuda()
    bank = np.load(banks['train'], mmap_mode='r')
    points = torch.from_numpy(np.array(bank[:args.batch_size], copy=True)).cuda()
    labels = torch.tensor([r['label'] for r in manifests['train']['samples'][:args.batch_size]],
                          device='cuda')
    local = encoder.point_encoder.encoder.first_conv[0].weight
    tail = encoder.point_encoder.blocks.blocks[-1].mlp.fc2.weight
    bn = encoder.point_encoder.encoder.first_conv[1]
    frozen_before, tail_before = local.detach().clone(), tail.detach().clone()
    bn_before = bn.running_mean.detach().clone()
    parameters = [p for p in encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{'params': parameters, 'lr': 1e-4},
                                  {'params': head.parameters(), 'lr': 1e-3}])
    losses, gradients = [], []
    for step in range(10):
        training.seed_all(args.seed)
        optimizer.zero_grad(set_to_none=True)
        features = encoder(points)
        if features.shape != (len(labels), 1280) or not torch.isfinite(features).all():
            raise ValueError('Invalid ULIP-2 training embedding')
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
        raise ValueError(f'ULIP-2 smoke learning/freeze check failed: {result}')
    write_json(args.output/'smoke.json', result)
    training.report(args.output, 'smoke_passed', initial_loss=losses[0], final_loss=losses[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifests', 'point-bank-dir', 'checkpoint', 'source-dir', 'text-features', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    args.feature_dim = 1280
    torch.set_num_threads(8)
    training.seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise ValueError('NVIDIA CUDA is required')
    load_ulip2_text(args.text_features)
    manifests = training.load_manifests(args.manifests)
    config = {
        'version': 'ulip2-pointbert-scannet-supervised-r1',
        'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'code_sha256': file_hash(__file__),
        'dependencies_sha256': {name: file_hash(source_file(name)) for name in (
            'train_pointbert_scannet.py', 'train_uni3d_scannet.py', 'ulip2_classifier.py',
            'pointcloud_classifier_v82.py', 'category_common_v82.py', 'category_v10/data.py', 'run_storage.py')},
        'checkpoint_sha256': file_hash(args.checkpoint),
        'source_manifest_sha256': file_hash(args.source_dir/'sources.json'),
        'text_features_sha256': file_hash(args.text_features),
        'manifest_sha256': {s: file_hash(args.manifests/f'{s}.json') for s in manifests},
        'vocabulary': list(category_vocabulary()),
        'counts': {s: len(m['samples']) for s, m in manifests.items()},
        'regime': 'supervised ScanNet adaptation; not zero-shot',
        'input': '10000 points, XYZ+RGB; Z-up to Y-up: (x,y,z)->(x,z,-y)',
        'linear': 'normalized frozen 1280-D official ULIP-2 embedding + learned 197-way linear head; 40 epochs',
        'finetune': 'last 4 of 18 PointBERT blocks, final norm, 768-to-1280 projection and linear head',
        'augmentation': 'finetune train only: random Z rotation and XYZ jitter before Y-up conversion',
        'selection': 'lowest internal-dev in-vocabulary cross-entropy; patience 5, max 20 finetune epochs',
        'evaluation': 'same fixed 3413 previously inspected objects; 157 OOV retained in denominator',
        'text': 'frozen text features recorded for provenance; learned head supplies supervised scores',
        'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name(0)}
    with run_lock(args.output):
        stored = args.output/'config.json'
        if stored.exists() and json.loads(stored.read_text()) != config:
            raise ValueError('Run configuration changed; choose a new output directory')
        write_json(stored, config)
        try:
            if (args.output/'comparison.json').exists():
                return
            banks = validated_banks(args, manifests)
            encoder = load_encoder(args.checkpoint, args.source_dir).cuda()
            if args.smoke_only:
                smoke(args, encoder, banks, manifests)
                return
            encoder.requires_grad_(False)
            training.fit_head(args, encoder, banks, manifests)
            training.fit_tail(args, encoder, banks, manifests, tail_configurator=configure_tail)
            del encoder
            torch.cuda.empty_cache()
            training.evaluate(args, banks, manifests,
                              encoder_factory=lambda _: load_encoder(args.checkpoint, args.source_dir).cuda())
        except BaseException as exc:
            training.report(args.output, 'failed', error=f'{type(exc).__name__}: {exc}',
                            traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
