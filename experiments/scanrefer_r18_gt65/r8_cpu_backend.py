"""Inference-only CPU implementations of frozen PointNeXt CUDA operators.

No weights, architecture, point preparation, or category decisions are changed.
This backend must pass a historical score bridge before generating new inputs.
"""
import importlib
import sys
import types
from pathlib import Path

import numpy as np
import torch


def grouping(features, indices):
    flat = indices.long().reshape(indices.shape[0], -1)
    result = features.gather(2, flat[:, None].expand(-1, features.shape[1], -1))
    return result.reshape(indices.shape[0], features.shape[1], *indices.shape[1:])


def gather(features, indices):
    return features.gather(2, indices.long()[:, None].expand(-1, features.shape[1], -1))


def fps(xyz, count):
    # CUDA starts at index zero; each thread keeps its first maximum, then the
    # reduction keeps the lower thread on ties. Preserve that candidate order.
    points = xyz.detach().cpu().numpy()
    batch, size, _ = points.shape
    threads = min(1024, 2 ** (size.bit_length() - 1))
    priority = np.concatenate([np.arange(t, size, threads) for t in range(threads)])
    distance = np.full((batch, size), 1e10, dtype=np.float32)
    result = np.zeros((batch, count), dtype=np.int64)
    batches = np.arange(batch)
    old = np.zeros(batch, dtype=np.int64)
    for index in range(1, count):
        delta = points - points[batches, old][:, None]
        squared = delta[..., 0] ** 2 + delta[..., 1] ** 2 + delta[..., 2] ** 2
        np.minimum(distance, squared, out=distance)
        old = priority[distance[:, priority].argmax(1)]
        result[:, index] = old
    return torch.from_numpy(result).to(xyz.device)


def ball_query(radius, count, xyz, queries):
    # First support indices strictly inside the sphere; pad with first valid.
    support = xyz.detach().cpu().numpy()
    centers = queries.detach().cpu().numpy()
    result = np.zeros((*centers.shape[:2], count), dtype=np.int64)
    radius2 = np.float32(radius) ** np.float32(2)
    for batch in range(len(support)):
        delta = support[batch][None] - centers[batch][:, None]
        distance = delta[..., 0] ** 2 + delta[..., 1] ** 2 + delta[..., 2] ** 2
        for index, row in enumerate(distance < radius2):
            found = np.flatnonzero(row)[:count]
            if len(found):
                result[batch, index] = found[0]
                result[batch, index, :len(found)] = found
    return torch.from_numpy(result).to(xyz.device)


def load_encoder(root, checkpoint):
    """Load original classes with only inference imports and CPU ops selected."""
    root = Path(root).resolve()
    openpoints = root / 'models/pointnext/PointNeXt/openpoints'

    def package(name, path):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
        return module

    package('openpoints', openpoints)
    utils = package('openpoints.utils', openpoints / 'utils')
    utils.EasyConfig = importlib.import_module('openpoints.utils.config').EasyConfig
    checkpoint_utils = importlib.import_module('openpoints.utils.ckpt_util')
    for name in ('load_checkpoint', 'get_missing_parameters_message',
                 'get_unexpected_parameters_message', 'cal_model_parm_nums'):
        setattr(utils, name, getattr(checkpoint_utils, name))
    cpp = package('openpoints.cpp', openpoints / 'cpp')
    batch_ops = package('openpoints.cpp.pointnet2_batch', openpoints / 'cpp/pointnet2_batch')
    cpp.pointnet2_cuda = batch_ops.pointnet2_cuda = types.ModuleType('unused_cuda_ops')
    models = package('openpoints.models', openpoints / 'models')
    models.build_model_from_cfg = importlib.import_module('openpoints.models.build').build_model_from_cfg
    layers = package('openpoints.models.layers', openpoints / 'models/layers')
    conv = importlib.import_module('openpoints.models.layers.conv')
    for name in ('create_convblock1d', 'create_convblock2d', 'create_linearblock'):
        setattr(layers, name, getattr(conv, name))
    layers.create_act = importlib.import_module('openpoints.models.layers.activation').create_act
    group = importlib.import_module('openpoints.models.layers.group')
    group.grouping_operation, group.gather_operation, group.ball_query = grouping, gather, ball_query
    for name in ('create_grouper', 'get_aggregation_feautres'):
        setattr(layers, name, getattr(group, name))
    layers.furthest_point_sample = fps
    layers.random_sample = importlib.import_module('openpoints.models.layers.subsample').random_sample
    layers.three_interpolation = importlib.import_module('openpoints.models.layers.upsampling').three_interpolation
    layers.CHANNEL_MAP = importlib.import_module('openpoints.models.layers.local_aggregation').CHANNEL_MAP
    loss = package('openpoints.loss', openpoints / 'loss')
    loss.build_criterion_from_cfg = importlib.import_module('openpoints.loss.build').build_criterion_from_cfg
    package('openpoints.models.backbone', openpoints / 'models/backbone')
    importlib.import_module('openpoints.models.backbone.pointnext')
    package('openpoints.models.classification', openpoints / 'models/classification')
    importlib.import_module('openpoints.models.classification.cls_base')
    config = utils.EasyConfig()
    config.load(str(root / 'models/pointnext/pointnext-s.yaml'), recursive=True)
    encoder = models.build_model_from_cfg(config.model)

    # Exact PcdClassifier hierarchy and projection used by the frozen checkpoint.
    class PcdClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.obj_encoder = encoder
            self.pc_projection = torch.nn.Parameter(torch.empty(256, 512))

        def forward(self, points):
            return self.obj_encoder(points[..., :4]) @ self.pc_projection

    model = PcdClassifier()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    return model.eval().requires_grad_(False)
