"""CPU checks for the supervised Uni3D adapter and shared training path."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from seeground.classifiers.pointcloud_classifier_v82 import prepare_points
from seeground.classifiers.uni3d_classifier import prepare_uni3d_points
from seeground.experiments.train_uni3d_scannet import Uni3DEncoder, configure_tail, to_uni3d_frame
import seeground.experiments.train_pointbert_scannet as training


class Uni3DTrainingTests(unittest.TestCase):
    def test_cached_frame_matches_zero_shot_preprocessing(self):
        rng = np.random.default_rng(9)
        xyz = rng.normal(size=(90, 3)).astype(np.float32)
        rgb = rng.integers(0, 256, size=(90, 3), dtype=np.uint8)
        cache = torch.from_numpy(prepare_points(xyz, rgb, 'openshape', 42)[None])
        actual = to_uni3d_frame(cache).numpy()[0]
        np.testing.assert_array_equal(actual, prepare_uni3d_points(xyz, rgb, 42))

    def test_adapter_preserves_rgb_and_tensor_gradient(self):
        points = torch.arange(60, dtype=torch.float32).reshape(2, 6, 5).requires_grad_()
        converted = to_uni3d_frame(points)
        torch.testing.assert_close(converted[:, :, 3:], points[:, 3:].transpose(1, 2))
        converted.sum().backward()
        expected = torch.ones_like(points)
        expected[:, 1] = -1
        torch.testing.assert_close(points.grad, expected)

    def test_only_tail_projection_and_final_norms_train(self):
        point = nn.Module()
        point.encoder = nn.Sequential(nn.Linear(6, 8), nn.BatchNorm1d(8))
        point.visual = nn.Module()
        point.visual.blocks = nn.ModuleList([nn.Linear(8, 8) for _ in range(12)])
        point.visual.norm = nn.LayerNorm(8)
        point.visual.fc_norm = nn.Identity()
        point.trans2embed = nn.Linear(8, 16)
        point.pos_embed = nn.Linear(3, 8)
        encoder = Uni3DEncoder(point)
        configure_tail(encoder)
        for name, parameter in encoder.named_parameters():
            expected = (any(name.startswith(f'point_encoder.visual.blocks.{i}.') for i in range(8, 12))
                        or name.startswith('point_encoder.visual.norm.')
                        or name.startswith('point_encoder.trans2embed.'))
            self.assertEqual(parameter.requires_grad, expected, name)
        self.assertFalse(point.encoder[1].training)

    def test_shared_head_supports_uni3d_and_preserves_pointbert_default(self):
        vocabulary = training.category_vocabulary()
        manifest = {'samples': [{'label': i % 2, 'category': vocabulary[i % 2]} for i in range(4)]}
        tensor_factory, randperm_factory = torch.tensor, torch.randperm
        for dimension in (1024, 1280):
            with self.subTest(dimension=dimension), TemporaryDirectory() as directory:
                args = SimpleNamespace(output=Path(directory))
                if dimension == 1024:
                    args.feature_dim = dimension
                features = torch.nn.functional.normalize(torch.randn(4, dimension), dim=-1)
                with patch.object(training, 'cache_features', return_value=features), \
                     patch.object(training, 'report'), \
                     patch.object(torch.Tensor, 'cuda', lambda tensor, **kwargs: tensor), \
                     patch.object(nn.Module, 'cuda', lambda module: module), \
                     patch.object(torch, 'tensor', lambda data, **kwargs: tensor_factory(data)), \
                     patch.object(torch, 'randperm', lambda n, **kwargs: randperm_factory(n)):
                    training.fit_head(args, None, {'train': None, 'dev': None},
                                      {'train': manifest, 'dev': manifest})
                saved = torch.load(args.output/'linear_best.pt', weights_only=True)
                self.assertEqual(saved['head']['weight'].shape, (197, dimension))
                self.assertEqual(len(json.loads((args.output/'linear_history.json').read_text())), 40)

    def test_metrics_retain_oov_denominator(self):
        vocabulary = training.category_vocabulary()
        manifest = {'samples': [{'label': 0, 'category': vocabulary[0]},
                                {'label': -1, 'category': 'not in vocabulary'}]}
        scores = torch.zeros(2, len(vocabulary))
        scores[:, 0] = 5
        result = training.metrics(scores, manifest)
        self.assertEqual(result['correct'], 1)
        self.assertEqual(result['count'], 2)
        self.assertEqual(result['accuracy'], .5)
        self.assertEqual(result['class_mean_accuracy'], .5)
        self.assertEqual(result['targets_outside_vocabulary'], 1)


if __name__ == '__main__':
    unittest.main()
