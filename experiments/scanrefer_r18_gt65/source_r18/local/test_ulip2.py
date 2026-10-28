"""Check ULIP-2 frame, semantic space, projection and supervised freeze range."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch
from torch import nn

from category_common_v82 import category_vocabulary
from pointcloud_classifier_v82 import PROMPT, prepare_points
from train_ulip2_scannet import configure_tail
from ulip2_classifier import (TEXT_MODEL, TEXT_PRETRAINED, TEXT_REVISION, TEXT_SHA256,
                              ULIP2Encoder, encoder_state, load_ulip2_text,
                              prepare_ulip2_points, to_ulip2_frame)


class ULIP2Tests(unittest.TestCase):
    def test_frame_and_rgb_match_cached_inputs(self):
        rng = np.random.default_rng(17)
        xyz = rng.normal(size=(100, 3)).astype(np.float32)
        rgb = rng.integers(0, 256, size=(100, 3), dtype=np.uint8)
        source = prepare_points(xyz, rgb, 'openshape', 42).T
        actual = prepare_ulip2_points(xyz, rgb, 42)
        np.testing.assert_array_equal(actual[:, :3], source[:, [0, 2, 1]] * [1, 1, -1])
        np.testing.assert_array_equal(actual[:, 3:], source[:, 3:])
        np.testing.assert_array_equal(actual, to_ulip2_frame(torch.from_numpy(source.T[None]))[0])

    def test_frame_preserves_gradients(self):
        points = torch.randn(2, 6, 20, requires_grad=True)
        to_ulip2_frame(points).sum().backward()
        expected = torch.ones_like(points)
        expected[:, 1] = -1
        torch.testing.assert_close(points.grad, expected)

    def test_learned_projection_is_used(self):
        class ConstantPoint(nn.Module):
            def forward(self, points):
                return torch.ones(len(points), 768)
        encoder = ULIP2Encoder(ConstantPoint())
        with torch.no_grad():
            encoder.pc_projection.fill_(.5)
        torch.testing.assert_close(encoder(torch.zeros(2, 6, 20)), torch.full((2, 1280), 384.))

    def test_only_last_four_norm_and_projection_train(self):
        point = nn.Module()
        point.encoder = nn.Sequential(nn.Linear(6, 8), nn.BatchNorm1d(8))
        point.blocks = nn.Module()
        point.blocks.blocks = nn.ModuleList([nn.Linear(8, 8) for _ in range(18)])
        point.norm = nn.LayerNorm(8)
        point.pos_embed = nn.Linear(3, 8)
        encoder = ULIP2Encoder(point)
        configure_tail(encoder)
        for name, parameter in encoder.named_parameters():
            expected = (name == 'pc_projection' or name.startswith('point_encoder.norm.')
                        or any(name.startswith(f'point_encoder.blocks.blocks.{i}.') for i in range(14, 18)))
            self.assertEqual(parameter.requires_grad, expected, name)
        self.assertFalse(point.encoder[1].training)

    def test_checkpoint_projection_shape_and_namespace(self):
        state = {'module.pc_projection': torch.zeros(768, 1280),
                 'module.point_encoder.cls_token': torch.zeros(1, 1, 384),
                 'module.logit_scale': torch.zeros(())}
        result = encoder_state({'state_dict': state})
        self.assertEqual(set(result), {'pc_projection', 'point_encoder.cls_token'})
        state['module.pc_projection'] = torch.zeros(768, 1024)
        with self.assertRaises(ValueError):
            encoder_state({'state_dict': state})

    def test_incompatible_text_space_is_rejected(self):
        metadata = dict(vocabulary=list(category_vocabulary()), prompt=PROMPT,
                        text_model=TEXT_MODEL, text_pretrained=TEXT_PRETRAINED,
                        text_revision=TEXT_REVISION, text_checkpoint_sha256=TEXT_SHA256,
                        strict_load=True)
        with TemporaryDirectory() as directory:
            path = Path(directory)/'text.npz'
            np.savez(path, embeddings=np.ones((197, 1280)), metadata=json.dumps(metadata))
            self.assertEqual(load_ulip2_text(path)[0].shape, (197, 1280))
            metadata['text_model'] = 'EVA02-E-14-plus'
            np.savez(path, embeddings=np.ones((197, 1280)), metadata=json.dumps(metadata))
            with self.assertRaises(ValueError):
                load_ulip2_text(path)


if __name__ == '__main__':
    unittest.main()
