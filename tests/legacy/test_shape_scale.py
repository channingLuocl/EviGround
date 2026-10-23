"""Behavioral checks for scale residuals, selection and scoring."""
import unittest

import numpy as np
import torch

from seeground.geometry.shape_scale import (ScaleHead, bootstrap, choose_weights, combine, geometry_features,
                         grid, score)


class ScaleExperimentTests(unittest.TestCase):
    def test_geometry_scale_and_position(self):
        boxes = np.array([[1, 2, 3, .5, 1, 2], [4, 5, 6, 0, .2, 1.]])
        moved = boxes.copy()
        moved[:, :3] += 100
        np.testing.assert_array_equal(geometry_features(boxes, 'size'), geometry_features(moved, 'size'))
        swapped = boxes.copy()
        swapped[:, [3, 4]] = swapped[:, [4, 3]]
        np.testing.assert_array_equal(geometry_features(boxes, 'size'), geometry_features(swapped, 'size'))
        scaled = boxes[:1].copy()
        scaled[:, 3:] *= 3
        np.testing.assert_allclose(geometry_features(boxes[:1], 'ratio'), geometry_features(scaled, 'ratio'), atol=1e-6)
        np.testing.assert_allclose(geometry_features(scaled, 'size') - geometry_features(boxes[:1], 'size'), np.log(3), atol=1e-6)
        self.assertTrue(torch.isfinite(geometry_features(boxes, 'size')).all())
        boxes[0, 3] = -1
        with self.assertRaises(ValueError):
            geometry_features(boxes, 'size')

    def test_zero_initialization_gradient_and_training(self):
        torch.manual_seed(42)
        values = torch.randn(48, 3)
        labels = (values[:, 0] > 0).long()
        mean, std = values.mean(0), values.std(0, unbiased=False)
        head = ScaleHead(mean, std, 2)
        base = torch.zeros(48, 2)
        torch.testing.assert_close(base + head(values), base, rtol=0, atol=0)
        optimizer = torch.optim.AdamW(head.parameters(), lr=.03)
        initial = float(torch.nn.functional.cross_entropy(base, labels))
        for _ in range(40):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(base + head(values), labels)
            loss.backward()
            self.assertTrue(torch.isfinite(head.net[-1].weight.grad).all())
            optimizer.step()
        self.assertLess(float(loss.detach()), initial * .5)
        self.assertGreater(float(head.net[0].weight.grad.abs().sum()), 0)
        torch.testing.assert_close(head.mean, mean)
        torch.testing.assert_close(head.std, std)
        with torch.inference_mode():
            other = ScaleHead(torch.zeros(3), torch.ones(3), 2)
            other.load_state_dict(head.state_dict())
            torch.testing.assert_close(head(values), other(values))

    def test_grid_selection_and_full_vocabulary(self):
        tied = [{'weights': [1., 0.], 'mean_accuracy': .7},
                {'weights': [0., 0.], 'mean_accuracy': .7},
                {'weights': [0., 1.], 'mean_accuracy': .7}]
        self.assertEqual(choose_weights(tied)['weights'], [0., 0.])
        tied[2]['mean_accuracy'] = .8
        self.assertEqual(choose_weights(tied)['weights'], [0., 1.])
        self.assertEqual([len(grid(m)) for m in ('E0', 'E1', 'E2', 'E2-R', 'E3')], [1, 5, 5, 5, 25])
        base = np.arange(8, dtype=np.float32)[None]
        residual = np.zeros_like(base)
        residual[0, 0] = 100
        np.testing.assert_array_equal(combine(base, residual, residual, residual, 'E3', [0, 0]), base)
        self.assertEqual(int(combine(base, residual, residual, residual, 'E1', [1, 0]).argmax()), 0)

    def test_oov_scoring_and_bootstrap(self):
        manifest = {'vocabulary': ['chair', 'table'], 'samples': [
            {'category': 'chairs', 'label': 0}, {'category': 'table', 'label': 1},
            {'category': 'unlisted', 'label': -1}]}
        result = score(np.array([[3., 0.], [0., 3.], [3., 0.]], dtype=np.float32), manifest)
        self.assertEqual(result['count'], 3)
        self.assertEqual(result['correct'], 2)
        self.assertAlmostEqual(result['accuracy'], 2/3)
        self.assertAlmostEqual(result['class_mean_accuracy'], 2/3)
        self.assertEqual(result['targets_outside_vocabulary'], 1)
        self.assertEqual(bootstrap(np.zeros(3), ['scene1_00', 'scene1_01', 'scene2_00']), [0., 0.])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
