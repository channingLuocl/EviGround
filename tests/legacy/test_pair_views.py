import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from seeground.experiments.train_pair_views import Gate, choose, fit_head, gate_features, prior, room_folds, standardize
from seeground.experiments.audit_pair_views import features as numpy_features


class PairViewTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_rooms_and_rescans_stay_together_in_both_layers(self):
        rows = [{'scene': f'scene{i:04d}_{scan:02d}'} for i in range(30) for scan in range(3)]
        outer = room_folds(rows, 'outer')
        for salt in ('inner0', 'inner1', 'inner2'):
            inner = room_folds(rows, salt)
            for room in range(30):
                self.assertEqual(len(set(outer[3*room:3*room+3])), 1)
                self.assertEqual(len(set(inner[3*room:3*room+3])), 1)
            for f in range(3):
                train = {r['scene'].split('_')[0] for i, r in enumerate(rows) if outer[i] != f}
                test = {r['scene'].split('_')[0] for i, r in enumerate(rows) if outer[i] == f}
                self.assertFalse(train & test)

    def test_gate_initially_equals_fixed_fusion_and_preserves_agreement(self):
        x = torch.randn(12, 16)
        disagreement = torch.tensor([True, False]*6)
        for weight in (0., .4, 1.):
            gate = Gate(x.mean(0), x.std(0).clamp_min(1e-6), weight)
            torch.testing.assert_close(gate(x, disagreement), torch.full((12,), weight), rtol=0, atol=0)
            self.assertEqual(sum(p.numel() for p in gate.parameters()), 289)
            with torch.no_grad():
                gate.network[-1].bias.fill_(2.)
            torch.testing.assert_close(gate(x, disagreement)[~disagreement], torch.full((6,), weight))

    def test_features_are_target_free_and_invariant_to_view_order(self):
        b, r, v = torch.randn(10, 8), torch.randn(10, 8), torch.randn(10, 4, 8)
        for mode in ('confidence', 'view_quality', 'class_pair'):
            a, disagreement = gate_features(b, r, v, mode)
            z, _ = gate_features(b, r, v[:, [2, 0, 3, 1]], mode)
            self.assertEqual(a.shape, (10, 16))
            self.assertTrue(torch.isfinite(a).all())
            torch.testing.assert_close(a, z, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(disagreement, b.argmax(1) != r.argmax(1))

    def test_selection_preserves_macro_constraint(self):
        base = torch.tensor([[3., 0.], [3., 0.], [0., 3.]])
        rgb = torch.tensor([[0., 3.], [3., 0.], [3., 0.]])
        labels = torch.tensor([0, 0, 1])
        chosen, grid = choose(standardize(base), {'ce': [rgb]*3}, labels, prior(labels, 2), [('ce', 0.)])
        self.assertEqual(chosen['total_correct'], 9)
        self.assertEqual(chosen['mean_macro'], 1.)
        self.assertEqual(len(grid), 11)

    def test_independent_numpy_features_match_all_three_gate_variants(self):
        b, r, v = torch.randn(12, 197), torch.randn(12, 197), torch.randn(12, 4, 197)
        for name, mode in zip(('C', 'D', 'E'), ('confidence', 'view_quality', 'class_pair')):
            actual, disagreement = gate_features(b, r, v, mode)
            expected, other = numpy_features(b.numpy().astype(float), r.numpy().astype(float), v.numpy().astype(float), name)
            np.testing.assert_allclose(actual.numpy(), expected, atol=1e-5, rtol=1e-5)
            np.testing.assert_array_equal(disagreement.numpy(), other)

    def test_expert_learns_and_saves_last_epoch_without_validation(self):
        x = torch.eye(4).repeat(8, 1)
        labels = torch.arange(4).repeat(8)
        text = -torch.eye(4)
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'head.pt'
            w, b = fit_head(x, labels, text, 'balanced', 42, p)
            history = __import__('json').loads(p.with_suffix('.history.json').read_text())
            self.assertEqual(len(history), 40)
            self.assertLess(history[-1], history[0])
            self.assertFalse(torch.equal(w, 10*text))
            self.assertEqual(torch.load(p, weights_only=True)['epochs'], 40)
            self.assertFalse(w.requires_grad or b.requires_grad)


if __name__ == '__main__':
    unittest.main()
