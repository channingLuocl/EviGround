"""Learning, checkpoint criterion, and development-only fusion selection checks."""
import unittest
import numpy as np
import torch
from torch.nn import functional as F
from train_scan_domain import make_head, choose, checkpoint_key


class TrainingTests(unittest.TestCase):
    def test_real_optimization_and_reload(self):
        torch.set_num_threads(1)
        text = torch.zeros(3, 512)
        features = torch.zeros(6, 512)
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        features[torch.arange(6), labels] = 1
        head = make_head(text)
        optimizer = torch.optim.AdamW(head.parameters(), lr=.1)
        initial = float(F.cross_entropy(head(features), labels).detach())
        for _ in range(20):
            optimizer.zero_grad()
            loss = F.cross_entropy(head(features), labels)
            loss.backward()
            optimizer.step()
        self.assertLess(float(loss.detach()), initial/3)
        other = make_head(text)
        other.load_state_dict(head.state_dict(), strict=True)
        torch.testing.assert_close(other(features), head(features))

    def test_checkpoint_prefers_accuracy_then_ce_then_earliest(self):
        rows = [{'epoch': e, 'dev': {'correct': c, 'known_CE': loss}}
                for e, c, loss in [(1, 8, .2), (2, 9, .5), (3, 9, .4), (4, 9, .4)]]
        self.assertEqual(min(rows, key=checkpoint_key)['epoch'], 3)

    def test_fusion_can_choose_supervised_rgb_and_preserves_oov(self):
        shape = np.array([[0., 3.], [3., 0.], [3., 0.]])
        rgb = np.array([[3., 0.], [0., 3.], [0., 3.]])
        labels = np.array([0, 1, -1])
        parts = {'base': shape, 'shape': np.stack([shape]*3), 'rgb': np.stack([rgb]*3), 'valid': np.ones(3, bool)}
        selection, grid = choose(parts, labels)
        self.assertEqual(selection['primary_method'], 'S2')
        self.assertEqual(selection['fusion']['S3']['total_correct'], 6)
        self.assertAlmostEqual(selection['fusion']['S3']['mean_accuracy'], 2/3)
        self.assertEqual(len(grid['S3']), 11)


if __name__ == '__main__':
    unittest.main()
