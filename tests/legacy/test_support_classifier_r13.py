import unittest

import torch
from torch.nn import functional as F

from eviground.experiments.support_classifier_r13 import SupportHead, SPECS, training_support, weighted_loss


class SupportTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(15)
        self.samples = ([{'label': 0, 'scene': 'scene0000_00'}]*8 +
                        [{'label': 0, 'scene': 'scene0000_01'}] +
                        [{'label': 0, 'scene': 'scene0001_00'}] +
                        [{'label': 1, 'scene': 'scene0002_00'}]*3)
        self.support = training_support(self.samples, 3)
        self.text = F.normalize(torch.randn(3, 12), dim=-1)
        self.features = F.normalize(torch.randn(13, 12), dim=-1)

    def test_room_weighting_reduces_concentration_without_changing_class_mass(self):
        s = self.support
        self.assertEqual(s['rooms'].tolist(), [2, 1, 0])
        factors = s['room_factor']
        for label in [0, 1]:
            torch.testing.assert_close(factors[s['labels']==label].sum(), s['counts'][label])
        torch.testing.assert_close(factors[:9], factors[:1].expand(9))
        self.assertGreater(float(factors[9]), float(factors[0]))
        self.assertLess(float(factors[:9].sum()/factors[9]), 9.)
        torch.testing.assert_close(factors[10:], torch.ones(3))

    def test_support_counts_rescans_as_one_room_and_never_reads_categories(self):
        changed = [{**r, 'category': 'deliberately irrelevant'} for r in self.samples]
        other = training_support(changed, 3)
        for name, value in self.support.items():
            torch.testing.assert_close(value, other[name], atol=0, rtol=0)
        self.assertEqual(float(other['gains'][2]), 0.)
        self.assertTrue(((other['gains']>=0)&(other['gains']<=1)).all())

    def test_all_arms_start_at_exact_pretrained_text_logits(self):
        for spec in SPECS:
            model = SupportHead(self.text, self.support['gains'], spec).eval()
            torch.testing.assert_close(model(self.features), 10*self.features @ self.text.T, atol=0, rtol=0)

    def test_absent_class_semantic_score_survives_actual_training(self):
        model = SupportHead(self.text, self.support['gains'], 'support_room')
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.01)
        before = (10*self.features @ self.text.T)[:, 2]
        initial = model.residual[-1].weight.detach().clone()
        for _ in range(12):
            optimizer.zero_grad()
            loss = weighted_loss(model(self.features), self.support['labels'],
                                 self.support['object_weight']*self.support['room_factor'])
            loss.backward(); optimizer.step()
        model.eval()
        torch.testing.assert_close(model(self.features)[:, 2], before, atol=0, rtol=0)
        self.assertFalse(torch.equal(initial, model.residual[-1].weight))
        self.assertIsNone(model.text.grad)

    def test_object_loss_matches_class_weighted_ce(self):
        logits = torch.randn(13, 3, requires_grad=True)
        labels = self.support['labels']
        class_weight = (self.support['counts']+1).rsqrt()
        class_weight /= class_weight.mean()
        torch.testing.assert_close(weighted_loss(logits, labels, self.support['object_weight']),
                                   F.cross_entropy(logits, labels, weight=class_weight))


if __name__ == '__main__':
    unittest.main()
