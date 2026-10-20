import unittest

import numpy as np
import torch
from torch.nn import functional as F

from audit_classifier_geometry_r12 import independent_head
from geometry_classifier_r12 import GeometryHead, SPECS, class_weights


class GeometryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(13)
        self.dimensions = {'uni3d': 12, 'ulip2': 15, 'shape': 8, 'rgb': 8}
        self.texts = {k: F.normalize(torch.randn(5, self.dimensions[k]), dim=-1) for k in ('uni3d', 'ulip2')}
        self.values = {k: F.normalize(torch.randn(20, d), dim=-1) for k, d in self.dimensions.items()}

    def test_linear_models_use_their_matching_text_space(self):
        for name in ('uni3d', 'ulip2'):
            model = GeometryHead(name+'_linear', self.dimensions, 5, self.texts).eval()
            torch.testing.assert_close(model(self.values), 10*self.values[name] @ self.texts[name].T)
            altered = {k: (v if k == name else torch.zeros_like(v)) for k, v in self.values.items()}
            torch.testing.assert_close(model(self.values), model(altered), atol=0, rtol=0)

    def test_rgb_control_is_unaffected_by_geometry(self):
        model = GeometryHead('rgb_mlp_control', self.dimensions, 5, self.texts).eval()
        altered = {k: (v if k == 'rgb' else torch.randn_like(v)*100) for k, v in self.values.items()}
        torch.testing.assert_close(model(self.values), model(altered), atol=0, rtol=0)

    def test_independent_numpy_forward_after_training(self):
        labels = torch.arange(20) % 5
        for spec in SPECS:
            model = GeometryHead(spec, self.dimensions, 5, self.texts)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
            for _ in range(3):
                optimizer.zero_grad()
                F.cross_entropy(model(self.values), labels).backward(); optimizer.step()
            model.eval()
            record = {'spec': spec, 'state_dict': model.state_dict()}
            expected = independent_head(record, {k: v.numpy() for k, v in self.values.items()})
            actual = model(self.values).detach().numpy()
            np.testing.assert_allclose(actual, expected, atol=5e-6, rtol=2e-5)
            np.testing.assert_array_equal(actual.argmax(1), expected.argmax(1))

    def test_heads_learn_a_separable_feature_problem(self):
        values = {k: F.one_hot(torch.arange(30)%5, d).float() for k, d in self.dimensions.items()}
        labels = torch.arange(30)%5
        for spec in SPECS:
            torch.manual_seed(42)
            model = GeometryHead(spec, self.dimensions, 5, self.texts)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.02)
            history = []
            # Random text initialization can strongly oppose these artificial labels.
            # Use enough updates to test learnability (real fits use 4,520 updates).
            for _ in range(200):
                optimizer.zero_grad(); loss = F.cross_entropy(model(values), labels)
                loss.backward(); optimizer.step(); history.append(float(loss.detach()))
            self.assertLess(history[-1], history[0])
            self.assertTrue(torch.isfinite(torch.tensor(history)).all())
            model.eval(); self.assertGreaterEqual(float((model(values).argmax(1)==labels).float().mean()), .8, spec)

    def test_class_weights_use_only_supplied_training_frequencies(self):
        weights = class_weights(torch.tensor([0]*9+[1]), 3)
        self.assertGreater(weights[1], weights[0])
        torch.testing.assert_close(weights.mean(), torch.tensor(1.))
        self.assertTrue(torch.isfinite(weights).all())


if __name__ == '__main__':
    unittest.main()
