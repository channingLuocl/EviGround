import copy
import unittest

import torch
from torch.nn import functional as F

from seeground.classifiers.pointnext_residual_r9 import ADAPTER, ResidualPointNext, losses, original_state
from seeground.classifiers.classifier_residual_r9 import choose
from tests.legacy.test_pointnext_color_r6 import toy


class ResidualTests(unittest.TestCase):
    def test_initial_adapter_preserves_function_and_every_original_state(self):
        source = toy()
        teacher = copy.deepcopy(source)
        model = ResidualPointNext(source, 'color_kd')
        x = torch.randn(4, 19, 7)
        expected = teacher.obj_encoder(x[..., :4]) @ teacher.pc_projection
        torch.testing.assert_close(model(x), expected, atol=0, rtol=0)
        for key, value in original_state(teacher.state_dict()).items():
            torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)
        self.assertTrue(all(p.requires_grad == name.startswith(ADAPTER) for name, p in model.named_parameters()))
        self.assertTrue(all(not m.training for m in model.modules()))

    def test_optimization_changes_adapter_and_keeps_all_original_parameters_exact(self):
        source = toy()
        expected = original_state(copy.deepcopy(source.state_dict()))
        model = ResidualPointNext(source, 'color_ce')
        x = torch.randn(4, 21, 7)
        text = F.normalize(torch.randn(5, 12), dim=-1)
        teacher = F.normalize(model(x).detach(), dim=-1)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        for _ in range(3):
            optimizer.zero_grad()
            loss = losses(model(x), teacher, text, torch.arange(4), torch.ones(5), 'color_ce')[0]
            loss.backward()
            optimizer.step()
        for key, value in expected.items():
            torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)
        self.assertGreater(float(dict(model.named_parameters())[ADAPTER + '2.weight'].detach().abs().sum()), 0)
        self.assertFalse(torch.equal(model(x), model(torch.cat((x[..., :4], x[..., 4:] + 1), -1))))

    def test_geometry_control_cannot_use_rgb_even_after_adapter_update(self):
        model = ResidualPointNext(toy(), 'geometry_kd')
        with torch.no_grad():
            dict(model.named_parameters())[ADAPTER + '2.weight'].normal_()
        x = torch.randn(3, 18, 7)
        changed = x.clone()
        changed[..., 4:] += 100
        torch.testing.assert_close(model(x), model(changed), atol=0, rtol=0)

    def test_distillation_is_zero_for_teacher_and_pulls_perturbed_logits_back(self):
        text = F.normalize(torch.randn(7, 12), dim=-1)
        teacher = F.normalize(torch.randn(8, 12), dim=-1)
        y = torch.arange(8) % 7
        same = losses(teacher, teacher, text, y, torch.ones(7), 'color_kd')
        self.assertLess(abs(float(same[2])), 1e-6)
        changed = losses(teacher + torch.randn_like(teacher), teacher, text, y, torch.ones(7), 'color_kd')
        self.assertGreater(float(changed[2]), 0)
        torch.testing.assert_close(changed[0], changed[1] + changed[2])

    def test_validation_objectives_preserve_declared_tradeoffs_and_reference_fallback(self):
        ref = dict(name='r8_M', epoch=0, base='r8_M', weight=0., parameters=0, mean_correct=314., mean_macro=.62)
        accuracy = {**ref, 'name': 'color_ce', 'parameters': 100, 'mean_correct': 320., 'mean_macro': .60}
        macro = {**ref, 'name': 'color_kd', 'parameters': 100, 'mean_correct': 314., 'mean_macro': .64}
        rows = [ref, accuracy, macro]
        self.assertEqual(choose(rows, ref, 'P'), accuracy)
        self.assertEqual(choose(rows, ref, 'B'), macro)
        self.assertEqual(choose(rows, ref, 'M'), macro)
        self.assertEqual(choose([ref, accuracy], ref, 'B'), ref)


if __name__ == '__main__':
    unittest.main()
