"""Behavioral checks for four-view aggregation and the isolated image tail."""
import unittest
import torch
from torch import nn
from torch.nn import functional as F

from clip_tail_r11 import ClipTail, objective, prefix_tokens


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mix = nn.Linear(8, 8)

    def forward(self, x):
        return x + self.mix(x.mean(1, keepdim=True))


class Visual(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList([Block() for _ in range(12)])
        self.transformer.batch_first = True
        self.ln_post = nn.LayerNorm(8)
        self.proj = nn.Parameter(torch.randn(8, 5))
        self.pool_type, self.attn_pool, self.final_ln_after_pool = 'tok', None, False

    def _embeds(self, x):
        return x

    def forward(self, x):
        for block in self.transformer.resblocks:
            x = block(x)
        return self.ln_post(x)[:, 0] @ self.proj


def fixture():
    torch.manual_seed(4)
    visual = Visual().eval()
    text = F.normalize(torch.randn(3, 5), dim=-1)
    images = torch.randn(8, 7, 8)
    tokens = prefix_tokens(visual, images).reshape(2, 4, 7, 8)
    return visual, text, images, tokens


def test_tail_reassembles_original_encoder_and_view_permutations():
    visual, text, images, tokens = fixture()
    model = ClipTail(visual, text)
    expected = F.normalize(F.normalize(visual(images), dim=-1).reshape(2, 4, 5).mean(1), dim=-1)
    torch.testing.assert_close(model.features(tokens), expected, atol=0, rtol=0)
    torch.testing.assert_close(model(tokens)[0], 10 * expected @ text.T)
    torch.testing.assert_close(model.features(tokens[:, [2, 0, 3, 1]]), expected)


def test_updates_only_the_selected_tail_without_mutating_prefix(spec):
    visual, text, images, tokens = fixture()
    before = {k: v.clone() for k, v in visual.state_dict().items()}
    model = ClipTail(visual, text)
    model.configure(spec)
    teacher = model.features(tokens).detach()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.05)
    initial = model.block.mix.weight.detach().clone()
    loss, ce, anchor = objective(*model(tokens), teacher, torch.tensor([0, 1]), torch.ones(3), spec)
    assert anchor.item() == 0 and loss.item() == ce.item()
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    assert torch.equal(initial, model.block.mix.weight) == (spec == 'frozen_head')
    assert all(torch.equal(before[k], v) for k, v in visual.state_dict().items())
    assert model.head.weight.grad.norm() > 0


def test_anchor_changes_gradient_and_never_uses_teacher_gradients():
    visual, text, _, tokens = fixture()
    model = ClipTail(visual, text)
    model.configure('tail_anchor')
    teacher = torch.randn(2, 5, requires_grad=True)
    logits, features = model(tokens)
    total, ce, anchor = objective(logits, features, teacher, torch.tensor([0, 1]), torch.ones(3), 'tail_anchor')
    torch.testing.assert_close(total, ce + anchor)
    total.backward()
    assert teacher.grad is None and model.projection.grad.norm() > 0


if __name__ == '__main__':
    suite = unittest.TestSuite([
        unittest.FunctionTestCase(test_tail_reassembles_original_encoder_and_view_permutations),
        unittest.FunctionTestCase(test_anchor_changes_gradient_and_never_uses_teacher_gradients),
    ] + [unittest.FunctionTestCase(
        lambda spec=spec: test_updates_only_the_selected_tail_without_mutating_prefix(spec),
        description=spec) for spec in ('frozen_head', 'tail_head', 'tail_anchor')])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
