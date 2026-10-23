"""Train the final CLIP image block while the first eleven blocks stay fixed."""
import copy

import torch
from torch import nn
from torch.nn import functional as F

SPECS = ('frozen_head', 'tail_head', 'tail_anchor')


@torch.no_grad()
def prefix_tokens(visual, images):
    assert not visual.training and len(visual.transformer.resblocks) == 12
    x = visual._embeds(images)
    if not visual.transformer.batch_first:
        x = x.transpose(0, 1).contiguous()
    for block in visual.transformer.resblocks[:-1]:
        x = block(x)
    if not visual.transformer.batch_first:
        x = x.transpose(0, 1)
    return x.contiguous()


class ClipTail(nn.Module):
    def __init__(self, visual, text):
        super().__init__()
        assert visual.pool_type == 'tok' and visual.attn_pool is None
        self.block = copy.deepcopy(visual.transformer.resblocks[-1])
        self.norm = copy.deepcopy(visual.ln_post)
        self.projection = nn.Parameter(visual.proj.detach().clone())
        self.batch_first = visual.transformer.batch_first
        self.norm_after_pool = visual.final_ln_after_pool
        self.head = nn.Linear(text.shape[1], len(text), device=text.device)
        with torch.no_grad():
            self.head.weight.copy_(10 * text)
            self.head.bias.zero_()
        self.configure('frozen_head')

    def configure(self, spec):
        assert spec in SPECS
        self.spec = spec
        self.requires_grad_(spec != 'frozen_head')
        self.head.requires_grad_(True)
        self.eval()  # CLIP has no training-time stochastic layer in this experiment.

    def features(self, tokens):
        assert tokens.ndim == 4 and tokens.shape[1] == 4
        batch, views, length, width = tokens.shape
        x = tokens.reshape(batch * views, length, width)
        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()
        x = self.block(x)
        if not self.batch_first:
            x = x.transpose(0, 1)
        pooled = self.norm(x[:, 0]) if self.norm_after_pool else self.norm(x)[:, 0]
        features = F.normalize(pooled @ self.projection, dim=-1)
        return F.normalize(features.reshape(batch, views, -1).mean(1), dim=-1)

    def forward(self, tokens):
        features = self.features(tokens)
        return self.head(features), features


def objective(logits, features, teacher, labels, weights, spec):
    ce = F.cross_entropy(logits, labels, weight=weights)
    anchor = (features - teacher.detach()).square().sum(-1).mean()
    return ce + (anchor if spec == 'tail_anchor' else 0), ce, anchor
