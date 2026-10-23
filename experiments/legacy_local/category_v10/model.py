"""Category-conditioned local aggregation; no semantic-part or visibility labels."""
import math

import torch
from torch import nn
from torch.nn import functional as F


MODES = ('frozen', 'global', 'attention', 'conditioned')


class CategoryHead(nn.Module):
    def __init__(self, category_text, mode='conditioned', width=64):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f'Unknown mode: {mode}')
        self.mode, self.width = mode, width
        self.register_buffer('category_text', F.normalize(category_text.float(), dim=-1))
        categories, dimension = category_text.shape
        if mode != 'frozen':
            self.global_adapter = nn.Linear(dimension, categories, bias=False)
            nn.init.zeros_(self.global_adapter.weight)
        if mode in ('attention', 'conditioned'):
            # The two local variants have exactly the same trainable parameters.
            self.key = nn.Linear(dimension + 7, width)
            self.query = nn.Linear(dimension, width, bias=False)
            self.local_adapter = nn.Sequential(nn.Linear(dimension, width), nn.GELU(),
                                               nn.Linear(width, dimension, bias=False))
            nn.init.zeros_(self.local_adapter[-1].weight)
            self.gate = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 1))
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.constant_(self.gate[-1].bias, -2.)

    def forward(self, global_features, local_features, geometry, patch_mask=None):
        global_features = F.normalize(global_features, dim=-1)
        local_features = F.normalize(local_features, dim=-1)
        base = 10 * global_features @ self.category_text.T
        logits = base if self.mode == 'frozen' else base + self.global_adapter(global_features)
        if self.mode in ('frozen', 'global'):
            return {'logits': logits, 'global_logits': logits}
        probabilities = logits.detach().softmax(-1)
        if self.mode == 'conditioned':
            # For each category, contrast its text feature with the current alternatives.
            # All categories can receive corrections; there is no hard Top-5 ceiling
            # and no insertion of the training target into an inference shortlist.
            other = probabilities @ self.category_text
            competitors = (other[:, None] - probabilities[..., None] * self.category_text)
            competitors = competitors / (1 - probabilities[..., None]).clamp_min(1e-6)
            query = self.category_text[None] - competitors
        else:
            # Ordinary global-query attention: every category shares the same patch weights.
            query = global_features[:, None]
        attention = torch.einsum('bkd,bcd->bkc',
                                self.key(torch.cat([local_features, geometry], -1)),
                                self.query(query)) / math.sqrt(self.width)
        if patch_mask is not None:
            if patch_mask.shape != local_features.shape[:2] or not patch_mask.any(-1).all():
                raise ValueError('Each object needs at least one retained local patch')
            attention = attention.masked_fill(~patch_mask[..., None], -torch.inf)
        attention = attention.softmax(dim=1)
        local = F.normalize(local_features + self.local_adapter(local_features), dim=-1)
        local_logits = 10 * local @ self.category_text.T
        evidence = (attention * local_logits).sum(1) - base
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(-1)
        entropy = entropy[:, None].expand_as(base) / math.log(base.shape[-1])
        gate = self.gate(torch.stack([logits / 10, entropy, evidence / 10], -1)).sigmoid()[..., 0]
        return {'logits': logits + gate * evidence, 'global_logits': logits,
                'patch_attention': attention, 'local_gate': gate}


def adaptation_loss(first, second, labels, consistency_weight=.1):
    """Same two-view CE/consistency objective for all trained ablations."""
    supervised = (F.cross_entropy(first['logits'], labels)
                  + F.cross_entropy(second['logits'], labels)) / 2
    consistency = F.kl_div(F.log_softmax(second['logits'], -1),
                           first['logits'].detach().softmax(-1), reduction='batchmean')
    return supervised + consistency_weight * consistency
