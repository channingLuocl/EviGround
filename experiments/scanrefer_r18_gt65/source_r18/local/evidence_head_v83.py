"""Trainable evidence calibration. Estimated states are not measured visibility."""
import torch
from torch import nn
from torch.nn import functional as F


MODES = ('global', 'attention', 'voting', 'evidence')


class EvidenceHead(nn.Module):
    def __init__(self, category_text, part_text, mode='evidence'):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f'Unknown ablation: {mode}')
        self.mode = mode
        self.register_buffer('category_text', F.normalize(category_text.float(), dim=-1))
        self.register_buffer('part_text', F.normalize(part_text.float(), dim=-1))
        dimension, categories = category_text.shape[1], category_text.shape[0]
        parts = len(part_text)
        self.global_adapter = nn.Linear(dimension, categories, bias=False)
        nn.init.zeros_(self.global_adapter.weight)
        if mode != 'global':
            # The fusion controls receive the SAME state supervision as evidence mode.
            self.state_head = nn.Sequential(nn.Linear(11, 32), nn.ReLU(), nn.Linear(32, 3))
            self.part_bias = nn.Parameter(torch.zeros(parts, 3))
        if mode in ('attention', 'voting'):
            outputs = 1 if mode == 'attention' else dimension
            self.reliability = nn.Sequential(nn.Linear(dimension + 7, 64), nn.ReLU(), nn.Linear(64, outputs))
            self.local_adapter = nn.Linear(dimension + 3*parts, categories, bias=False)
            nn.init.zeros_(self.local_adapter.weight)
        if mode == 'evidence':
            # max/mean/std part similarity + weighted layout[7] + global part similarity.
            self.part_relation = nn.Linear(parts, categories, bias=False)
            nn.init.zeros_(self.part_relation.weight)

    def combine_evidence(self, global_logits, probabilities):
        # Unknown contributes exactly zero; absent is a learned estimate, not a ray test.
        evidence = probabilities[..., 0] - probabilities[..., 1]
        return global_logits + self.part_relation(evidence)

    def forward(self, global_features, local_features, geometry):
        global_features = F.normalize(global_features, dim=-1)
        local_features = F.normalize(local_features, dim=-1)
        logits = 10 * global_features @ self.category_text.T + self.global_adapter(global_features)
        if self.mode == 'global':
            return {'logits': logits}
        similarities = local_features @ self.part_text.T
        weights = (10 * similarities).softmax(dim=1)
        layout = torch.einsum('bkp,bkg->bpg', weights, geometry)
        stats = torch.stack([similarities.max(dim=1).values, similarities.mean(dim=1),
                             similarities.std(dim=1, unbiased=False),
                             global_features @ self.part_text.T], dim=-1)
        state_logits = self.state_head(torch.cat([stats, layout], dim=-1)) + self.part_bias
        probabilities = state_logits.softmax(dim=-1)
        if self.mode in ('attention', 'voting'):
            raw = self.reliability(torch.cat([local_features, geometry], dim=-1))
            if self.mode == 'attention':
                weights = raw.softmax(dim=1)
            else:
                # Diagonal Gaussian product: precision-weighted latent mean, PSV-inspired.
                precision = F.softplus(raw) + 1e-4
                weights = precision / precision.sum(dim=1, keepdim=True)
            pooled = F.normalize((weights * local_features).sum(dim=1), dim=-1)
            # Generic fusion can use every state, including unknown, without our neutral constraint.
            logits = logits + self.local_adapter(torch.cat([pooled, probabilities.flatten(1)], dim=-1))
        else:
            logits = self.combine_evidence(logits, probabilities)
        return {'logits': logits,
                'state_logits': state_logits, 'state_probabilities': probabilities}


def training_loss(outputs, labels, states):
    category_loss = F.cross_entropy(outputs['logits'], labels)
    state_loss = category_loss.new_zeros(())
    valid = states >= 0
    if 'state_logits' in outputs and valid.any():
        state_loss = F.cross_entropy(outputs['state_logits'][valid], states[valid])
    return category_loss + state_loss, category_loss, state_loss
