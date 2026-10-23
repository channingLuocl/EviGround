"""Training-support-scaled semantic residuals and within-class room weighting."""
from collections import Counter

import torch
from torch import nn
from torch.nn import functional as F

SPECS = ('residual_object', 'residual_room', 'support_object', 'support_room')


def training_support(samples, classes):
    labels = torch.tensor([r['label'] for r in samples], dtype=torch.long)
    assert len(labels) and labels.min() >= 0 and labels.max() < classes
    cells = Counter((r['label'], r['scene'].split('_')[0]) for r in samples)
    counts = torch.bincount(labels, minlength=classes).float()
    rooms = torch.zeros(classes)
    root_mass = torch.zeros(classes)
    for (label, _), count in cells.items():
        rooms[label] += 1
        root_mass[label] += count**.5
    # Normalize within each class, preserving its total object-based loss weight.
    room_factor = torch.tensor([counts[r['label']] /
        (root_mass[r['label']] * cells[(r['label'], r['scene'].split('_')[0])]**.5)
        for r in samples])
    gains = (counts/(counts+16) * rooms/(rooms+4)).sqrt()
    class_weight = (counts+1).rsqrt()
    class_weight /= class_weight.mean()
    return {'labels': labels, 'counts': counts, 'rooms': rooms, 'gains': gains,
            'object_weight': class_weight[labels], 'room_factor': room_factor}


class SupportHead(nn.Module):
    def __init__(self, text, gains, spec):
        super().__init__()
        assert spec in SPECS
        self.register_buffer('text', text.detach().clone())
        self.register_buffer('gains', gains.detach().clone() if spec.startswith('support') else torch.ones_like(gains))
        self.residual = nn.Sequential(nn.LayerNorm(text.shape[1]), nn.Linear(text.shape[1], 256),
            nn.GELU(), nn.Dropout(.3), nn.Linear(256, len(text)))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, features):
        x = F.dropout(features, .1, self.training)
        return 10*features @ self.text.T + self.gains*self.residual(x)


def weighted_loss(logits, labels, weights):
    return (F.cross_entropy(logits, labels, reduction='none')*weights).sum()/weights.sum()
