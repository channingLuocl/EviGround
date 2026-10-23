"""Supervised heads over separate frozen geometry and image feature spaces."""
import torch
from torch import nn
from torch.nn import functional as F

SPECS = ('uni3d_linear', 'ulip2_linear', 'geometry_mlp', 'geometry_rgb_mlp',
         'shape_rgb_mlp_control', 'rgb_mlp_control')
MODALITIES = {
    'uni3d_linear': ('uni3d',),
    'ulip2_linear': ('ulip2',),
    'geometry_mlp': ('uni3d', 'ulip2'),
    'geometry_rgb_mlp': ('uni3d', 'ulip2', 'rgb'),
    'shape_rgb_mlp_control': ('shape', 'rgb'),
    'rgb_mlp_control': ('rgb',),
}


class GeometryHead(nn.Module):
    def __init__(self, spec, dimensions, classes, texts):
        super().__init__()
        self.spec = spec
        self.modalities = MODALITIES[spec]
        width = sum(dimensions[k] for k in self.modalities)
        if spec.endswith('_linear'):
            self.head = nn.Linear(width, classes)
            with torch.no_grad():
                # Each linear head starts in its own pretrained text space.
                self.head.weight.copy_(10 * texts[self.modalities[0]])
                self.head.bias.zero_()
        else:
            self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 256),
                                      nn.GELU(), nn.Dropout(.3), nn.Linear(256, classes))

    def forward(self, values):
        x = torch.cat([values[k] for k in self.modalities], dim=-1)
        if not self.spec.endswith('_linear'):
            x = F.dropout(x, .1, self.training)
        return self.head(x)


def class_weights(labels, classes):
    weights = (torch.bincount(labels, minlength=classes).float() + 1).rsqrt()
    return weights / weights.mean()


@torch.inference_mode()
def predict(model, values):
    model.eval()
    count = len(values[model.modalities[0]])
    return torch.cat([model({k: v[start:start+512] for k, v in values.items()})
                      for start in range(0, count, 512)]).cpu().numpy()
