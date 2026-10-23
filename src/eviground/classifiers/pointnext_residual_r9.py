"""A zero-initialized input adapter with a completely frozen original PointNeXt."""
import torch
from torch import nn
from torch.nn import functional as F

STEM = 'obj_encoder.encoder.encoder.0.0.convs.0.0'
ADAPTER = STEM + '.adapter.'
SPECS = ('geometry_kd', 'color_ce', 'color_kd')


class ResidualStem(nn.Module):
    def __init__(self, base):
        super().__init__()
        assert isinstance(base, nn.Conv1d) and base.in_channels == 4
        self.base = base
        self.adapter = nn.Sequential(nn.Conv1d(7, 64, 1), nn.GELU(), nn.Conv1d(64, base.out_channels, 1))
        self.adapter.to(device=base.weight.device, dtype=base.weight.dtype)
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, x):
        return self.base(x[:, :4]) + self.adapter(x)


class ResidualPointNext(nn.Module):
    def __init__(self, original, spec):
        super().__init__()
        assert spec in SPECS
        self.spec = spec
        self.obj_encoder = original.obj_encoder
        self.pc_projection = original.pc_projection
        parent = self.obj_encoder.encoder.encoder[0][0].convs[0]
        parent[0] = ResidualStem(parent[0])
        self.requires_grad_(False)
        parent[0].adapter.requires_grad_(True)
        self.eval()  # Preserve the original BN statistics and original dropout behavior.

    def forward(self, points):
        assert points.shape[-1] == 7
        if self.spec == 'geometry_kd':
            points = torch.cat((points[..., :4], torch.zeros_like(points[..., 4:])), -1)
        return self.obj_encoder(points) @ self.pc_projection


def losses(embedding, teacher, text, labels, weights, spec):
    logits = 10 * F.normalize(embedding.float(), dim=-1) @ text.T
    ce = F.cross_entropy(logits, labels, weight=weights)
    target = 10 * teacher.detach() @ text.T
    kd = 4 * F.kl_div(F.log_softmax(logits / 2, dim=-1),
                      F.softmax(target / 2, dim=-1), reduction='batchmean')
    return ce + (0 if spec == 'color_ce' else kd), ce, kd


def original_state(state):
    """Map the original checkpoint into the wrapped stem, excluding new parameters."""
    return {(name.replace(STEM + '.', STEM + '.base.', 1) if name.startswith(STEM + '.') else name): value
            for name, value in state.items()}
