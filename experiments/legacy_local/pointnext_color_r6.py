"""Add point RGB to the pretrained XYZ+height stem without changing its initial function."""
import numpy as np
import torch
from torch import nn

from pointcloud_classifier_v82 import prepare_points

STEM = 'obj_encoder.encoder.encoder.0.0.convs.0.0'
TRAINABLE = (STEM+'.', 'obj_encoder.encoder.encoder.4.', 'obj_encoder.encoder.encoder.5.',
             'obj_encoder.prediction.', 'pc_projection')


class ColorPointNext(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.obj_encoder = original.obj_encoder
        self.pc_projection = original.pc_projection
        # Released stage zero is a single Conv1d(4,32,1), without BN or activation.
        parent = self.obj_encoder.encoder.encoder[0][0].convs[0]
        old = parent[0]
        assert isinstance(old,nn.Conv1d) and old.in_channels==4 and old.groups==1 and old.kernel_size==(1,)
        new = nn.Conv1d(7,old.out_channels,1,bias=old.bias is not None,
                        device=old.weight.device,dtype=old.weight.dtype)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:,:4].copy_(old.weight)
            if old.bias is not None:
                new.bias.copy_(old.bias)
        parent[0] = new

    def forward(self, points):
        assert points.shape[-1]==7
        # The original PcdClassifier.forward slices [:4]; bypass that slice here.
        return self.obj_encoder(points) @ self.pc_projection


def configure(model):
    for name,p in model.named_parameters():
        p.requires_grad_(name.startswith(TRAINABLE))
    model.eval()


def pack_points(xyz, rgb, seed):
    xyz,rgb = np.asarray(xyz),np.asarray(rgb)
    if rgb.shape!=xyz.shape or not np.isfinite(rgb).all() or (rgb<0).any() or (rgb>255).any():
        raise ValueError('Expected matching finite RGB in 0..255')
    indices = np.random.RandomState(seed).choice(len(xyz),2048,replace=len(xyz)<2048)
    shape = prepare_points(xyz,None,'pointnext',seed)
    color = np.asarray(rgb[indices],dtype=np.float32)/255.-.5
    return np.concatenate((shape,color),axis=-1)


def color_input(points, spec, generator=None):
    result = points.clone()
    if spec=='geometry':
        result[...,4:] = 0
    elif spec=='color_regularized':
        assert generator is not None
        size = (len(points),1,1)
        rand = lambda shape: torch.rand(shape,device=points.device,generator=generator)
        gain = .8+.4*rand(size)
        shift = -.1+.2*rand((len(points),1,3))
        color = ((result[...,4:]+.5)*gain+shift).clamp(0,1)-.5
        keep = rand(size)>=.2
        result[...,4:] = color*keep
    elif spec!='color':
        raise ValueError(spec)
    return result
