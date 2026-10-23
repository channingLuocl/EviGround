import copy
import unittest

import numpy as np
import torch
from torch import nn

from eviground.classifiers.pointcloud_classifier_v82 import prepare_points
from eviground.classifiers.pointnext_color_r6 import ColorPointNext, color_input, configure, pack_points


class Stage(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs=nn.Sequential(nn.Sequential(nn.Conv1d(4,8,1)))

    def forward(self,x):
        return self.convs(x)


class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Module()
        self.encoder.encoder=nn.Sequential(nn.Sequential(Stage()))

    def forward(self,x):
        return self.encoder.encoder(x.transpose(1,2)).mean(-1)


def toy():
    model=nn.Module()
    model.obj_encoder=ToyEncoder()
    model.pc_projection=nn.Parameter(torch.randn(8,12))
    return model


class ColorTests(unittest.TestCase):
    def test_widening_preserves_original_function_for_any_rgb(self):
        original=toy()
        before=copy.deepcopy(original)
        model=ColorPointNext(original)
        points=torch.randn(4,27,7)
        expected=before.obj_encoder(points[...,:4])@before.pc_projection
        torch.testing.assert_close(model(points),expected)
        points[...,4:]=100*torch.randn_like(points[...,4:])
        torch.testing.assert_close(model(points),expected)
        torch.testing.assert_close(model.state_dict()['obj_encoder.encoder.encoder.0.0.convs.0.0.weight'][:,:4],
                                   before.state_dict()['obj_encoder.encoder.encoder.0.0.convs.0.0.weight'])

    def test_rgb_receives_gradient_and_changes_output(self):
        model=ColorPointNext(toy())
        configure(model)
        x=torch.randn(4,37,7)
        optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.01)
        first=model(x).detach()
        model(x).square().mean().backward()
        stem=model.obj_encoder.encoder.encoder[0][0].convs[0][0]
        self.assertGreater(float(stem.weight.grad[:,4:].norm()),0)
        optimizer.step()
        self.assertFalse(torch.equal(model(x),first))
        self.assertTrue(all(not m.training for m in model.modules()))

    def test_geometry_and_color_sample_same_point_indices(self):
        rng=np.random.default_rng(51)
        xyz=rng.normal(size=(19,3))
        rgb=np.arange(57).reshape(19,3)
        actual=pack_points(xyz,rgb,17)
        np.testing.assert_array_equal(actual[:,:4],prepare_points(xyz,None,'pointnext',17))
        indices=np.random.RandomState(17).choice(19,2048,replace=True)
        np.testing.assert_allclose(actual[:,4:],rgb[indices].astype(np.float32)/255.-.5,atol=0,rtol=0)

    def test_color_regularization_preserves_geometry_and_is_repeatable(self):
        x=torch.randn(10,24,7)
        x[...,4:]=torch.rand(10,24,3)-.5
        before=x.clone()
        a=color_input(x,'color_regularized',torch.Generator().manual_seed(8))
        b=color_input(x,'color_regularized',torch.Generator().manual_seed(8))
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        torch.testing.assert_close(a[...,:4],x[...,:4],atol=0,rtol=0)
        torch.testing.assert_close(x,before,atol=0,rtol=0)
        self.assertLessEqual(float(a[...,4:].max()),.5)
        self.assertGreaterEqual(float(a[...,4:].min()),-.5)
        self.assertEqual(float(color_input(x,'geometry')[...,4:].abs().sum()),0.)


if __name__=='__main__':
    unittest.main()
