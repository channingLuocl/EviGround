import unittest

import torch
from torch import nn
from torch.nn import functional as F

from seeground.classifiers.classifier_backbone_r5 import augment, configure, head_for, selector


class BackboneTests(unittest.TestCase):
    def test_augmentation_preserves_geometry_contract_and_input(self):
        points = torch.randn(5,64,4)
        original = points.clone()
        a = augment(points,torch.Generator().manual_seed(17))
        b = augment(points,torch.Generator().manual_seed(17))
        torch.testing.assert_close(points,original,atol=0,rtol=0)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        torch.testing.assert_close(a[...,:3].mean(1),torch.zeros(5,3),atol=1e-7,rtol=0)
        torch.testing.assert_close(a[...,:3].norm(dim=-1).amax(1),torch.ones(5))
        torch.testing.assert_close(a[...,3],a[...,2]-a[...,2].amin(1,keepdim=True))
        self.assertFalse(torch.equal(a,points))

    def test_semantic_head_initially_matches_matching_text(self):
        text = F.normalize(torch.randn(7,32),dim=-1)
        x = F.normalize(torch.randn(9,32),dim=-1)
        torch.testing.assert_close(head_for(text)(x),10*x@text.T)

    def test_only_last_stages_projection_train_and_bn_stays_eval(self):
        model = nn.Module()
        model.obj_encoder = nn.Module()
        model.obj_encoder.encoder = nn.Module()
        model.obj_encoder.encoder.encoder = nn.Sequential(*[nn.Linear(3,3) for _ in range(6)])
        model.obj_encoder.prediction = nn.Sequential(nn.BatchNorm1d(3),nn.Dropout(.5))
        model.pc_projection = nn.Parameter(torch.randn(3,4))
        configure(model,'last2')
        for name,p in model.named_parameters():
            expected = name.startswith(('obj_encoder.encoder.encoder.4.','obj_encoder.encoder.encoder.5.',
                                        'obj_encoder.prediction.','pc_projection'))
            self.assertEqual(p.requires_grad,expected,name)
        self.assertTrue(all(not m.training for m in model.modules()))
        configure(model,'frozen')
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_selection_enforces_validation_constraints_and_keeps_reference(self):
        reference = dict(name='r4_M',epoch=0,base='r4_M',weight=0.,parameters=0,mean_correct=314.,mean_macro=.62)
        more_true = {**reference,'name':'last2','mean_correct':320.,'mean_macro':.60,'parameters':100}
        more_macro = {**reference,'name':'last2_aug','mean_correct':310.,'mean_macro':.65,'parameters':100}
        self.assertEqual(selector([reference,more_true,more_macro],reference,'P'),reference)
        self.assertEqual(selector([reference,more_true,more_macro],reference,'M'),reference)
        both = {**more_true,'mean_macro':.64}
        self.assertEqual(selector([reference,both],reference,'P'),both)
        self.assertEqual(selector([reference,both],reference,'M'),both)


if __name__=='__main__':
    unittest.main()
