import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from seeground.experiments.audit_classifier_sweep_r3 import independent_network
from seeground.experiments.audit_pair_views import z
from seeground.experiments.audit_classifier_balance_r4 import numpy_candidates,paired_bootstrap
from seeground.classifiers.classifier_balance_r4 import (build_candidates,choose,class_weights,fit,make_model,
    sample_indices,training_loss)


class BalanceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(27)

    def test_zero_power_weights_reproduce_plain_ce(self):
        logits=torch.randn(9,5)
        labels=torch.arange(9)%5
        counts=torch.tensor([1.,2.,3.,10.,30.])
        torch.testing.assert_close(training_loss(logits,labels,counts,{},0),F.cross_entropy(logits,labels))
        torch.testing.assert_close(class_weights(counts,0),torch.ones(5))

    def test_focal_gamma_zero_equals_weighted_ce(self):
        logits=torch.randn(10,5,requires_grad=True)
        labels=torch.arange(10)%5
        counts=torch.tensor([1.,2.,3.,10.,30.])
        actual=training_loss(logits,labels,counts,{'power':.5,'focal':0.},0)
        expected=F.cross_entropy(logits,labels,weight=class_weights(counts,.5))
        torch.testing.assert_close(actual,expected)
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_deferred_weights_switch_at_epoch_twenty(self):
        logits=torch.randn(9,5)
        labels=torch.arange(9)%5
        counts=torch.tensor([1.,2.,3.,10.,30.])
        spec={'power':1.,'deferred':True}
        torch.testing.assert_close(training_loss(logits,labels,counts,spec,19),F.cross_entropy(logits,labels))
        torch.testing.assert_close(training_loss(logits,labels,counts,spec,20),F.cross_entropy(logits,labels,weight=class_weights(counts,1.)))

    def test_sampler_increases_rare_class_exposure_with_fixed_draw_budget(self):
        labels=torch.tensor([0]*1000+[1]*20+[2]*2)
        counts=torch.bincount(labels).float()
        generator=torch.Generator().manual_seed(42)
        indices=sample_indices(labels,counts,{'sampler':1.},generator)
        observed=torch.bincount(labels[indices],minlength=3)
        self.assertEqual(len(indices),len(labels))
        self.assertGreater(int(observed[2]),100)
        self.assertLess(int(observed[0]),600)
        self.assertTrue(bool((indices>=0).all() and (indices<len(labels)).all()))

    def test_weighted_smoothing_matches_torch_definition(self):
        logits=torch.randn(10,5)
        labels=torch.arange(10)%5
        counts=torch.arange(1,6).float()
        expected=F.cross_entropy(logits,labels,weight=class_weights(counts,.5),label_smoothing=.05)
        torch.testing.assert_close(training_loss(logits,labels,counts,{'power':.5,'smooth':.05},0),expected)

    def test_width_variants_match_independent_numpy_forward(self):
        text=F.normalize(torch.randn(5,512),dim=-1)
        data=(torch.randn(3,512),torch.randn(3,512),torch.randn(3,4,512))
        for width in (128,512):
            model=make_model(text,width=width).eval()
            expected=independent_network({'architecture':'rgb_mlp','state_dict':model.state_dict()},tuple(x.numpy().astype(float) for x in data))
            np.testing.assert_allclose(model(*data).detach(),expected,atol=1e-5,rtol=1e-5)

    def test_ensemble_formula_normalizes_each_seed_before_averaging(self):
        families={n:10 for n in ('old_rgb_mlp_ce','old_rgb_mlp_weighted','old_view_attention_ce')}
        bank={f'{name}_seed{seed}':torch.randn(7,5)*(seed-40) for name in families for seed in (42,43,44)}
        candidates,metadata=build_candidates(bank,families)
        for name in families:
            expected=np.mean([z(bank[f'{name}_seed{s}'].numpy()) for s in (42,43,44)],axis=0)
            np.testing.assert_allclose(candidates['ensemble_'+name][0],expected,atol=1e-6)
            self.assertEqual(metadata['ensemble_'+name]['variants'],1)
            self.assertEqual(metadata['single_'+name]['variants'],3)
        np.testing.assert_allclose(candidates['pair_old_rgb_mlp_ce_0.25'][0],
            .25*candidates['ensemble_old_rgb_mlp_ce'][0]+.75*candidates['ensemble_old_rgb_mlp_weighted'][0],atol=1e-6)
        rebuilt=numpy_candidates({k:v.numpy().astype(float) for k,v in bank.items()},metadata)
        for name,values in candidates.items():
            np.testing.assert_allclose(values,rebuilt[name],atol=1e-6)

    def test_room_bootstrap_accepts_one_ensemble_and_three_references(self):
        manifest={'samples':[{'scene':f'scene{i//2:04d}_00'} for i in range(8)]}
        left=np.ones((1,8),dtype=bool)
        right=np.zeros((3,8),dtype=bool)
        result=paired_bootstrap(left,right,manifest)
        self.assertEqual(result['delta_pp'],100.)
        self.assertEqual(result['room_95CI_pp'],[100.,100.])
        self.assertEqual(result['corrected_pairs'],[[8,8,8]])
        self.assertEqual(result['candidate_variants'],1)

    def test_selection_compares_average_not_number_of_trials(self):
        common={'parameters':1,'tau':0.,'weight':.4,'mean_macro':.7}
        rows=[{**common,'name':'single','mean_correct':3.,'correct_by_variant':[3,3,3]},
              {**common,'name':'ensemble','mean_correct':4.,'correct_by_variant':[4]}]
        reference={'mean_correct':2.,'mean_macro':.5}
        self.assertEqual(choose(rows,reference,'P')['name'],'ensemble')
        self.assertEqual(choose(rows,reference,'M')['name'],'ensemble')

    def test_training_records_actual_sampling_and_parameter_changes(self):
        text=F.normalize(torch.randn(5,16),dim=-1)
        x=F.normalize(torch.randn(32,16),dim=-1)
        data=(x,x,torch.randn(32,4,16))
        labels=torch.arange(32)%5
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            (out/'config.json').write_text('{}')
            args=SimpleNamespace(output=out,device='cpu')
            model,_=fit(args,{'name':'sampler','sampler':1.},42,data,labels,text,torch.bincount(labels,minlength=5).float())
            root=out/'models/sampler/seed42'
            record=torch.load(root/'model.pt',weights_only=True)
            sampling=__import__('json').loads((root/'sampling.json').read_text())
            self.assertEqual(record['draws'],1280)
            self.assertGreater(record['parameter_change_l2'],0)
            self.assertTrue(all(sum(row)==32 for row in sampling))
            self.assertEqual(len(sampling),40)
            self.assertTrue(torch.isfinite(model(*data)).all())


if __name__=='__main__':
    unittest.main()
