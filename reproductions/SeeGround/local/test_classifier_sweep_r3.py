import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from classifier_sweep_r3 import (Classifier, analytic_predict, choose, fast_metric,
    fit_analytic, fit_network, loss_function, mix_features)
from audit_pair_train_test import recount
from audit_classifier_sweep_r3 import independent_network


class SweepTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        self.text=F.normalize(torch.randn(6,32),dim=-1)

    def test_residual_adapter_initially_matches_frozen_text(self):
        shape,rgb=F.normalize(torch.randn(9,32),dim=-1),F.normalize(torch.randn(9,32),dim=-1)
        views=torch.randn(9,4,32)
        for name,x in (('rgb_adapter',rgb),('shape_adapter',shape)):
            model=Classifier(name,self.text).eval()
            torch.testing.assert_close(model(shape,rgb,views),10*x@self.text.T)

    def test_attention_is_view_permutation_invariant(self):
        model=Classifier('view_attention',self.text).eval()
        shape,rgb,views=torch.randn(5,32),torch.randn(5,32),torch.randn(5,4,32)
        torch.testing.assert_close(model(shape,rgb,views),model(shape,rgb,views[:,[2,0,3,1]]),atol=1e-6,rtol=1e-5)

    def test_independent_numpy_all_architecture_forwards(self):
        # The attention auditor expands four 32-dimensional heads explicitly.
        text=F.normalize(torch.randn(6,512),dim=-1)
        shape,rgb=F.normalize(torch.randn(5,512),dim=-1),F.normalize(torch.randn(5,512),dim=-1)
        views=torch.randn(5,4,512)
        for name in ('rgb_mlp','joint_mlp','rgb_adapter','shape_adapter','view_attention'):
            model=Classifier(name,text).eval()
            if name.endswith('adapter'):
                with torch.no_grad():
                    model.adapter[-1].weight.normal_(std=.01)
            record={'architecture':name,'state_dict':model.state_dict()}
            actual=model(shape,rgb,views).detach().numpy()
            expected=independent_network(record,tuple(x.numpy().astype(float) for x in (shape,rgb,views)))
            np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-5)

    def test_mixup_uses_same_object_pair_in_both_modalities(self):
        x=torch.arange(12).reshape(4,3).float()
        p=torch.tensor([2,0,3,1])
        a,b,c=mix_features((x,x+100,x[:,None,:].repeat(1,4,1)),p,.3)
        torch.testing.assert_close(b,a+100)
        torch.testing.assert_close(c[:,0],a)

    def test_balanced_loss_uses_train_prior_and_weighted_loss_is_finite(self):
        scores=torch.randn(9,6,requires_grad=True)
        y=torch.arange(9)%6
        counts=torch.tensor([1.,2.,3.,4.,10.,20.])
        torch.testing.assert_close(loss_function(scores,y,counts,'balanced'),F.cross_entropy(scores+(counts+1).log(),y))
        value=loss_function(scores,y,counts,'weighted')
        value.backward()
        self.assertTrue(torch.isfinite(value) and torch.isfinite(scores.grad).all())

    def test_analytic_models_learn_train_only_separable_features(self):
        x=torch.eye(4).repeat(6,1)
        y=torch.arange(4).repeat(6)
        for kind,strength in (('prototype',0),('ridge',.1)):
            state=fit_analytic(x,y,4,kind,strength)
            output=analytic_predict(state,x)
            torch.testing.assert_close(output.argmax(-1),y)
            self.assertTrue(torch.isfinite(output).all())

    def test_fast_metric_matches_independent_recount_with_oov(self):
        manifest={'vocabulary':['couch','chair','table'],'samples':[
            {'category':'sofa','label':-1},{'category':'chair','label':1},
            {'category':'missing','label':-1},{'category':'table','label':2}]}
        scores=np.array([[3.,1.,0.],[1.,2.,0.],[2.,0.,1.],[0.,2.,1.]])
        actual=fast_metric(torch.from_numpy(scores),manifest)
        expected=recount(scores,manifest)[0]
        self.assertEqual(actual[0],expected['correct'])
        self.assertEqual(actual[1],expected['class_mean_accuracy'])

    def test_two_selection_objectives_enforce_distinct_constraints(self):
        rows=[{'name':str(i),'parameters':1,'tau':0.,'weight':.4,'total_correct':n,'mean_macro':m}
              for i,n,m in ((0,24,.5),(1,27,.6),(2,25,.7),(3,23,.9))]
        self.assertEqual(choose(rows,8,.5,'P')['name'],'1')
        self.assertEqual(choose(rows,8,.5,'M')['name'],'2')

    def test_training_updates_parameters_and_records_all_epochs(self):
        x=F.normalize(torch.randn(32,32),dim=-1)
        y=torch.arange(32)%6
        data=(x,x,torch.randn(32,4,32))
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            (out/'config.json').write_text('{}')
            args=SimpleNamespace(output=out,device='cpu')
            model,_=fit_network(args,'joint_mlp','mixup',42,data,y,self.text,torch.bincount(y,minlength=6).float())
            saved=torch.load(out/'models/joint_mlp_mixup/seed42/model.pt',weights_only=True)
            self.assertGreater(saved['parameter_change_l2'],0)
            self.assertEqual(saved['epochs'],40)
            self.assertEqual(saved['train_objects'],32)
            restored=Classifier('joint_mlp',self.text).eval()
            restored.load_state_dict(saved['state_dict'])
            torch.testing.assert_close(restored(*data),model(*data))


if __name__=='__main__':
    unittest.main()
