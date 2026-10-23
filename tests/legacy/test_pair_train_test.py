import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from eviground.experiments.audit_pair_train_test import gate, recount
from eviground.experiments.run_pair_train_test import choose_validation
from eviground.geometry.shape_scale import score
from eviground.experiments.train_pair_views import Gate


class ProtocolTests(unittest.TestCase):
    def test_validation_selection_uses_true_aliases_and_retains_oov(self):
        manifest={'vocabulary':['couch','chair'], 'samples':[
            {'label':-1,'category':'sofa'}, {'label':1,'category':'chair'},
            {'label':-1,'category':'unknown'}]}
        base=torch.tensor([[0.,2.],[2.,0.],[0.,2.]])
        rgb=torch.tensor([[2.,0.],[0.,2.],[0.,2.]])
        selected,grid=choose_validation(base,{'ce':[rgb]*3},manifest,
                                        torch.tensor([.5,.5]),[('ce',0.)])
        self.assertEqual(selected['total_correct'],6)
        self.assertAlmostEqual(selected['mean_macro'],2/3)
        self.assertEqual(len(grid),11)

    def test_gate_checkpoint_reproduces_predictions(self):
        model=Gate(torch.zeros(16),torch.ones(16),.4)
        with torch.no_grad():
            model.network[-1].bias.fill_(.2)
        other=Gate(model.mean,model.std,.4)
        other.load_state_dict(model.state_dict(),strict=True)
        x=torch.randn(5,16)
        different=torch.tensor([True,False,True,True,False])
        np.testing.assert_array_equal(model(x,different).detach(),other(x,different).detach())
        self.assertEqual(float(other(x,different)[1].detach()),float(torch.tensor(.4)))

    def test_independent_audit_forward_matches_trained_gate(self):
        torch.manual_seed(17)
        model=Gate(torch.randn(16),torch.rand(16)+.1,.4)
        with torch.no_grad():
            model.network[-1].weight.normal_()
            model.network[-1].bias.fill_(.1)
        x=torch.randn(15,16)
        different=torch.arange(15)%2==0
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'gate.pt'
            torch.save({'state_dict':model.state_dict(),'weight':.4},path)
            expected=gate(path,x.numpy().astype(float),different.numpy())
        np.testing.assert_allclose(expected,model(x,different).detach(),atol=1e-6)

    def test_independent_recount_matches_alias_and_oov_metrics(self):
        manifest={'vocabulary':['couch','chair','table'], 'samples':[
            {'category':'sofa','label':-1},{'category':'chair','label':1},
            {'category':'unknown','label':-1},{'category':'table','label':2}]}
        values=np.array([[3.,1.,0.],[1.,2.,0.],[2.,0.,1.],[0.,2.,1.]])
        actual=recount(values,manifest)[0]
        expected=score(values,manifest)
        for key,value in actual.items():
            self.assertAlmostEqual(value,expected[key])


if __name__=='__main__':
    unittest.main()
