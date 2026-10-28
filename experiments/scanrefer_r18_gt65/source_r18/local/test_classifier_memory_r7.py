import unittest

import numpy as np

from classifier_memory_r7 import choose, embedding, kernel_scores, unit
from audit_classifier_memory_r7 import independent_raw
import torch


class MemoryTests(unittest.TestCase):
    def test_nearest_ties_use_training_index(self):
        memory=np.array([[1.,0.],[1.,0.],[0.,1.]])
        labels=np.array([1,0,2])
        value=kernel_scores(memory,labels,np.array([[1.,0.]]),3,1,20.)
        np.testing.assert_array_equal(value,np.array([[0.,1.,0.]]))

    def test_kernel_votes_match_direct_nested_loop(self):
        memory=unit(np.array([[1.,2.],[2.,1.],[-1.,3.],[4.,-1.]]))
        labels=np.array([0,1,0,2])
        queries=unit(np.array([[1.,1.],[-2.,1.]]))
        for k in (0,2,100):
            expected=np.zeros((2,3))
            for i,q in enumerate(queries):
                pairs=sorted(enumerate(q@memory.T),key=lambda p:(-p[1],p[0]))
                for j,s in pairs[:k or len(memory)]:
                    expected[i,labels[j]]+=np.exp(10*(np.clip(s,-1,1)-1))
            np.testing.assert_allclose(kernel_scores(memory,labels,queries,3,k,10),expected,atol=1e-14,rtol=1e-12)

    def test_joint_space_is_equal_modality_cosine(self):
        shape=np.array([[2.,1.],[1.,-2.]])
        rgb=np.array([[5.,-1.],[1.,3.]])
        x=embedding(shape,rgb,'joint')
        expected=.5*(unit(shape)@unit(shape).T+unit(rgb)@unit(rgb).T)
        np.testing.assert_allclose(x@x.T,expected)

    def test_independent_torch_retrieval_matches_numpy(self):
        rng=np.random.default_rng(17)
        memory=unit(rng.normal(size=(53,16)))
        query=unit(rng.normal(size=(11,16)))
        labels=rng.integers(0,7,53)
        for k,beta in ((0,10.),(0,20.),(20,20.),(53,20.)):
            actual=kernel_scores(memory,labels,query,7,k,beta)
            expected=independent_raw(torch.from_numpy(memory),torch.from_numpy(query),torch.from_numpy(labels),7,k,beta)
            np.testing.assert_allclose(actual,expected,atol=1e-14,rtol=1e-12)

    def test_validation_constraints_keep_existing_reference(self):
        ref={'name':'r4_M','base':'r4_M','weight':0.,'memory_bytes':0,'correct':314,'macro':.62}
        overall={**ref,'name':'new','correct':320,'macro':.60,'memory_bytes':1024}
        macro={**ref,'name':'other','correct':310,'macro':.65,'memory_bytes':1024}
        self.assertEqual(choose([ref,overall,macro],ref,'P'),ref)
        self.assertEqual(choose([ref,overall,macro],ref,'M'),ref)
        both={**overall,'macro':.63}
        self.assertEqual(choose([ref,both],ref,'P'),both)


if __name__=='__main__':
    unittest.main()
