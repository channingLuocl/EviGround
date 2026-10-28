import unittest

import numpy as np

from classifier_sampling_r8 import choose, rgb_branch, vote, yaw
from audit_classifier_sampling_r8 import pick, scores


class SamplingTests(unittest.TestCase):
    def test_yaw_preserves_height_z_radius_and_input(self):
        points=np.random.default_rng(5).normal(size=(39,4)).astype(np.float32)
        original=points.copy()
        actual=yaw(points,10)
        np.testing.assert_array_equal(points,original)
        np.testing.assert_array_equal(actual[:,2:],points[:,2:])
        np.testing.assert_allclose(np.linalg.norm(actual[:,:3],axis=-1),np.linalg.norm(points[:,:3],axis=-1),atol=3e-7,rtol=1e-6)
        np.testing.assert_allclose(yaw(actual,-10),points,atol=3e-7,rtol=1e-6)

    def test_rgb_recovery_reconstructs_frozen_fusion(self):
        rng=np.random.default_rng(7)
        base,rgb=rng.normal(size=(2,5,9))
        reference=.4*base+.6*rgb
        recovered=rgb_branch(reference,base)
        np.testing.assert_allclose(recovered,rgb,atol=1e-15,rtol=1e-14)
        np.testing.assert_allclose(.4*base+.6*recovered,reference,atol=1e-15,rtol=1e-14)

    def test_voting_is_mean_of_cosines(self):
        rng=np.random.default_rng(17)
        features=rng.normal(size=(8,9,12))
        text=rng.normal(size=(7,12))
        indices=[0,3,5]
        expected=np.mean([10*features[:,i]@text.T for i in indices],axis=0)
        np.testing.assert_allclose(vote(features,text,indices),expected)
        np.testing.assert_allclose(vote(features,text,indices),scores(features,text,indices))

    def test_overall_and_balanced_objectives_are_distinct(self):
        ref={'name':'r4_M','passes':1,'rgb_weight':.6,'correct':314,'macro':.62}
        accuracy={**ref,'name':'sample3','passes':3,'correct':320,'macro':.60}
        self.assertEqual(choose([ref,accuracy],ref,'P'),accuracy)
        self.assertEqual(choose([ref,accuracy],ref,'B'),ref)
        self.assertEqual(choose([ref,accuracy],ref,'M'),ref)
        for mode in ('P','B','M'):
            self.assertEqual(choose([ref,accuracy],ref,mode),pick([ref,accuracy],ref,mode))


if __name__=='__main__':
    unittest.main()
