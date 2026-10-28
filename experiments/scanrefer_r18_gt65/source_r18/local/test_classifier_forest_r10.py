import unittest
import numpy as np
from classifier_forest_r10 import mapped_probability

class MappingTests(unittest.TestCase):
 def test_unobserved_training_classes_do_not_shift_vocabulary_indices(self):
  class Forest:
   classes_=np.array([0,2,5])
   def predict_proba(self,x): return np.array([[.1,.2,.7],[.8,.2,0.]])
  actual=mapped_probability(Forest(),np.zeros((2,3)),7)
  np.testing.assert_array_equal(actual,[[.1,0,.2,0,0,.7,0],[.8,0,.2,0,0,0,0]])
  self.assertEqual(actual[0].argmax(),5)

if __name__=='__main__': unittest.main()
