import numpy as np

from scipy.linalg import expm
from scipy import integrate

import mathrobo as mr

def test_cmtm_so3():
  v = np.random.rand(3) 
  r = mr.SO3.exp(v)

  so3 = mr.SO3(r)
  res = mr.CMTM[mr.SO3](so3)

  np.testing.assert_array_equal(res.mat(), so3.mat())