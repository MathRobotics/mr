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
  
def test_cmtm_so3_vel():
  v = np.random.rand(3) 
  r = mr.SO3.exp(v)

  so3 = mr.SO3(r)  
  vel = np.random.rand(1,3) 

  res = mr.CMTM[mr.SO3](so3, vel)
  
  mat = np.zeros((6,6))
  mat[0:3,0:3] = mat[3:6,3:6] = so3.mat()
  mat[3:6,0:3] = so3.mat() @ so3.hat(vel[0])

  np.testing.assert_array_equal(res.mat(), mat)
  
def test_cmtm_so3_vel_acc():
  v = np.zeros(3)
  r = mr.SO3.exp(v)

  so3 = mr.SO3(r)  
  vel = np.random.rand(2,3)

  res = mr.CMTM[mr.SO3](so3, vel)
  
  mat = np.zeros((9,9))
  mat[0:3,0:3] = mat[3:6,3:6] = mat[6:9,6:9] = so3.mat()
  mat[3:6,0:3] = mat[6:9,3:6] = so3.mat() @ so3.hat(vel[0])
  mat[6:9,0:3] = so3.mat() @ (so3.hat(vel[1]) + so3.hat(vel[0]) @ so3.hat(vel[0])) * 0.5

  np.testing.assert_array_equal(res.mat(), mat)