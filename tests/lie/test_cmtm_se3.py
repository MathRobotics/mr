import numpy as np

from scipy.linalg import expm
from scipy import integrate

import mathrobo as mr

def test_cmtm_se3():
  v = np.random.rand(6) 
  r = mr.SE3.exp(v)

  se3 = mr.SE3.set_mat(r)
  res = mr.CMTM[mr.SE3](se3)

  np.testing.assert_array_equal(res.mat(), se3.mat())
  
def test_cmtm_se3_vec1d():
  v = np.random.rand(6) 
  r = mr.SE3.exp(v)

  se3 = mr.SE3.set_mat(r)  
  vel = np.random.rand(1,6) 

  res = mr.CMTM[mr.SE3](se3, vel)
  
  mat = np.zeros((8,8))
  mat[0:4,0:4] = mat[4:8,4:8] = se3.mat()
  mat[4:8,0:4] = se3.mat() @ se3.hat(vel[0])

  np.testing.assert_array_equal(res.mat(), mat)
  
def test_cmtm_se3_vec2d():
  v = np.zeros(6)
  r = mr.SE3.exp(v)
  se3 = mr.SE3.set_mat(r)  
  vec = np.random.rand(2,6)

  res = mr.CMTM[mr.SE3](se3, vec)
  
  mat = np.zeros((12,12))
  mat[0:4,0:4] = mat[4:8,4:8] = mat[8:12,8:12] = se3.mat()
  mat[4:8,0:4] = mat[8:12,4:8] = se3.mat() @ se3.hat(vec[0])
  mat[8:12,0:4] = se3.mat() @ (se3.hat(vec[1]) + se3.hat(vec[0]) @ se3.hat(vec[0])) * 0.5

  np.testing.assert_array_equal(res.mat(), mat)
  
def test_cmtm_se3_adj():
  v = np.random.rand(6) 
  r = mr.SE3.exp(v)

  se3 = mr.SE3.set_mat(r)
  res = mr.CMTM[mr.SE3](se3)

  np.testing.assert_array_equal(res.adj_mat(), se3.adj_mat())
  
def test_cmtm_se3_vec1d():
  v = np.random.rand(6) 
  r = mr.SE3.exp(v)

  se3 = mr.SE3.set_mat(r)  
  vel = np.random.rand(1,6) 

  res = mr.CMTM[mr.SE3](se3, vel)
  
  mat = np.zeros((12,12))
  mat[0:6,0:6] = mat[6:12,6:12] = se3.adj_mat()
  mat[6:12,0:6] = se3.adj_mat() @ se3.hat_adj(vel[0])

  np.testing.assert_array_equal(res.adj_mat(), mat)
  
def test_cmtm_se3_adj_vec2d():
  v = np.zeros(6)
  r = mr.SE3.exp(v)
  se3 = mr.SE3.set_mat(r)  
  vec = np.random.rand(2,6)

  res = mr.CMTM[mr.SE3](se3, vec)
  
  mat = np.zeros((18,18))
  mat[0:6,0:6] = mat[6:12,6:12] = mat[12:18,12:18] = se3.adj_mat()
  mat[6:12,0:6] = mat[12:18,6:12] = se3.adj_mat() @ se3.hat_adj(vec[0])
  mat[12:18,0:6] = se3.adj_mat() @ (se3.hat_adj(vec[1]) + se3.hat_adj(vec[0]) @ se3.hat_adj(vec[0])) * 0.5

  np.testing.assert_array_equal(res.adj_mat(), mat)
  
def test_cmtm_se3_getter():
  v = np.zeros(6)
  r = mr.SE3.exp(v)
  se3 = mr.SE3.set_mat(r)  
  vec = np.random.rand(2,6)

  res = mr.CMTM[mr.SE3](se3,vec)
  
  np.testing.assert_array_equal(res.elem_mat(), se3.mat())
  np.testing.assert_array_equal(res.elem_vecs(0), vec[0])
  np.testing.assert_array_equal(res.elem_vecs(1), vec[1])
  
def test_cmtm_se3_inv():
  v = np.zeros(6)
  r = mr.SE3.exp(v)
  se3 = mr.SE3.set_mat(r)  
  
  for i in range(3):
    vel = np.random.rand(i,6)

    res = mr.CMTM[mr.SE3](se3, vel)
    
    mat = np.eye(4*(i+1))

    np.testing.assert_allclose(res.mat() @ res.inverse(), mat, rtol=1e-15, atol=1e-15)
    
def test_cmtm_se3_adj_inv():
  v = np.zeros(6)
  r = mr.SE3.exp(v)
  se3 = mr.SE3.set_mat(r)  
  
  for i in range(3):
    vel = np.random.rand(i,6)

    res = mr.CMTM[mr.SE3](se3, vel)
    
    mat = np.eye(6*(i+1))

    np.testing.assert_allclose(res.adj_mat() @ res.inverse_adj(), mat, rtol=1e-15, atol=1e-15)
    
def test_cmtm_se3_tangent_mat():
  v = np.zeros(6)
  r = mr.SE3.exp(v)
  se3 = mr.SE3.set_mat(r)  
  
  res = mr.CMTM[mr.SE3](se3)
  
  mat = np.eye(4)
  
  np.testing.assert_array_equal(res.tangent_mat(), mat)
  
def test_cmtm_se3_vec1d_tangent_mat():
  v = np.zeros(6)
  r = mr.SE3.exp(v)
  se3 = mr.SE3.set_mat(r)  
  vel = np.random.rand(1,6)
  
  res = mr.CMTM[mr.SE3](se3, vel)
  
  mat = np.eye(8)
  mat[4:8, 0:4] = - mr.SE3.hat(vel[0])
  
  np.testing.assert_array_equal(res.tangent_mat(), mat)
  
def test_cmtm_se3_vec2d_tangent_mat():
  v = np.zeros(6)
  r = mr.SE3.exp(v)
  se3 = mr.SE3.set_mat(r)  
  vel = np.random.rand(2,6)
  
  res = mr.CMTM[mr.SE3](se3, vel)
  
  mat = np.eye(12)
  mat[4:8, 0:4] = mat[8:12, 4:8] = - mr.SE3.hat(vel[0])
  mat[8:12, 0:4] = - (mr.SE3.hat(vel[1]) - mr.SE3.hat(vel[0]) @ mr.SE3.hat(vel[0])) 

  np.testing.assert_array_equal(res.tangent_mat(), mat)
  
def test_cmtm_se3_tangent_adj_mat():
  v = np.zeros(6)
  r = mr.SE3.exp(v)
  se3 = mr.SE3.set_mat(r)  
  
  res = mr.CMTM[mr.SE3](se3)
  
  mat = np.eye(6)
  
  np.testing.assert_array_equal(res.tangent_adj_mat(), mat)
  
def test_cmtm_se3_vec1d_tangent_adj_mat():
  v = np.zeros(6)
  r = mr.SE3.exp(v)
  se3 = mr.SE3.set_mat(r)  
  vel = np.random.rand(1,6)
  
  res = mr.CMTM[mr.SE3](se3, vel)
  
  mat = np.eye(12)
  mat[6:12, 0:6] = - mr.SE3.hat_adj(vel[0])
  
  np.testing.assert_array_equal(res.tangent_adj_mat(), mat)
  
def test_cmtm_se3_vec2d_tangent_adj_mat():
  v = np.zeros(6)
  r = mr.SE3.exp(v)
  se3 = mr.SE3.set_mat(r)  
  vel = np.random.rand(2,6)
  
  res = mr.CMTM[mr.SE3](se3, vel)
  
  mat = np.eye(18)
  mat[6:12, 0:6] = mat[12:18, 6:12] = - mr.SE3.hat_adj(vel[0])
  mat[12:18, 0:6] = - (mr.SE3.hat_adj(vel[1]) - mr.SE3.hat_adj(vel[0]) @ mr.SE3.hat_adj(vel[0])) 

  np.testing.assert_array_equal(res.tangent_adj_mat(), mat)
  
def test_cmtm_se3_tangent_inv():
  v = np.zeros(6)
  r = mr.SE3.exp(v)
  se3 = mr.SE3.set_mat(r)  
  
  for i in range(3):
    vel = np.random.rand(i,6)

    res = mr.CMTM[mr.SE3](se3, vel)
    
    mat = np.eye(4*(i+1))

    np.testing.assert_allclose(res.tangent_mat() @ res.tangent_mat_inv(), mat, rtol=1e-15, atol=1e-15)
    
def test_cmtm_se3_tangent_adj_inv():
  v = np.zeros(6)
  r = mr.SE3.exp(v)
  se3 = mr.SE3.set_mat(r)  
  
  for i in range(3):
    vel = np.random.rand(i,6)

    res = mr.CMTM[mr.SE3](se3, vel)
    
    mat = np.eye(6*(i+1))

    np.testing.assert_allclose(res.tangent_adj_mat() @ res.tangent_adj_mat_inv(), mat, rtol=1e-15, atol=1e-15)
    
def test_cmtm_se3_matmul():
  v1, v2 = np.random.rand(2,6) 
  r1 = mr.SE3.exp(v1)
  r2 = mr.SE3.exp(v2)
  
  mat1 = mr.CMTM[mr.SE3](mr.SE3.set_mat(r1))
  mat2 = mr.CMTM[mr.SE3](mr.SE3.set_mat(r2))
  res = mat1@mat2
  
  sol = mr.SE3.set_mat(r1) @ mr.SE3.set_mat(r2)
  
  np.testing.assert_array_equal(res.mat(), sol.mat())
  
def test_cmtm_se3_vec1d_matmul():
  v1, v2 = np.random.rand(2,6) 
  r1 = mr.SE3.exp(v1)
  r2 = mr.SE3.exp(v2)
  
  vel1, vel2 = np.random.rand(1,6), np.random.rand(1,6)
  
  mat1 = mr.CMTM[mr.SE3](mr.SE3.set_mat(r1), vel1)
  mat2 = mr.CMTM[mr.SE3](mr.SE3.set_mat(r2), vel2)
  res = mat1 @ mat2
  
  sol_mat = mr.SE3.set_mat(r1) @ mr.SE3.set_mat(r2)
  sol_vec = np.zeros((1,6))
  sol_vec[0] = mr.SE3.set_mat(r2) @ vel1[0] + vel2[0]
  
  sol = mr.CMTM[mr.SE3](sol_mat, sol_vec)
    
  np.testing.assert_array_equal(res.mat(), sol.mat())
  
def test_cmtm_se3_vec2d_matmul():
  v1, v2 = np.random.rand(2,6) 
  r1 = mr.SE3.exp(v1)
  r2 = mr.SE3.exp(v2)
  
  vel1, vel2 = np.random.rand(2,6), np.random.rand(2,6)
  
  mat1 = mr.CMTM[mr.SE3](mr.SE3.set_mat(r1), vel1)
  mat2 = mr.CMTM[mr.SE3](mr.SE3.set_mat(r2), vel2)
  res = mat1 @ mat2
  
  sol_mat = mr.SE3.set_mat(r1) @ mr.SE3.set_mat(r2)
  sol_vec = np.zeros((2,6))
  sol_vec[0] = mr.SE3.set_mat(r2) @ vel1[0] + vel2[0]
  sol_vec[1] = mr.SE3.set_mat(r2) @ vel1[1] + mr.SE3.hat_adj(mr.SE3.set_mat(r2) @ vel2[0]) @ vel1[0] + vel2[1]
  
  sol = mr.CMTM[mr.SE3](sol_mat, sol_vec)
    
  np.testing.assert_array_equal(res.mat(), sol.mat())