import numpy as np

from scipy.linalg import expm
from scipy import integrate

import mathrobo as mr

def test_se3():
  res = mr.SE3()
  e = np.identity(4)

  np.testing.assert_array_equal(res.mat(), e)
  
def test_se3_inv():
  v = np.random.rand(6)
  v[0:3] = v[0:3] / np.linalg.norm(v[0:3])
  r = mr.SO3.exp(v[0:3]) 
  
  mat = mr.SE3(r, v[3:6])
  
  res = mat.mat() @ mat.inverse()
  
  e = np.identity(4)
  
  np.testing.assert_allclose(res, e, rtol=1e-15, atol=1e-15)
  
def test_se3_adj():
  v = np.random.rand(6)
  v[0:3] = v[0:3] / np.linalg.norm(v[0:3])
  r = mr.SO3.exp(v[0:3]) 
  
  res = mr.SE3(r, v[3:6])
  
  m = np.zeros((6,6))
  m[0:3, 0:3] = r 
  m[3:6,0:3] = mr.SO3.hat(v[3:6])@r
  m[3:6, 3:6] = r 
  
  np.testing.assert_allclose(res.adjoint(), m)
  
def test_se3_set_adj():
  v = np.random.rand(6)
  r = mr.SO3.exp(v[0:3]) 
  mat = mr.SE3(r, v[3:6])
  
  res = mr.SE3()
  res.set_adj_mat(mat.adjoint())
  
  np.testing.assert_allclose(res.mat(), mat.mat())
  
def test_se3_adj_inv():
  v = np.random.rand(6)
  v[0:3] = v[0:3] / np.linalg.norm(v[0:3])
  r = mr.SO3.exp(v[0:3]) 
  
  mat = mr.SE3(r, v[3:6])
  
  res = mat.adjoint() @ mat.adj_inv()
  
  e = np.identity(6)
  
  np.testing.assert_allclose(res, e, rtol=1e-15, atol=1e-15)

def test_se3_hat():
  v = np.random.rand(6)  
  m = np.array([[0., -v[2], v[1], v[3]],
                [v[2], 0., -v[0], v[4]],
                [-v[1], v[0], 0., v[5]],
                [  0.,  0.,  0.,  0.]])
  
  res = mr.SE3.hat(v)

  np.testing.assert_array_equal(res, m)
  
def test_se3_hat_commute():
  v1 = np.random.rand(6)
  v2 = np.append(np.random.rand(3), 0.)

  res1 = mr.SE3.hat(v1) @ v2
  res2 = mr.SE3.hat_commute(v2) @ v1
  
  np.testing.assert_array_equal(res1, res2)
  
def test_se3_vee():
  v = np.random.rand(6)
  
  hat = mr.SE3.hat(v)
  res = mr.SE3.vee(hat)
  
  np.testing.assert_array_equal(v, res)
  
def test_se3_exp():
  v = np.random.rand(6)
  a = np.random.rand(1)
  res = mr.SE3.exp(v, a)

  m = expm(a*mr.SE3.hat(v))
  
  np.testing.assert_allclose(res, m)
  
def test_se3_exp_integ():
  v = np.random.rand(6)
  a = np.random.rand(1)
  res = mr.SE3.exp_integ(v, a)

  def integrad(s):
    return expm(s*mr.SE3.hat(v))
  
  m, _ = integrate.quad_vec(integrad, 0, a)
  
  m[3,3] = m[3,3] / a
  
  np.testing.assert_allclose(res, m)
  
def test_se3_hat_adj():
  v = np.random.rand(6)  
  m = np.array([[0., -v[2], v[1], 0., 0., 0.],
                [v[2], 0., -v[0], 0., 0., 0.],
                [-v[1], v[0], 0., 0., 0., 0.],
                [0., -v[5], v[4], 0., -v[2], v[1]],
                [v[5], 0., -v[3], v[2], 0., -v[0]],
                [-v[4], v[3], 0., -v[1], v[0], 0.]])
  
  res = mr.SE3.hat_adj(v)

  np.testing.assert_array_equal(res, m)
  
def test_se3_hat_adj_commute():
  v1 = np.random.rand(6)
  v2 = np.random.rand(6)
  
  res1 = mr.SE3.hat_adj(v1) @ v2
  res2 = mr.SE3.hat_commute_adj(v2) @ v1
  
  np.testing.assert_allclose(res1, res2)
  
def test_se3_exp_adj():
  v = np.random.rand(6)
  a = np.random.rand(1)
  res = mr.SE3.exp_adj(v, a)

  m = expm(a*mr.SE3.hat_adj(v))
  
  np.testing.assert_allclose(res, m)
  
def test_se3_exp_integ_adj():
  vec = np.random.rand(6)
  angle = np.random.rand()

  res = mr.SE3.exp_integ_adj(vec, angle)

  def integrad(s):
    return expm(s*mr.SE3.hat_adj(vec))
  
  m, _ = integrate.quad_vec(integrad, 0, angle)
  
  print(res)
  print(m)
    
  np.testing.assert_allclose(res, m)
  
def test_se3_jac_lie_wrt_scaler():
  v = np.random.rand(6)
  dv = np.random.rand(6)
  a = np.random.rand()
  eps = 1e-8
  
  res = mr.jac_lie_wrt_scaler(mr.SE3, v, a, dv)
  
  r = mr.SE3.exp(v, a)
  v_ = v + dv*eps
  r_ = mr.SE3.exp(v_, a)
  
  dr = (r_ - r) / eps
  
  np.testing.assert_allclose(res, dr, 1e-3)
  
def test_se3_jac_lie_wrt_scaler_integ():
  v = np.random.rand(6)
  dv = np.random.rand(6)
  a = np.random.rand()
  eps = 1e-8
  
  def integrad(s):
    return mr.jac_lie_wrt_scaler(mr.SE3, v, s, dv)
  
  res, _ = integrate.quad_vec(integrad, 0, a)
  
  r = mr.SE3.exp_integ(v, a)
  v_ = v + dv*eps
  r_ = mr.SE3.exp_integ(v_, a)
  
  dr = (r_ - r) / eps
  
  np.testing.assert_allclose(res, dr, 1e-3)

def test_se3_jac_adj_lie_wrt_scaler():
  v = np.random.rand(6)
  dv = np.random.rand(6)
  a = np.random.rand()
  eps = 1e-8
  
  res = mr.jac_adj_lie_wrt_scaler(mr.SE3, v, a, dv)
  
  r = mr.SE3.exp_adj(v, a)
  v_ = v + dv*eps
  r_ = mr.SE3.exp_adj(v_, a)
  
  dr = (r_ - r) / eps
  
  np.testing.assert_allclose(res, dr, 1e-4)
  
def test_se3_adj_jac_adj_lie_wrt_scaler_integ():
  v = np.random.rand(6)
  dv = np.random.rand(6)
  a = np.random.rand()
  eps = 1e-8
  
  def integrad(s):
    return mr.jac_adj_lie_wrt_scaler(mr.SE3, v, s, dv)
  
  res, _ = integrate.quad_vec(integrad, 0, a)
  
  r = mr.SE3.exp_integ_adj(v, a)
  v_ = v + dv*eps
  r_ = mr.SE3.exp_integ_adj(v_, a)
  
  dr = (r_ - r) / eps
  
  np.testing.assert_allclose(res, dr, 1e-3)