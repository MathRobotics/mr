import sympy as sp
import numpy as np

from scipy import integrate

import mathrobo as mr

def test_so3_hat():
  x = sp.symbols("x_{0:3}", Integer=True)
  v = sp.Matrix(x)
  
  m = sp.Matrix([[0, -v[2], v[1]],[v[2], 0, -v[0]],[-v[1], v[0], 0]])
  
  res = mr.SO3.hat(v, 'sympy')
  
  assert res == m
  
def test_so3_hat_commute():
  x = sp.symbols("x_{0:3}", Integer=True)
  v = sp.Matrix(x)
  y = sp.symbols("y_{0:3}", Integer=True)
  w = sp.Matrix(y)
  
  res1 = mr.SO3.hat(v, 'sympy') @ w
  res2 = mr.SO3.hat_commute(w, 'sympy') @ v
  
  assert res1 == res2
  
def test_so3_vee():
  x = sp.symbols("x_{0:3}", Integer=True)
  v = sp.Matrix(x)
  
  hat = mr.SO3.hat(v, 'sympy')
  res = mr.SO3.vee(hat, 'sympy')

  assert res == v

def test_so3_mat():
  a = sp.symbols('a')
  x = sp.symbols("x_{0:3}", Integer=True)
  v = sp.Matrix(x)
  
  r = mr.SO3.mat(v, a, 'sympy')

  angle = np.random.rand(1)
  vec = np.random.rand(3)
  
  vec = vec / np.linalg.norm(vec)
  
  res = mr.sympy_subs_mat(r, x, vec)
  res = res.subs([(a, angle[0])]) 
  
  m = mr.SO3.mat(vec, angle)
  
  np.testing.assert_allclose(m, mr.sympy_to_numpy(res))
  
def test_so3_integ_mat():
  a = sp.symbols('a')
  x = sp.symbols("x_{0:3}", Integer=True)
  v = sp.Matrix(x)
  
  r = mr.SO3.integ_mat(v, a, 'sympy')

  angle = np.random.rand(1)
  vec = np.random.rand(3)
  
  vec = vec / np.linalg.norm(vec)

  res = mr.sympy_subs_mat(r, x, vec)
  res = res.subs([(a, angle[0])]) 
  
  m = mr.SO3.integ_mat(vec, angle)
  
  np.testing.assert_allclose(m, mr.sympy_to_numpy(res))
  
def test_so3_integ2nd_mat():
  a = sp.symbols('a')
  x = sp.symbols("x_{0:3}", Integer=True)
  v = sp.Matrix(x)
  
  r = mr.SO3.integ2nd_mat(v, a, 'sympy')

  angle = np.random.rand(1)
  vec = np.random.rand(3)
  
  vec = vec / np.linalg.norm(vec)

  res = mr.sympy_subs_mat(r, x, vec)
  res = res.subs([(a, angle[0])]) 
  
  m = mr.SO3.integ2nd_mat(vec, angle)
  
  np.testing.assert_allclose(m, mr.sympy_to_numpy(res))
  
def test_so3_jac_lie_wrt_scaler():
  a = sp.symbols('a')
  x = sp.symbols("x_{0:3}", Integer=True)
  v = sp.Matrix(x)
  dx = sp.symbols("dx_{0:3}", Integer=True)
  dv = sp.Matrix(dx)
  
  r = mr.jac_lie_wrt_scaler(mr.SO3, v, a, dv, 'sympy')

  angle = np.random.rand(1)
  vec = np.random.rand(3)
  dvec = np.random.rand(3)
  
  vec = vec / np.linalg.norm(vec)

  res = mr.sympy_subs_mat(r, x, vec)
  res = mr.sympy_subs_mat(res, dx, dvec)
  res = res.subs([(a, angle[0])]) 

  m = mr.jac_lie_wrt_scaler(mr.SO3, vec, angle, dvec)
  
  np.testing.assert_allclose(m, mr.sympy_to_numpy(res))

# def test_so3_jac_lie_wrt_scaler_integ():
#   a_ = sp.symbols('a_')
#   a = sp.symbols('a')
#   x = sp.symbols("x_{0:3}", Integer=True)
#   v = sp.Matrix(x)
#   dx = sp.symbols("dx_{0:3}", Integer=True)
#   dv = sp.Matrix(dx)
  
#   r_ = mr.jac_lie_wrt_scaler(mr.SO3, v, a_, dv, 'sympy')
#   r = sp.integrate(r_, [a_, 0, a])
  
#   angle = np.random.rand(1)
#   vec = np.random.rand(3)
#   dvec = np.random.rand(3)
  
#   vec = vec / np.linalg.norm(vec)
  
#   def integrad(s):
#     return mr.jac_lie_wrt_scaler(mr.SO3, vec, s, dvec)
  
#   m, _ = integrate.quad_vec(integrad, 0, angle)

#   res = sympy_subs_mat(r, x, vec)
#   res = sympy_subs_mat(res, dx, dvec)
#   res = res.subs([(a, angle[0])]) 
  
#   np.testing.assert_allclose(m, sympy_to_numpy(res))