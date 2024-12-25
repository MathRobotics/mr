from typing import TypeVar, Generic

from ..basic import *

T = TypeVar('T')

class CMTM(Generic[T]):
  def __init__(self, elem_mat, elem_vecs = np.array([]), LIB = 'numpy'): 
    '''
    Constructor
    '''
    self._mat = elem_mat
    self._vecs = elem_vecs
    self._dof = elem_mat.mat().shape[0]
    self._n = elem_vecs.shape[0] + 1
    self.lib = LIB
    
  def __mat_elem(self, p):
    if p == 0:
      return self._mat.mat()
    else:
      mat = zeros( (self._dof, self._dof) ) 
      for i in range(p):
        mat = mat + self.__mat_elem(p-(i+1)) @ self._mat.hat(self._vecs[i])

      return mat / p
    
  def mat(self):
    mat = identity(self._dof * self._n)
    for i in range(self._n):
      for j in range(self._n):
        if i >= j :
          mat[self._dof*i:self._dof*(i+1),self._dof*j:self._dof*(j+1)] = self.__mat_elem(abs(i-j))
    return mat
  
  def __adj_mat_elem(self, p):
    if p == 0:
      return self._mat.adj_mat()
    else:
      mat = zeros( (self._dof, self._dof) ) 
      for i in range(p):
        mat = mat + self.__adj_mat_elem(p-(i+1)) @ self._mat.hat_adj(self._vecs[i])
        
      return mat / p
    
  def adj_mat(self):
    mat = identity(self._dof * self._n)
    for i in range(self._n):
      for j in range(self._n):
        if i >= j :
          mat[self._dof*i:self._dof*(i+1),self._dof*j:self._dof*(j+1)] = self.__adj_mat_elem(abs(i-j))
    return mat
  
  def elem_mat(self):
    return self._mat.mat()
  
  def elem_vecs(self, i):
    if(self._n - 1 > i ):
      if(self._vecs.ndim == 1):
        return self._vecs
      else:
        return self._vecs[i]

  def __mat_inv_elem(self, p):
    if p == 0:
      return self._mat.inverse()
    else:
      mat = zeros( (self._dof, self._dof) ) 
      for i in range(p):
        mat = mat - T.hat(self._vecs[i]) @ self.__mat_inv_elem(p-(i+1))
        
      return mat / p
  
  def inverse(self):
    mat = identity(self._dof * self._n)
    for i in range(self._n):
      for j in range(self._n):
        if i >= j :
          mat[self._dof*i:self._dof*(i+1),self._dof*j:self._dof*(j+1)] = self.__mat_inv_elem(abs(i-j))
    return mat
  
  def __mat_adj_inv_elem(self, p):
    if p == 0:
      return self._mat.adj_inv()
    else:
      mat = zeros( (self._dof, self._dof) ) 
      for i in range(p):
        mat = mat - T.hat_adj(self._vecs[i]) @ self.__mat_adj_inv_elem(p-(i+1))
        
      return mat / p
  
  def inverse_adj(self):
    mat = identity(self._dof * self._n)
    for i in range(self._n):
      for j in range(self._n):
        if i >= j :
          mat[self._dof*i:self._dof*(i+1),self._dof*j:self._dof*(j+1)] = self.__mat_adj_inv_elem(abs(i-j))
    return mat
  
  def __tangent_mat_elem(self, p):
    mat = identity( (self._dof, self._dof) ) 
    for i in range(p):
      mat = mat - self.__tangent_mat_elem(p-(i+1)) @ self._mat.hat(self._vecs[i])
    return mat
  
  def tangent_mat(self):
    mat = identity(self._dof * self._n)
    for i in range(self._n):
      for j in range(self._n):
        if i >= j :
          mat[self._dof*i:self._dof*(i+1),self._dof*j:self._dof*(j+1)] = self.__tangent_mat_elem(abs(i-j))
      
  def tangent_mat_inv(self):
    mat = identity(self._dof * self._n)
    for i in range(self._n):
      for j in range(self._n):
        if i >= j :
          mat[self._dof*i:self._dof*(i+1),self._dof*j:self._dof*(j+1)] = self._mat.hat(self._vecs[abs(i-j)])
  
  def __tangent_adj_mat_elem(self, p):
    mat = identity( (self._dof, self._dof) ) 
    for i in range(p):
      mat = mat - self.__tangent_adj_mat_elem(p-(i+1)) @ self._mat.hat_adj(self._vecs[i])
    return mat
  
  def tangent_adj_mat(self):
    mat = identity(self._dof * self._n)
    for i in range(self._n):
      for j in range(self._n):
        if i >= j :
          mat[self._dof*i:self._dof*(i+1),self._dof*j:self._dof*(j+1)] = self.__tangent_adj_mat_elem(abs(i-j))
  
  def tangent_adj_mat_inv(self):
    mat = identity(self._dof * self._n)
    for i in range(self._n):
      for j in range(self._n):
        if i >= j :
          mat[self._dof*i:self._dof*(i+1),self._dof*j:self._dof*(j+1)] = self._mat.hat_adj(self._vecs[abs(i-j)])