from typing import TypeVar, Generic

from ..basic import *
from lie_abst import *

T = TypeVar('T')

class CMTM(Generic[T]):
  def __init__(self, elem_mat, elem_vecs = np.array([]), LIB = 'numpy'): 
    '''
    Constructor
    '''
    self._mat = elem_mat
    self._vec = elem_vecs
    self._dof = elem_mat.shape[0]
    self._n = elem_vecs.shape[0] + 1
    self.lib = LIB
    
  def __mat_elem(self, p):
    if p == 0:
      return self._mat
    else:
      mat = zeros( (self._dof, self._dof) ) 
      for i in range(p):
        mat = mat + self.__mat_elem(p-(i+1)) @ T.hat(self._vec[i])
        
      return mat / p
    
  def mat(self):
    mat = identity(self._dof * self._n)
    for i in range(self._n):
      for j in range(self._n):
        if i > j :
          mat[self._dof*i:self._dof*j] = self.__mat_elem(abs(i-j))
    return mat
  
  def __adj_mat_elem(self, p):
    if p == 0:
      return self._mat
    else:
      mat = zeros( (self._dof, self._dof) ) 
      for i in range(p):
        mat = mat + self.__adj_mat_elem(p-(i+1)) @ T.hat_adj(self._vec[i])
        
      return mat / p
    
  def adj_mat(self):
    mat = identity(self._dof * self._n)
    for i in range(self._n):
      for j in range(self._n):
        if i > j :
          mat[self._dof*i:self._dof*j] = self.__adj_mat_elem(abs(i-j))
    return mat
  
  def elem_mat(self):
    return self._mat
  
  def elem_vecs(self, i):
    return self._vec[i]
  
  def inverse(self):
    pass
  
  def __tangent_mat_elem(self, p):
    mat = identity( (self._dof, self._dof) ) 
    for i in range(p):
      mat = mat + self.__tangent_mat_elem(p-(i+1)) @ -T.hat(self._vec[i])
    return mat
  
  def tangent_mat(self):
    mat = identity(self._dof * self._n)
    for i in range(self._n):
      for j in range(self._n):
        if i > j :
          mat[self._dof*i:self._dof*j] = self.__tangent_mat_elem(abs(i-j))
          
  def __tangent_adj_mat_elem(self, p):
    mat = identity( (self._dof, self._dof) ) 
    for i in range(p):
      mat = mat + self.__tangent_adj_mat_elem(p-(i+1)) @ -T.hat_adj(self._vec[i])
    return mat
  
  def tangent_adj_mat(self):
    mat = identity(self._dof * self._n)
    for i in range(self._n):
      for j in range(self._n):
        if i > j :
          mat[self._dof*i:self._dof*j] = self.__tangent_adj_mat_elem(abs(i-j))
      
  def tangent_mat_inv(self):
    mat = identity(self._dof * self._n)
    for i in range(self._n):
      for j in range(self._n):
        if i > j :
          mat[self._dof*i:self._dof*j] = T.hat(self._vec[abs(i-j)])
  
  def tangent_adj_mat_inv(self):
    mat = identity(self._dof * self._n)
    for i in range(self._n):
      for j in range(self._n):
        if i > j :
          mat[self._dof*i:self._dof*j] = T.hat_adj(self._vec[abs(i-j)])