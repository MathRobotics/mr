from typing import TypeVar, Generic

from ..basic import *
from lie_abst import *

T = TypeVar('T')

class CMTM(Generic[T]):
  def __init__(self, lie_mat, lie_vec = np.array([]), LIB = 'numpy'): 
    '''
    Constructor
    '''
    self._mat = lie_mat
    self._vec = lie_vec
    self._dof = lie_mat.shape[0]
    self._n = lie_vec.shape[0] + 1
    self.lib = LIB
    
  def __mat_elem(self, p):
    if p == 0:
      return self._mat
    else:
      mat = zeros( (self._dof, self._dof) ) 
      for i in range(p):
        mat = mat + self.__mat_elem(p-i-1) @ T.hat(self._vec[i])
        
      return mat / p
    
  def mat(self):
    mat = identity(self._dof * self._n)
    for i in range(self._n):
      for j in range(self._n):
        if i > j :
          mat[self._dof*i:self._dof*j] = self.__mat_elem(abs(i-j))
    return mat
  
  @staticmethod
  def set_mat(mat):
    pass
  
  def lie_mat(self):
    return self._mat
  
  def lie_vec(self, i):
    return self._vec[i]
  
  def inverse(self):
    pass