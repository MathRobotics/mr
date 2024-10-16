from ..basic import *

class LieAbstract:

  def __init__(self, LIB = 'numpy'): 
    '''
    Constructor
    '''
    pass

  @staticmethod
  def hat(vec, LIB = 'numpy'):
    '''
    hat operator on the tanget space vector
    '''
    pass
  
  @staticmethod
  def hat_commute(vec, LIB = 'numpy'):
    '''
    hat commute operator on the tanget space vector
    hat(a) @ b = hat_commute(b) @ a 
    '''
    pass

  @staticmethod
  def vee(vec_hat, LIB = 'numpy'):
    '''
    a = vee(hat(a))
    '''
    pass  
  
  @staticmethod
  def mat(vec, a, LIB = 'numpy'):
    pass

  @staticmethod
  def integ_mat(vec, a, LIB = 'numpy'):
    pass
  
  def inverse(self):
    pass
  
  def adjoint(self):
    '''
    adjoint expresion of Lie group
    '''
    pass
  
  @staticmethod
  def adj_hat(vec, LIB = 'numpy'):
    pass

  @staticmethod
  def adj_hat_commute(vec, LIB = 'numpy'):
    pass
  
  @staticmethod
  def adj_mat(vec, a, LIB = 'numpy'):
    pass
  
  @staticmethod
  def adj_integ_mat(vec, a, LIB = 'numpy'):
    pass