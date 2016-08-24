'''
'''

__all__=['BMatrix','dot']


import numpy as np
from numpy.linalg import svd

class BMatrix(object):
    '''
    '''
    
    def __init__(self,U,S,V):
        self.U=U
        self.S=S
        self.V=V

    def __str__(self):
        return 'U,S,V:\n%s\n%s\n%s'%(self.U,self.S,self.V)

    @property
    def M(self):
        return np.einsum('ij,j,jk->ik',self.U,self.S,self.V)

def dot(M,B):
    buff=np.einsum('ij,jk,k->ik',M,B.U,B.S)
    U,S,V=svd(buff)
    V=V.dot(B.V)
    return BMatrix(U,S,V)


