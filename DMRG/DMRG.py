'''
DMRG.
'''

from ..Math.Tensor import *
from ..Math.MPS import *
from ..Basics import *

class Block(object):
    '''
    '''
    def __init__(self,length,basis,H):
        self.length=length
        self.basis=basis
        self.H=H

class IDMRG(engine):
    '''
    '''
    def __init__(self,name,lattice,config,term,**karg):
        self.name=name
        self.lattice=lattice
        self.config=config
        
