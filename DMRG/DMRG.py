'''
DMRG.
'''

__all__=['Block','IDMRG']

from ..Math.Tensor import *
from ..Math.MPS import *
from ..Basics import *
from scipy.sparse import kron,identity
from copy import copy,deepcopy

class Block(object):
    '''
    '''
    def __init__(self,length,lattice,config,basis,mps,H):
        self.length=length
        self.lattice=lattice
        self.config=config
        self.basis=basis
        self.mps=mps
        self.H=H

    def expansion(self,point,internal):
        length=self.length+1
        lattice=deepcopy(self.lattice).expand(point)
        config=copy(self.config)
        config[point.pid]=internal
        basis=self.basis
        H=kron(self.H,identity(internal.ns))
        return Block(length,lattice,config,basis,H)

class IDMRG(Engine):
    '''
    '''
    def __init__(self,name,lattice,config,term,**karg):
        self.name=name
        self.lattice=lattice
        self.config=config
        self.table=config.table
        self.term=term
        self.sys=Block()
        self.env=Block()

    def proceed(self):
        self.enlarge_lattice()
        self.find_operators()
        self.set_matrix()
        self.find_gs()
        self.truncate()

    def enlarge_lattice(self):
        sp,ep=self.lattice.point.values[0],self.lattice.point.values[0]
        
