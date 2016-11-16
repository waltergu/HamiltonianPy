'''
Finite DMRG.
'''

__all__=['fDMRG']

from ..Math.Tensor import *
from ..Basics import *
from MPS import *
from MPO import *

class fDMRG(Engine):
    '''
    '''

    def __init__(self,name,lattice,terms,config,degfres,chain,**karg):
        '''
        '''
        self.name=name
        self.lattice=lattice
        self.terms=terms
        self.config=config
        self.degfres=degfres
        self.chain=chain

    def sweep(self,nstates):
        '''
        '''
        for nstate in nstates:
            while self.chain.cut>1:
                self.chain.two_site_sweep(direction='L',nmax=nstate)
            while self.chain.cut<self.chain.nsite-1:
                self.chain.two_site_sweep(direction='R',nmax=nstate)
