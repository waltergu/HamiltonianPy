'''
Finite DMRG.
'''

__all__=['fDMRG']

from ..Math.Tensor import *
from ..Basics import *
from MPS import *

class fDMRG(Engine):
    '''
    '''

    def __init__(self,name,lattice,terms,config,degfres,mps,**karg):
        '''
        '''
        self.name=name
        self.lattice=lattice
        self.terms=terms
        self.config=config
        self.degfres=degfres
        self.mps=mps
        self.generators={'h':Generator(bonds=self.lattice.bonds,config=self.config,terms=self.terms)}
        self.blocks={'A':[],'B':[]}
        self.connections={}
        self.cache={}

    def set_blocks(self,layer):
        '''
        '''
        block_table=self.degfres.table(layer=layer)
