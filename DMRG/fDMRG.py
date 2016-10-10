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

    def __init__(self,name,lattice,terms,config,degfres,mps=None,**karg):
        '''
        '''
        self.name=name
        self.lattice=lattice
        self.terms=terms
        self.config=config
        self.degfres=degfres
        self.mps=mps
        self.generators={'h':Generator(bonds=self.lattice.bonds,config=self.config,terms=self.terms)}
        self.optstrs={layer:{'h':[]} for layer in degfres.layers}
        self.blocks={layer:{'A':[],'B':[]} for layer in degfres.layers}
        self.connections={}
        self.cache={}

    def set_optstrs(self,layer):
        '''
        '''
        for key,generator in self.generators.items():
            for operator in generator.operators.values():
                self.optstrs[layer][key].append(OptStr.from_operator(operator,self.degfres,layer))

    def init_blocks(self,layer):
        '''
        '''
        pass
