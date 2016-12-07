'''
Finite DMRG.
'''

__all__=['fDMRG']

import numpy as np
from ..Math.Tensor import *
from ..Basics import *
from MPS import *
from MPO import *

class fDMRG(Engine):
    '''
    '''

    def __init__(self,name,lattice,terms,config,degfres,chain,mask=[],dtype=np.complex128,**karg):
        '''
        '''
        self.name=name
        self.lattice=lattice
        self.terms=terms
        self.config=config
        self.degfres=degfres
        self.chain=chain
        self.mask=mask
        self.dtype=dtype

    @classmethod
    def from_idmrg(cls,idmrg,**karg):
        '''
        '''
        attrs=['din','dout','name','lattice','terms','config','degfres','chain','mask','dtype']
        return cls(**{attr:karg.get(attr,idmrg.__dict__.get(attr)) for attr in attrs})

    def sweep(self,nstates):
        '''
        '''
        for i,nstate in enumerate(nstates):
            suffix='st'if i==0 else ('nd' if i==1 else ('rd' if i==2 else 'th'))
            while self.chain.cut>1:
                self.chain.two_site_sweep(info='%s %s%s sweep(<<)'%(self.name,i+1,suffix),direction='L',nmax=nstate)
            while self.chain.cut<self.chain.nsite-1:
                self.chain.two_site_sweep(info='%s %s%s sweep(>>)'%(self.name,i+1,suffix),direction='R',nmax=nstate)

    def level_up(self,n=1):
        '''
        '''
        level=self.degfres.level(next(iter(self.chain.table)).identifier)
        assert level+n-1<=len(self.degfres.layers)
        operators=Generator(bonds=self.lattice.bonds,config=self.config,terms=self.terms,dtype=self.dtype).operators
        if self.mask==['nambu']:
            for operator in operators.values():
                operators+=operator.dagger
        optstrs=[OptStr.from_operator(operator,degfres=self.degfres,layer=self.degfres.layers[level+n-1]) for operator in operators.values()]
        self.chain=self.chain.level_up(optstrs=optstrs,degfres=self.degfres,n=n)
