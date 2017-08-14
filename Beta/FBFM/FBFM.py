'''
Spin wave theory for spin reserved flat band ferromagnets.
    * constants: FBFM_PRIORITY
    * classes: 
    * functions: 
'''

__all__=[]

import numpy as np
import HamiltonianPy as HP
import scipy.linalg as sl
import matplotlib.pyplot as plt

FBFM_PRIORITY=('spin','scope','nambu','site','orbital')

class FBFM(HP.Engine):
    def __init__(self,lattice=None,config=None,terms=None,interactions=None,bz=None,**karg):
        assert config.priority==FBFM_PRIORITY
        self.lattice=lattice
        self.config=config
        self.terms=terms
        self.interactions=interactions
        self.bz=bz
        self.generator=HP.Generator(bonds=lattice.bonds,config=config,table=config.table(mask=['nambu']),terms=terms,half=True)
        self.igenerator=HP.Generator(bonds=lattice.bonds,config=config,table=config.table(mask=['spin','nambu']),terms=interactions,half=False)
        self.status.update(const=self.generator.parameters['const'],alter=self.generator.parameters['alter'])
        self.status.update(const=self.igenerator.parameters['const'],alter=self.igenerator.parameters['alter'])
        self.spdiagonalize()

    @property
    def nsp(self):
        return len(self.generator.table)

    @property
    def nk(self):
        return 1 if self.bz is None else len(self.bz.mesh('k'))

    def spdiagonalize(self):
        def matrix(k=[]):
            result=np.zeros((self.nsp,self.nsp),dtype=np.complex128)
            for opt in self.generator.operators.values():
                result[opt.seqs]+=opt.value*(1 if len(k)==0 else np.exp(-1j*np.inner(k,opt.rcoords[0])))
            result+=conjugate(result.T)
            return result
        dwesmesh,dwvsmesh=[],[]
        upesmesh,upvsmesh=[],[]
        for k in self.bz.mesh('k'):
            m=matrix(k)
            es,vs=sl.eigh(m[:self.nsp/2,:self.nsp/2],eigvals=(0,self.nsp/4))
            dwesmesh.append(es)
            dwvsmesh.append(vs)
            es,vs=sl.eigh(m[self.nsp/2:,self.nsp/2:],eigvals=(0,self.nsp/4))
            upesmesh.append(es)
            upvsmesh.append(vs)
        self.esmesh=np.array([dwesmesh,upesmesh])
        self.vsmesh=np.array([dwvsmesh,upvsmesh])

    def update(self,**karg):
        self.generator.update(**karg)
        self.igenerator.update(**karg)
        self.status.update(alter=karg)
        self.spdiagonalize()

    def matrix(self,k=None,**karg):
        self.update(**karg)
        result=np.zeros((self.nk*self.nsp**2/16,self.nk*self.nsp**2/16),dtype=np.complex128)
        pass
