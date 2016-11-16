'''
Infinite DMRG, including:
1) functions: TwoSiteGrowOfLattices
2) classes: iDMRG
'''

__all__=['TwoSiteGrowOfLattices','iDMRG']

import numpy as np
from ..Basics import *
from ..Math import Label
from MPS import *
from MPO import *
from Chain import *

def TwoSiteGrowOfLattices(scopes,block,vector):
    '''
    Return a generator over a sequence of SuperLattice with blocks added two-by-two in the center.
    Parameters:
        scopes: list of hashable objects
            The scopes of the blocks to be added two-by-two into the SuperLattice.
        block: Lattice
            The building block of the SuperLattice.
        vector: 1d ndarray
            The translation vector of the left blocks and right blocks before the addition of two new ones.
    Returns: generator
        A generator over the sequence of SuperLattice with blocks added two-by-two in the center.
    '''
    assert len(scopes)%2==0
    for i in xrange(len(scopes)/2):
        A=scopes.pop(0)
        B=scopes.pop(-1)
        if i==0:
            aps=[Point(p.pid._replace(scope=A),rcoord=p.rcoord-vector/2,icoord=p.icoord) for p in block.values()]
            bps=[Point(p.pid._replace(scope=B),rcoord=p.rcoord+vector/2,icoord=p.icoord) for p in block.values()]
        else:
            aps=[Point(p.pid,rcoord=p.rcoord-vector,icoord=p.icoord) for p in aps]
            bps=[Point(p.pid,rcoord=p.rcoord+vector,icoord=p.icoord) for p in bps]
            aps.extend([Point(p.pid._replace(scope=A),rcoord=p.rcoord-vector/2,icoord=p.icoord) for p in block.values()])
            bps.extend([Point(p.pid._replace(scope=B),rcoord=p.rcoord+vector/2,icoord=p.icoord) for p in block.values()])
        yield SuperLattice.compose(name=block.name,points=aps+bps,vectors=block.vectors,nneighbour=block.nneighbour,max_coordinate_number=block.max_coordinate_number)

class iDMRG(Engine):
    '''
    Infinite density matrix renormalization group method.
    Attribues:
        name: sting
            The name of the iDMRG.
        block: Lattice
            The building block of the iDMRG.
        vector: 1d ndarray
            The translation vector of the left blocks and right blocks before the addition of two new ones.
        lattice: SuperLattice
            The final lattice of the iDMRG.
        terms: list of Term
            The terms of the iDMRG.
        config: IDFConfig
            The configuration of the internal degrees of freedom on the lattice.
        degfres: DegFreTree
            The physical degrees of freedom tree.
        chain: Chain
            The chain of the iDMRG.
        mask: [] or ['nambu']
            [] for spin systems and ['nambu'] for fermionic systems.
        dtype: np.float64, np.complex128
            The data type.
    '''

    def __init__(self,name,block,vector,terms,config,degfres,chain,mask=[],dtype=np.complex128,**karg):
        '''
        Constructor.
        Parameters:
            name: sting
                The name of the iDMRG.
            block: Lattice
                The building block of the iDMRG.
            vector: 1d ndarray
                The translation vector of the left blocks and right blocks before the addition of two new ones.
            terms: list of Term
                The terms of the iDMRG.
            config: IDFConfig
                The configuration of the internal degrees of freedom on the lattice.
            degfres: DegFreTree
                The physical degrees of freedom tree.
            chain: Chain
                The initial chain.
            mask: [] or ['nambu']
                [] for spin systems and ['nambu'] for fermionic systems.
            dtype: np.float64,np.complex128, optional
                The data type.
        '''
        self.name=name
        self.block=block
        self.vector=vector
        self.terms=terms
        self.config=config
        self.degfres=degfres
        self.chain=chain
        self.mask=mask
        self.dtype=dtype

    def grow(self,scopes,targets=None,nmax=200,tol=5*10**-14):
        '''
        Two site growth of the idmrg.
        Parameters:
            scopes: list of hashable objects
                The scopes of the blocks to be added two-by-two into the chain.
            targets: sequence of QuantumNumber,optional
                The target space at each growth of the chain.
            nmax: integer
                The maximum singular values to be kept.
            tol: float64
                The tolerance of the singular values.
        '''
        layer=self.degfres.layers[0]
        for i,(lattice,target) in enumerate(zip(TwoSiteGrowOfLattices(scopes=scopes,block=self.block,vector=self.vector),targets)):
            QuantumNumberCollection.history.clear()
            self.config.reset(pids=lattice)
            self.degfres.reset(leaves=self.config.table(mask=self.mask).keys())
            operators=Generator(bonds=lattice.bonds,config=self.config,terms=self.terms,dtype=self.dtype).operators
            if self.mask==['nambu']:
                for operator in operators.values():
                    operators+=operator.dagger
            optstrs=[OptStr.from_operator(operator,degfres=self.degfres,layer=layer) for operator in operators.values()]
            indices=sorted(self.degfres.indices(layer=layer),key=lambda index: index.to_tuple(priority=self.degfres.priority))
            AL=Label([('layer',layer),('tag',indices[i])],[('qnc',self.degfres[indices[i]])])
            BL=Label([('layer',layer),('tag',indices[i+1])],[('qnc',self.degfres[indices[i+1]])])
            self.chain.two_site_grow(AL,BL,optstrs,target,nmax=nmax,tol=tol)
        self.lattice=lattice
