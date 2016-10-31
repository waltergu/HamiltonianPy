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
        scopes: list of string
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
    '''

    def __init__(self,name,block,vector,terms,config,degfres,chain,**karg):
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
        '''
        self.name=name
        self.block=block
        self.vector=vector
        self.terms=terms
        self.config=config
        self.degfres=degfres
        self.chain=chain
        indices=sorted(self.degfres.indices(layer=self.degfres.layers[0]),key=lambda index: index.to_tuple(priority=self.degfres.priority))
        scopes=[index.pid.scope for index in indices]
        for lattice in TwoSiteGrowOfLattices(scopes=scopes,block=self.block,vector=self.vector):
            optstrs=[]
            for operator in Generator(bonds=lattice.bonds,config=self.config,terms=self.terms,dtype=np.float64).operators.values():
                optstrs.append(OptStr.from_operator(operator,degfres=self.degfres,layer=self.degfres.layers[0]))
            A,B=indices.pop(0),indices.pop(-1)
            AL=Label([('layer',self.degfres.layers[0]),('tag',A)],[('qnc',self.degfres[A])])
            BL=Label([('layer',self.degfres.layers[0]),('tag',B)],[('qnc',self.degfres[B])])
            self.chain.two_site_grow(AL,BL,optstrs)
        self.lattice=lattice
