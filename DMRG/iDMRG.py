'''
Infinite DMRG, including:
1) functions: TwoSiteLattices
2) classes: iDMRG
'''

__all__=['TwoSiteLattices','iDMRG']

import numpy as np
from ..Basics import *
from ..Math import Label
from MPS import *
from MPO import *
from Chain import *

def TwoSiteLattices(scopes,block,vector):
    '''
    
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
    '''

    def __init__(self,name,block,vector,terms,config,degfres,chain,**karg):
        '''
        Constructor.
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
        for lattice in TwoSiteLattices(scopes=scopes,block=self.block,vector=self.vector):
            optstrs=[]
            for operator in Generator(bonds=lattice.bonds,config=self.config,terms=self.terms,dtype=np.float64).operators.values():
                optstrs.append(OptStr.from_operator(operator,degfres=self.degfres,layer=self.degfres.layers[0]))
            A,B=indices.pop(0),indices.pop(-1)
            AL=Label([('layer',self.degfres.layers[0]),('tag',A)],[('qnc',self.degfres[A])])
            BL=Label([('layer',self.degfres.layers[0]),('tag',B)],[('qnc',self.degfres[B])])
            self.chain.two_site_grow(AL,BL,optstrs)
