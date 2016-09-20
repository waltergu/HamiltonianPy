'''
Block for DMRG algorithm, including:
1) classes: Block
'''

__all__=['Block']

import numpy as np
from ..Basics import *
from ..Math import kron
from copy import copy

class Block(object):
    '''
    '''
    def __init__(self,form,length,lattice,config,terms,qns,us,H):
        '''
        Constructor.
        '''
        self.form=form
        self.length=length
        self.lattice=lattice
        self.config=config
        self.table=config.table()
        self.terms=terms
        self.qns=qns
        self.us=us
        self.H=H

    def union(self,other,target=None):
        '''
        The union of two block.
        '''
        form=other.form if self.form is None else (self.form if (self.form==other.form or other.form is None) else None)
        length=self.length+other.length
        lattice=Lattice(name=self.name,points=self.values()+other.values(),nneighbour=self.nneighbour)
        config=copy(self.config)
        config.update(other.config)
        terms=self.terms
        qns=self.qns+other.qns
        us=self.us.extend(other.us)
        connections=Generator(
                bonds=      [bond for bond in lattice.bonds if (bond.spoint.pid in self.lattice and bond.epoint.pid in other.lattice) or (bond.spoint.pid in other.lattice and bond.epoint.pid in self.lattice)],
                config=     config,
                terms=      terms
                )
        H=0
        for opt in connections.operator.values():
            value,opt1,opt2=opt.decompose(self.table,other.table)
            m1=OptStr.from_operator(value*opt1,self.table).matrix(self.us,form=self.form)
            m2=OptStr.from_operator(opt2,other.table).matrix(other.us,form=other.form)
            H+=kron(m1,m2,self.qns,other.qns,qns,target)
        H+=kron(self.H,other.H,self.qns,other.qns,qns,target)
        return Block(length=length,lattice=lattice,config=config,terms=terms,qns=qns,us=us,H=H)

    def truncate(self,u):
        pass
