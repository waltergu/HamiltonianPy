'''
Block for DMRG algorithm, including:
1) classes: Block
'''

__all__=['Block']

import numpy as np
from ..Basics import *
#from ..Math import kron
from copy import copy

class Block(object):
    '''
    '''

    def __init__(self,id,operators,qns,us):
        '''
        '''
        self.id=id
        self.operators={'h':operators}
        self.qns=qns
        self.us=us
        self.cache={}

    def H(self,degfres,layer):
        '''
        '''
        if 'H' in self.cache:
            return self.cache['H']
        else:
            result=0
            for operator in self.operators['h']:
                result+=OptStr.from_operator(operator,degfres,layer).matrix(self.us)
            return result



class BlockConfig(object):
    '''
    '''

    def __init__(self,form,config,mode='PID'):
        self.form=form
        self.config=config
        self.table=config.table
        self.mode=mode
        self.labels=[]
        self.map={}
        if self.mode=='PID':
            self.length=len(self.config)
            for i,pid in enumerate(sorted(self.config)):
                L,S,R=Label(i),Label(pid),Label((i+1)%self.length)
                self.labels.append((L,S,R))
            for index in self.table:
                self.map[index]=index
        elif self.mode=='INDEX':
            self.length=len(self.table)
            for i,index in enumerate(sorted(table,key=table.get)):
                L,S,R=Label(i),Label(index),Label((i+1)%self.length)
                self.labels.append((L,S,R))
                self.map[index]=S
        else:
            raise ValueError("BlockConfig error: mode(%s) not supported."%mode)
        self.shapes=[[None,None,None]]*self.length

    def __add__(self,other):
        assert self.mode==other.mode
        form=other.form if self.form is None else (self.form if (self.form==other.form or other.form is None) else None)
        config=self.config+other.config
        result=BlockConfig(form,config,mode=self.mode)
        result.shapes=[]
        result.shapes.extend(self.shapes)
        result.shapes.extend(other.shapes)

class hhBlock(object):
    '''
    '''
    def __init__(self,bconfig,lattice,terms,qns,us,H):
        '''
        Constructor.
        '''
        self.bconfig=bconfig
        self.lattice=lattice
        self.terms=terms
        self.qns=qns
        self.us=us
        self.H=H

    def union(self,other,target=None):
        '''
        The union of two block.
        '''

        lattice=Lattice(name=self.name,points=self.values()+other.values(),nneighbour=self.nneighbour)
        bconfig=self.bconfig+other.bconfig
        terms=self.terms
        qns=self.qns+other.qns
        us=MPS.compose(self.us,other.us)
        connections=Generator(
                bonds=      [bond for bond in lattice.bonds if (bond.spoint.pid in self.lattice and bond.epoint.pid in other.lattice) or (bond.spoint.pid in other.lattice and bond.epoint.pid in self.lattice)],
                config=     bconfig.config,
                terms=      terms
                )
        H=0
        for opt in connections.operator.values():
            value,opt1,opt2=opt.decompose(self.bconfig.table,other.bconfig.table)
            m1=OptStr.from_operator(value*opt1,self.bconfig.map).matrix(self.us,form=self.bconfig.form)
            m2=OptStr.from_operator(opt2,other.bconfig.map).matrix(other.us,form=other.bconfig.form)
            H+=kron(m1,m2,self.qns,other.qns,qns,target)
        H+=kron(self.H,other.H,self.qns,other.qns,qns,target)
        return Block(bconfig=bconfig,lattice=lattice,terms=terms,qns=qns,us=us,H=H)

    def truncate(self,u,qns):
        '''
        '''
        self.qns=qns
        if self.bconfig.form=='L':
            self.bconfig.shapes[-1][2]=u.shape[1]
            self.us[-1]=Tensor(u.reshape(),labels=self.bconfig.labels[-1])
            self.H=u.T.conjugate().dot(self.H).dot(u)
        elif self.bconfig.form=='R':
            pass
        else:
            raise ValueError()
