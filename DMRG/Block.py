'''
Block for DMRG algorithm, including:
1) classes: Block
'''

__all__=['Block']

import numpy as np
from ..Basics import *
from copy import copy

class Block(set):
    '''
    '''

    def __init__(self,indices,form,optstrs,qncs,us):
        '''
        '''
        self.update(indices)
        self.form=form
        self.optstrs=optstrs
        self.qncs=qncs
        self.us=us
        self.cache={}

    @property
    def H(self):
        '''
        '''
        if 'H' in self.cache:
            return self.cache['H']
        else:
            if len(self)==0:
                result=Tensor(1.0,labels=[])
            else:
                result=Tensor(0.0,labels=[]) if len(self.optstrs)==0 else 0.0
                for optstr in self.optstrs:
                    result+=optstr.matrix(self.us,self.form)
            self.cache['H']=result
            return result

    @staticmethod
    def empty(form):
        '''
        '''
        return Block(indices=[],form=form,optstrs=[],qncs=[],us=[])

    @staticmethod
    def single(index,optstrs,qnc):
        '''
        '''
        return Block(indices=[index],form=None,optstrs=optstrs,qncs=[qnc],us=[])

    def union(self,other,connections,degfres,layer):
        '''
        '''
        assert len(other)==1 and other.form==None
        indices=self|other
        optstrs=self.optstrs+other.optstrs+connections
        qncs=copy(self.qncs)
        us=copy(self.us)
        if self.form=='L':
            form='L'
            qncs.append(self.qncs[-1].tensorsum(other.qncs[0],history=True))
        elif self.form=='R':
            form='R'
            qncs.insert(0,other.qncs[0].tensorsum(self.qncs[0],history=True))
        else:
            raise ValueError("Block union error: the form of the first block(%s) is not 'L' or 'R'."%(self.form))
        result=Block(indices=indices,form=form,optstrs=optstrs,qncs=qncs,us=us)
        #result.cache['H']=
        return result

    def trancation(self,):
        pass
