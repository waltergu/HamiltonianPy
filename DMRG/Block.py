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

    def __init__(self,form,degfres,optstrs,qns,us):
        '''
        '''
        self.form=form
        self.update(degfres)
        self.optstrs=optstrs
        self.qns=qns
        self.us=us
        self.cache={}

    def H(self,degfres):
        '''
        '''
        if 'H' in self.cache:
            return self.cache['H']
        else:
            result=0
            for optstr in self.optstrs:
                result+=optstr.matrix(self.us,self.form)
            return result
