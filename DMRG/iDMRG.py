'''
=============================================
Infinite density matrix renormalization group
=============================================

iDMRG, including:
    * classes: iDMRG
'''

__all__=['iDMRG']

#import os
#import re
import numpy as np
#import pickle as pk
#import itertools as it
#import scipy.sparse as sp
#import HamiltonianPy.Misc as hm
#import matplotlib.pyplot as plt
#from numpy.linalg import norm
from HamiltonianPy import *
from HamiltonianPy.TensorNetwork import *
#from copy import copy,deepcopy

class iDMRG(Engine):
    '''
    Infinite density matrix renormalization group method.

    Attributes
    ----------
    mps : MPS
    mpo : MPO
    '''

    def __init__(self,mps,mpo):
        '''
        Constructor.

        Parameters:
            mps : MPS
            mpo : MPO
        '''
        self.mps=mps
        self.mpo=mpo

    def iterate(self):
        '''
        '''
        pass
