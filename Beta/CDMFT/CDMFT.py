'''
====================================
Cellular dynamical mean field theory
====================================

CDMFT, including:
    * classes: CDMFT
    * functions: 
'''

__all__=['CDMFT']

import numpy as np
import HamiltonianPy as HP
import HamiltonianPy.ED as ED
import HamiltonianPy.Misc as HM
import matplotlib.pyplot as plt

class CDMFT(ED.ED):
    '''
    This class implements the algorithm of cellular dynamical mean field theory for fermionic systems.

    Attributes
    ----------
    '''

    def __init__(self,**karg):
        pass
