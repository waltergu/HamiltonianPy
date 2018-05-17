'''
=============================================
Infinite density matrix renormalization group
=============================================

iDMRG, including:
    * classes: iDMRG
    * function: iDMRGTSG
'''

__all__=['iDMRG','iDMRGTSG']

from .DMRG import *

class iDMRG(DMRG):
    '''
    Infinite density matrix renormalization group.
    '''

    def iterate(self):
        pass

def iDMRGTSG(engine,app):
    '''
    This method iterative update the iDMRG.
    '''
    pass
