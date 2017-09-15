'''
===================================
Miscellaneous classes and functions
===================================

Miscellaneous classes and functions, including:
    * functions: SSF
'''

__all__=['SSF']

def SSF(mps1,mps2,sys):
    '''
    Subsystem fidelity between two mpses.

    Parameters
    ----------
    mps1,mps2 : MPS
        The mpses between which the subsystem fidelity is to be calculated.
    sys : slice
        The system part of the SSF.

    References
    ----------
    See J. -G. Liu, Z. -L. Gu, J. -X. Li and Q.-H. Wang, arXiv:1609.09309.
    '''
    assert mps1.nsite==mps2.nsite
    if sys.start in (None,0):
        return
