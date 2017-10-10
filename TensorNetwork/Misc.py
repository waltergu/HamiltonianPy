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
    See http://iopscience.iop.org/article/10.1088/1367-2630/aa6a4b/meta;jsessionid=0CED3A73D4B3970070E6216EC3AA6BF1.c1.iopscience.cld.iop.org
    '''
    assert mps1.nsite==mps2.nsite
    if sys.start in (None,0):
        return
