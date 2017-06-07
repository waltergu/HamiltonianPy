'''
----------------------------
Spin operator representation
----------------------------

Spin operator representation, including:
    * functions: soptrep
'''

__all__=['soptrep']

from numpy import *
from ...Misc import kron

def soptrep(operator,table,**options):
    '''
    This function returns the csr_formed sparse matrix representation of an operator.

    Parameters
    ----------
    operator : SOperator
        The operator whose matrix representation is wanted.
    table : Table
        The index-sequence table.
    options : dict, optional
        * entry 'cut': integer
            The position where the spin string is cut.
        * entry 'permutations': 2-tuple of 1d ndarray
            The permutations for the left/right part of the spin string after the cut.
        * rcs,timers:
            See Hamiltonian.Misc.kron for details.

    Returns
    -------
    csr_matrix
        The matrix representation of the input operator.
    '''
    temp=[eye(int(index.S*2)+1 if hasattr(index,'S') else 2) for index in sorted(table.keys(),key=table.get)]
    for index,spin in zip(operator.indices,operator.spins):
        temp[table[index]]=asarray(spin)
    if options.get('cut',None) is None:
        result=array(operator.value)
        for matrix in temp:
            result=kron(result,matrix)
    else:
        cut,permutations,rcs,timers=options.get('cut'),options.get('permutations'),options.get('rcs'),options.get('timers',None)
        m1=array(operator.value)
        for matrix in temp[:cut]:
            m1=kron(m1,matrix)
        m2=ones_like(operator.value)
        for matrix in temp[cut:]:
            m2=kron(m2,matrix)
        m1=m1[permutations[0][:,None],permutations[0]]
        m2=m2[permutations[1][:,None],permutations[1]]
        result=kron(m1,m2,rcs,timers)
    return result
