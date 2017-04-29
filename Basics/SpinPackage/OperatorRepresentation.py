'''
----------------------------
Spin operator representation
----------------------------

Spin operator representation, including:
    * functions: s_opt_rep
'''

__all__=['s_opt_rep']

from numpy import *
from scipy.sparse import kron

def s_opt_rep(operator,table):
    '''
    This function returns the csr_formed sparse matrix representation of an operator on the occupation number basis.
    '''
    temp=[eye(int(index.S*2)+1) for index in sorted(table.keys(),key=table.get)]
    for index,spin in zip(operator.indices,operator.spins):
        temp[table[index]]=asarray(spin)
    result=operator.value
    for matrix in temp:
        result=kron(result,matrix,format='csr')
        result.eliminate_zeros()
    return result
