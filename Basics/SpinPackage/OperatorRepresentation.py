'''
Spin operator representation, including:
1) functions: s_opt_rep
'''

__all__=['s_opt_rep']

from numpy import *
from scipy.sparse import kron

def s_opt_rep(operator,table):
    '''
    This function returns the csr_formed sparse matrix representation of an operator on the occupation number basis.
    '''
    temp=[eye(index.S+1) for index in sorted(table.keys(),key=table.get)]
    for spin,seq in zip(operator.spins,operator.seqs):
        temp[seq]=asarray(spin)
    result=operator.value
    for matrix in temp:
        result=kron(result,matrix,format='csr')
        result.eliminate_zeros()
    return result
