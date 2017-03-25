'''
Kronecker product and Kronecker sum for sparse matrices, including
1) functions: kron
'''

__all__=['kron']

import numpy as np
import scipy.sparse as sp
from fkron import *

def kron(m1,m2,rcs=None):
    '''
    Kronecker product of two matrices.
    Parameters:
        m1,m2: 2d ndarray
            The matrices.
        rcs: list of sorted integer, optional
            The wanted rows and cols.
    Returns: csr_matrix
        The product.
    '''
    if rcs is None:
        result=sp.kron(m1,m2,format=format)
    else:
        assert m1.dtype==m2.dtype and m1.shape[0]==m1.shape[1] and m2.shape[0]==m2.shape[1]
        m1=sp.csr_matrix(m1)
        m2=sp.csr_matrix(m2)
        rcs1=np.divide(rcs,m2.shape[0])
        rcs2=np.mod(rcs,m2.shape[0])
        slices=np.zeros(m1.shape[0]*m2.shape[0],dtype=np.int64)
        slices[...]=-1
        slices[rcs]=np.array(xrange(len(rcs)))
        nnz=(m1.indptr[rcs1+1]-m1.indptr[rcs1]).dot(m2.indptr[rcs2+1]-m2.indptr[rcs2])
        if nnz>0:
            if m1.dtype==np.float32:
                data,indices,indptr=fkron_r4(m1.data,m1.indices,m1.indptr,rcs1,m2.data,m2.indices,m2.indptr,rcs2,nnz,slices)
            elif m1.dtype==np.float64:
                data,indices,indptr=fkron_r8(m1.data,m1.indices,m1.indptr,rcs1,m2.data,m2.indices,m2.indptr,rcs2,nnz,slices)
            elif m1.dtype==np.complex64:
                data,indices,indptr=fkron_c4(m1.data,m1.indices,m1.indptr,rcs1,m2.data,m2.indices,m2.indptr,rcs2,nnz,slices)
            elif m1.dtype==np.complex128:
                data,indices,indptr=fkron_c8(m1.data,m1.indices,m1.indptr,rcs1,m2.data,m2.indices,m2.indptr,rcs2,nnz,slices)
            else:
                raise ValueError("_fkron_ error: only matrices with dtype being float32, float64, complex64 or complex128 are supported.")
            result=sp.csr_matrix((data,indices,indptr),shape=(len(rcs),len(rcs)))
        else:
            result=sp.csr_matrix((len(rcs),len(rcs)),dtype=m1.dtype)
        return result
