'''
Kronecker product and Kronecker sum for sparse matrices, including
1) functions: kron,kronsum
'''

__all__=['kron','kronsum']

import numpy as np
import scipy.sparse as sp
from fkron import *

def kron(m1,m2,rows=None,cols=None,format='csr'):
    '''
    Kronecker product of two matrices.
    Parameters:
        m1,m2: 2d ndarray
            The matrices.
        rows,cols: list of sorted integer, optional
            The wanted rows and cols.
        format: string, optional
            The format of the product.
    Returns: sparse matrix whose format is specified by the parameter format
        The product.
    '''
    if rows is not None and cols is not None:
        if len(rows)>=len(cols):
            result=_fkron_(sp.csc_matrix(m1),sp.csc_matrix(m2),cols)[rows,:].asformat(format)
        else:
            result=_fkron_(sp.csr_matrix(m1),sp.csr_matrix(m2),rows)[:,cols].asformat(format)
    elif rows is not None:
        result=_fkron_(sp.csr_matrix(m1),sp.csr_matrix(m2),rows).asformat(format)
    elif cols is not None:
        result=_fkron_(sp.csc_matrix(m1),sp.csc_matrix(m2),cols).asformat(format)
    else:
        result=sp.kron(m1,m2,format=format)
    return result

def _fkron_(m1,m2,rcs):
    '''
    Python wrapper for fkron_r4, fkron_r8, fkron_c4 and fkron_c8.
    Parameters:
        m1,m2: csr_matrix/csc_matrix
            The matrices.
        rcs: list of sorted integer, optional
            The wanted rows/columns of the result.
    Returns: csr_matrix/csc_matrix
        The result.
    '''
    assert m1.dtype==m2.dtype
    if isinstance(m1,sp.csr_matrix) and isinstance(m2,sp.csr_matrix):
        mode='R'
        rcs1=np.divide(rcs,m2.shape[0])
        rcs2=np.mod(rcs,m2.shape[0])
        cls=sp.csr_matrix
    elif isinstance(m1,sp.csc_matrix) and isinstance(m2,sp.csc_matrix):
        mode='C'
        rcs1=np.divide(rcs,m2.shape[1])
        rcs2=np.mod(rcs,m2.shape[1])
        cls=sp.csc_matrix
    else:
        raise ValueError('_fkron_ error: m1 and m2 should be both instances of csr_matrix or csc_matrix.')
    nnz=(m1.indptr[rcs1+1]-m1.indptr[rcs1]).dot(m2.indptr[rcs2+1]-m2.indptr[rcs2])
    if nnz>0:
        if m1.dtype==np.float32:
            data,indices,indptr,shape=fkron_r4(mode,m1.data,m1.indices,m1.indptr,m1.shape,rcs1,m2.data,m2.indices,m2.indptr,m2.shape,rcs2,nnz)
        elif m1.dtype==np.float64:
            data,indices,indptr,shape=fkron_r8(mode,m1.data,m1.indices,m1.indptr,m1.shape,rcs1,m2.data,m2.indices,m2.indptr,m2.shape,rcs2,nnz)
        elif m1.dtype==np.complex64:
            data,indices,indptr,shape=fkron_c4(mode,m1.data,m1.indices,m1.indptr,m1.shape,rcs1,m2.data,m2.indices,m2.indptr,m2.shape,rcs2,nnz)
        elif m1.dtype==np.complex128:
            data,indices,indptr,shape=fkron_c8(mode,m1.data,m1.indices,m1.indptr,m1.shape,rcs1,m2.data,m2.indices,m2.indptr,m2.shape,rcs2,nnz)
        else:
            raise ValueError("_fkron_ error: only matrices with dtype being float32, float64, complex64 or complex128 are supported.")
        result=cls((data,indices,indptr),shape=shape)
    else:
        if mode=='R':
            result=sp.csr_matrix((len(rcs),m1.shape[1]*m2.shape[1]),dtype=m1.dtype)
        else:
            result=sp.csc_matrix((m1.shape[0]*m2.shape[0],len(rcs)),dtype=m1.dtype)
    return result

def kronsum(m1,m2,rows=None,cols=None,format='csr'):
    '''
    Kronecker sum of two matrices.
    Parameters:
        m1,m2: 2d ndarray
            The matrices.
        rows,cols: list of sorted integer, optional
            The wanted rows and cols.
        format: string, optional
            The format of the product.
    Returns: sparse matrix whose format is specified by the parameter format
        The Kronecker sum.
    '''
    #return kron(m2,sp.identity(m1.shape[0],dtype=m1.dtype),rows,cols,format)+kron(sp.identity(m2.shape[0],dtype=m2.dtype),m1,rows,cols,format)
    if rows is not None and cols is not None:
        if len(rows)>=len(cols):
            result=(_fkron_identity_('R',sp.csc_matrix(m2),cols,m1.shape[0]).asformat('csr')[rows,:].asformat(format)+
                    _fkron_identity_('L',sp.csc_matrix(m1),cols,m2.shape[0]).asformat('csr')[rows,:].asformat(format))
        else:
            result=(_fkron_identity_('R',sp.csr_matrix(m2),rows,m1.shape[0]).asformat('csc')[:,cols].asformat(format)+
                    _fkron_identity_('L',sp.csr_matrix(m1),rows,m2.shape[0]).asformat('csc')[:,cols].asformat(format))
    elif rows is not None:
        result=(_fkron_identity_('R',sp.csr_matrix(m2),rows,m1.shape[0]).asformat(format)+
                _fkron_identity_('L',sp.csr_matrix(m1),rows,m2.shape[0]).asformat(format))
    elif cols is not None:
        result=(_fkron_identity_('R',sp.csc_matrix(m2),cols,m1.shape[0]).asformat(format)+
                _fkron_identity_('L',sp.csc_matrix(m1),cols,m2.shape[0]).asformat(format))
    else:
        result=sp.kronsum(m1,m2,format=format)
    return result

def _fkron_identity_(mode,m,rcs,idn):
    '''
    Python wrapper for fkron_identity_r4, fkron_identity_r8, fkron_identity_c4 and fkron_identity_c8.
    Parameters:
        mode: 'L' or 'R'
            'L' for 'kron(identity,m)' and 'R' for 'kron(m,identity)'
        m: csr_matrix/csc_matrix
            The sparse matrix.
        rcs: list of integer
            The wanted rows/columns.
        idn: integer
            The dimension of the identity.
    '''
    if mode[0:1]=='L':
        if isinstance(m,sp.csr_matrix):
            mode+='R'
            ircs=np.divide(rcs,m.shape[0])
            mrcs=np.mod(rcs,m.shape[0])
            cls=sp.csr_matrix
        elif isinstance(m,sp.csc_matrix):
            mode+='C'
            ircs=np.divide(rcs,m.shape[1])
            mrcs=np.mod(rcs,m.shape[1])
            cls=sp.csc_matrix
        else:
            raise ValueError('_fkron_identity_ error: the input matrix should be csr_matrix or csc_matrix.')
    elif mode[0:1]=='R':
        if isinstance(m,sp.csr_matrix):
            mode+='R'
            cls=sp.csr_matrix
        elif isinstance(m,sp.csc_matrix):
            mode+='C'
            cls=sp.csc_matrix
        else:
            raise ValueError('_fkron_identity_ error: the input matrix should be csr_matrix or csc_matrix.')
        mrcs=np.divide(rcs,idn)
        ircs=np.mod(rcs,idn)
    else:
        raise ValueError("_fkron_identity_ error: mode(%s) should be 'L' or 'R'"%(mode[0:1]))
    nnz=(m.indptr[mrcs+1]-m.indptr[mrcs]).sum()
    if nnz>0:
        if m.dtype==np.float32:
            data,indices,indptr,shape=fkron_identity_r4(mode,m.data,m.indices,m.indptr,m.shape,mrcs,idn,ircs,nnz)
        elif m.dtype==np.float64:
            data,indices,indptr,shape=fkron_identity_r8(mode,m.data,m.indices,m.indptr,m.shape,mrcs,idn,ircs,nnz)
        elif m.dtype==np.complex64:
            data,indices,indptr,shape=fkron_identity_c4(mode,m.data,m.indices,m.indptr,m.shape,mrcs,idn,ircs,nnz)
        elif m.dtype==np.complex128:
            data,indices,indptr,shape=fkron_identity_c8(mode,m.data,m.indices,m.indptr,m.shape,mrcs,idn,ircs,nnz)
        else:
            raise ValueError("_fkron_identity_ error: only matrix with dtype being float32, float64, complex64 or complex128 is supported.")
        result=cls((data,indices,indptr),shape=shape)
    else:
        if mode[1:2]=='R':
            result=sp.csr_matrix((len(rcs),m.shape[1]*idn),dtype=m.dtype)
        else:
            result=sp.csc_matrix((m.shape[0]*idn,len(rcs)),dtype=m.dtype)
    return result
