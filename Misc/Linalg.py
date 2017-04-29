'''
---------------
Linear algebras
---------------

Linear algebras as a supplement to `numpy.linalg` and `scipy.linalg`, including
    * constants: TOL
    * functions: kron, overlap, reorder, dagger, truncated_svd, eigsh, block_diag, solve, deparallelization
'''

__all__=['TOL','kron','overlap','reorder','dagger','truncated_svd','eigsh','block_diag','solve','deparallelization']

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as pl
import scipy.linalg as sl
from fkron import *

TOL=5*10**-12

def kron(m1,m2,timers=None,**karg):
    '''
    Kronecker product of two matrices.

    Parameters
    ----------
    m1,m2 : 2d ndarray
        The matrices.

    Returns
    -------
    csr_matrix
        The product.
    '''
    if karg.get('rcs',None) is None:
        result=sp.kron(m1,m2,format='csr')
    else:
        assert len(karg)==4 and m1.dtype==m2.dtype and m1.shape[0]==m1.shape[1] and m2.shape[0]==m2.shape[1]
        rcs,rcs1,rcs2,slices=karg['rcs'],karg['rcs1'],karg['rcs2'],karg['slices']
        with timers.get('csr'):
            m1=sp.csr_matrix(m1)
            m2=sp.csr_matrix(m2)
        with timers.get('fkron'):
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

def overlap(*args):
    '''
    Calculate the overlap between two vectors or among a matrix and two vectors.
    Usage:
        * ``overlap(vector1,vector2)``, with 
            vector1,vector2: 1d ndarray
                The vectors between which the overlap is to calculate.
        * ``overlap(vector1,matrix,vector2)``, with
            vector1,vector2: 1d ndarray
                The ket and bra in the overlap.
            matrix: 2d ndarray-like
                The matrix between the two vectors.
    '''
    assert len(args) in (2,3)
    if len(args)==2:
        return np.vdot(args[0],args[1])
    else:
        return np.vdot(args[0],args[1].dot(args[2]))

def reorder(array,axes=None,permutation=None):
    '''
    Reorder the axes of an array from the ordinary numpy.kron order to the correct quantum number collection order.

    Parameters
    ----------
    array : ndarray-like
        The original array in the ordinary numpy.kron order.
    axes : list of integer, optional
        The axes of the array to be reordered.
    permutation : 1d ndarray of integers, optional
        The permutation array applied to the required axes.

    Returns
    -------
    ndarray-like
        The axes-reordered array.
    '''
    result=array
    if permutation is not None:
        axes=xrange(array.ndim) if axes is None else axes
        for axis in axes:
            temp=[slice(None,None,None)]*array.ndim
            temp[axis]=permutation
            result=result[tuple(temp)]
    return result

def dagger(m):
    '''
    The Hermitian conjugate of a matrix.
    '''
    assert m.ndim==2
    if m.dtype in (np.int,np.int8,np.int16,np.int32,np.int64,np.float,np.float16,np.float32,np.float64,np.float128):
        return m.T
    else:
        return m.T.conjugate()

def truncated_svd(m,nmax=None,tol=None,return_truncation_err=False,**karg):
    '''
    Perform the truncated svd.

    Parameters
    ----------
    m : 2d ndarray
        The matrix to be truncated_svded.
    nmax : integer, optional
        The maximum number of singular values to be kept. If it is None, it takes no effect.
    tol : float64, optional
        The truncation tolerance. If it is None, it takes no effect.
    return_truncation_err : logical, optional
        If it is True, the truncation err will be returned.
    karg : dict
        Please see http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html for details.

    Returns
    -------
    u,s,v : ndarray
        The truncated result.
    err : float64, optional
        The truncation error.
    '''
    u,s,v=sl.svd(m,**karg)
    nmax=len(s) if nmax is None else min(nmax,len(s))
    tol=s[nmax-1] if tol is None else max(s[nmax-1],tol)
    indices=(s>=tol)
    if return_truncation_err:
        u,s,v,err=u[:,indices],s[indices],v[indices,:],(s[~indices]**2).sum()
        return u,s,v,err
    else:
        u,s,v=u[:,indices],s[indices],v[indices,:]
        return u,s,v

def eigsh(A,max_try=6,**karg):
    '''
    Find the eigenvalues and eigenvectors of the real symmetric square matrix or complex hermitian matrix A.
    This is a wrapper for scipy.sparse.linalg.eigsh to handle the exceptions it raises.

    Parameters
    ----------
    A : An NxN matrix, array, sparse matrix, or LinearOperator
        The matrix whose eigenvalues and eigenvectors is to be computed.
    max_try : integer, optional
        The maximum number of tries to do the computation when the computed eigenvalues/eigenvectors do not converge.
    karg : dict
        Please refer to https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.linalg.eigsh.html for details.
    '''
    if A.shape==(1,1):
        assert 'M' not in karg
        assert 'sigma' not in karg
        es=np.array([A.todense()[0,0]])
        vs=np.ones((1,1),dtype=A.dtype)
    else:
        num=1
        while True:
            try:
                es,vs=pl.eigsh(A,**karg)
                break
            except pl.ArpackNoConvergence as err:
                if num<max_try:
                    num+=1
                else:
                    raise err
    return es,vs

def block_diag(*ms):
    '''
    Create a block diagonal matrix from provided ones.

    Parameters
    ----------
    ms : list of 2d ndarray
        The input matrices.

    Returns
    -------
    2d ndarray
        The constructed block diagonal matrix.
    '''
    if len(ms)==0: ms=[np.zeros((0,0))]
    shapes=np.array([a.shape for a in ms])
    dtype=np.find_common_type([m.dtype for m in ms],[])
    result=np.zeros(np.sum(shapes,axis=0),dtype=dtype)
    r,c=0,0
    for i,(cr,cc) in enumerate(shapes):
        result[r:r+cr,c:c+cc]=ms[i]
        r+=cr
        c+=cc
    return result

def solve(A,b,rtol=10**-8):
    '''
    Solve the matrix equation A*x=b by QR decomposition.

    Parameters
    ----------
    A : 2d ndarray
        The coefficient matrix.
    b : 1d ndarray
        The ordinate values.
    rtol : np.float64
        The relative tolerance of the solution.

    Returns
    -------
    1d ndarray
        The solution.

    Raises
    ------
    LinAlgError
        When no solution exists.
    '''
    assert A.ndim==2
    nrow,ncol=A.shape
    if nrow>=ncol:
        result=np.zeros(ncol,dtype=np.find_common_type([],[A.dtype,b.dtype]))
        q,r=sl.qr(A,mode='economic',check_finite=False)
        temp=q.T.dot(b)
        for i,ri in enumerate(r[::-1]):
            result[-1-i]=(temp[-1-i]-ri[ncol-i:].dot(result[ncol-i:]))/ri[-1-i]
    else:
        temp=np.zeros(nrow,dtype=np.find_common_type([],[A.dtype,b.dtype]))
        q,r=sl.qr(dagger(A),mode='economic',check_finite=False)
        for i,ri in enumerate(dagger(r)):
            temp[i]=(b[i]-ri[:i].dot(temp[:i]))/ri[i]
        result=q.dot(temp)
    if not np.allclose(A.dot(result),b,rtol=rtol):
        raise sl.LinAlgError('solve error: no solution.')
    return result

def deparallelization(m,mode='R',zero=10**-8,tol=10**-6,return_indices=False):
    '''
    Deparallelize the rows or columns of a matrix.

    Parameters
    ----------
    m : 2d ndarray
        The matrix to be deparallelized.
    mode : 'R' or 'C', optional
        'R' for deparallelization of rows and 'C' for decomposition of columns.
    zero : np.float64, optional
        The absolute value to identity zero vectors.
    tol : np.float64, optional
        The relative tolerance for rows or columns that can be considered as paralleled.
    return_truncation_err : logical, optional
        When True, the indices of the kept rows or columns will be returned.
        Otherwise not.

    Returns
    -------
    M : 2d ndarray
        The deparallelized rows or columns.
    T : 2d ndarray
        The coefficient matrix that satisfies T*M==m('R') or M*T==m('C').
    indices : 1d ndarray of integers, optional
        The indices of the kept rows or columns.
    '''
    assert mode in ('R','C') and m.ndim==2
    M,data,rows,cols,indices=[],[],[],[],[]
    if mode=='R':
        for i,row in enumerate(m):
            inds=np.argwhere(np.abs(row)>zero)
            if len(inds)==0:
                data.append(0)
                rows.append(i)
                cols.append(0)
            else:
                for j,krow in enumerate(M):
                    factor=(krow[inds[0]]/row[inds[0]])[0]
                    if np.allclose(row*factor,krow,rtol=tol):
                        data.append(1.0/factor)
                        rows.append(i)
                        cols.append(j)
                        break
                else:
                    data.append(1.0)
                    rows.append(i)
                    cols.append(len(M))
                    M.append(row)
                    indices.append(i)
        M=np.asarray(M)
        T=sp.coo_matrix((data,(rows,cols)),shape=(m.shape[0],M.shape[0])).toarray()
        if return_indices:
            return T,M,indices
        else:
            return T,M
    else:
        for i,col in enumerate(m.T):
            inds=np.argwhere(np.abs(col)>zero)
            if len(inds)==0:
                data.append(0)
                rows.append(0)
                cols.append(i)
            else:
                for j,kcol in enumerate(M):
                    factor=(kcol[inds[0]]/col[inds[0]])[0]
                    if np.allclose(col*factor,kcol,rtol=tol):
                        data.append(1.0/factor)
                        rows.append(j)
                        cols.append(i)
                        break
                else:
                    data.append(1.0)
                    rows.append(len(M))
                    cols.append(i)
                    M.append(col)
                    indices.append(i)
        M=np.asarray(M).T
        T=sp.coo_matrix((data,(rows,cols)),shape=(M.shape[1],m.shape[1])).toarray()
        if return_indices:
            return M,T,indices
        else:
            return M,T
