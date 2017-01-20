'''
Miscellaneous constants, classes or functions, including:
1) constants: TOL
2) functions: dagger,truncated_svd,eigsh,block_diag
'''

__all__=['TOL','dagger','truncated_svd','eigsh','block_diag']

import numpy as np
import scipy.sparse.linalg as pl
import scipy.linalg as sl

TOL=5*10**-12

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
    Parameters:
        m: 2d ndarray
            The matrix to be truncated_svded.
        nmax: integer, optional
            The maximum number of singular values to be kept. 
            If it is None, it takes no effect.
        tol: float64, optional
            The truncation tolerance.
            If it is None, it takes no effect.
        return_truncation_err: logical, optional
            If it is True, the truncation err will be returned.
        For other parameters, please see http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html for details.
    Returns:
        u,s,v: ndarray
            The truncated result.
        err: float64, optional
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
    Parameters:
        A: An NxN matrix, array, sparse matrix, or LinearOperator
            The matrix whose eigenvalues and eigenvectors is to be computed.
        max_try: integer, optional
            The maximum number of tries to do the computation when the computed eigenvalues/eigenvectors do not converge.
        karg: dict
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
    Parameters:
        ms: list of 2d ndarray
            The input matrices.
    Returns: 2d ndarray
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
