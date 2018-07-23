'''
---------------
Linear algebras
---------------

Linear algebras as a supplement to `numpy.linalg`, `scipy.linalg` and 'scipy.sparse.linalg', including
    * constants: TOL
    * classes: Lanczos, LinearOperator
    * functions: kron, overlap, reorder, dagger, truncatedsvd, eigsh, blockdiag, solve, deparallelization
'''

__all__=['TOL','Lanczos','LinearOperator','kron','overlap','reorder','dagger','truncatedsvd','eigsh','blockdiag','solve','deparallelization']

import numpy as np
import numpy.linalg as nl
import scipy.sparse as sp
import scipy.sparse.linalg as pl
import scipy.linalg as sl
import itertools as it
from copy import copy
from fkron import *

TOL=5*10**-12

class Lanczos(object):
    '''
    Band Lanczos method.

    Attributes
    ----------
    matrix : csr_matrix
        The csr-formed sparse Hermitian matrix.
    candidates : list of 1d ndarray
        The candidate vectors of the Krylov space.
    maxiter : int
        The maximum number of Lanczos iterations.
    dtol : float
        The tolerance of the Lanczos basis.
    keepstate : logical
        True for keeping all the Lanczos basis and False for keeping only the necessary Lanczos basis in the orthogonalization process.
    nv0 : int
        The number of starting vectors.
    stop : logical
        True when the Lanczos iteration has stopped.
    niter : int
        The number of iterations that has been carried out.
    vectors : list of 1d ndarray
        The Lanczos basis.
    deflations : list of int
        The index of the deflated basis.
    _T_ : 2d ndarray
        The Lanczos representation of the input Hermitian matrix.
    P : 2d ndarray
        The projection of the Lanczos basis onto the input vectors.
    '''

    def __init__(self,matrix,v0=None,maxiter=1000,dtol=TOL,keepstate=False):
        '''
        Constructor.

        Parameters
        ----------
        matrix : csr_matrix
            The csr-formed sparse Hermitian matrix.
        v0 : list of 1d ndarray, optional
            The starting vectors of the Krylov space.
        maxiter : int, optional
            The maximum number of Lanczos iterations.
        dtol : float, optional
            The tolerance of the Lanczos basis.
        keepstate : logical, optional
            True for keeping all the Lanczos basis and False for keeping only the necessary Lanczos basis in the orthogonalization process.
        '''
        self.matrix=matrix
        self.candidates=[np.random.random(self.matrix.shape[0])] if v0 is None else list(v0)
        self.maxiter=maxiter
        self.dtol=dtol
        self.keepstate=keepstate
        self.nv0=len(self.candidates)
        self.stop=False
        self.niter=0
        self.vectors=[None]*self.maxiter
        self.deflations=[]
        self._T_=np.zeros((self.maxiter,self.maxiter),dtype=self.matrix.dtype)
        self.P=np.zeros((self.nv0,self.nv0),dtype=self.matrix.dtype)

    @property
    def nc(self):
        '''
        The number of candidate vectors of the Krylov space.
        '''
        return len(self.candidates)

    @property
    def T(self):
        '''
        The Lanczos representation of the input Hermitian matrix.
        '''
        return self._T_[0:self.niter,0:self.niter]

    def iter(self):
        '''
        The Lanczos iteration.
        '''
        while len(self.candidates)>0:
            v=self.candidates.pop(0)
            norm=nl.norm(v)
            if norm>self.dtol:
                break
            elif self.niter>=self.nc:
                self.deflations.append(self.niter-self.nc)
        else:
            self.stop=True
        if not self.stop:
            self.vectors[self.niter]=v/norm
            if self.niter-self.nc-1>=0:
                self._T_[self.niter,self.niter-self.nc-1]=norm
            else:
                self.P[self.niter,self.niter-self.nc-1+self.nv0]=norm
            for k,vc in enumerate(self.candidates):
                overlap=np.vdot(self.vectors[self.niter],vc)
                if k+self.niter>=self.nc:
                    self._T_[self.niter,k+self.niter-self.nc]=overlap
                else:
                    self.P[self.niter,self.niter-self.nc+k+self.nv0]=overlap
                vc-=overlap*self.vectors[self.niter]
            v=self.matrix.dot(self.vectors[self.niter])
            for k in xrange(max(self.niter-self.nc-1,0),self.niter):
                self._T_[k,self.niter]=np.conjugate(self._T_[self.niter,k])
                v-=self._T_[k,self.niter]*self.vectors[k]
            for k in it.chain(self.deflations,[self.niter]):
                overlap=np.vdot(self.vectors[k],v)
                if k in set(self.deflations)|{self.niter}:
                    self._T_[k,self.niter]=overlap
                    self._T_[self.niter,k]=np.conjugate(overlap)
                v-=overlap*self.vectors[k]
            self.candidates.append(v)
            if not self.keepstate and self.niter>=self.nc and self.niter-self.nc not in self.deflations: self.vectors[self.niter-self.nc]=None
            self.niter+=1

    def eigs(self):
        '''
        The eigenvalues of the Lanczos matrix.

        Returns
        -------
        1d ndarray
            The eigenvalues of the Lanczos matrix.
        '''
        return sl.eigh(self.T,eigvals_only=True)

class LinearOperator(pl.LinearOperator):
    '''
    Linear operator with a count for the matrix-vector multiplications.

    Attributes:
        shape : 2-tuple
            The shape of the linear operator.
        dtype : np.float64, np.complex128, etc
            The data type of the linear operator.
        count : integer
            The count for the matrix-vector multiplications.
        _matvec_ : callable
            The matrix-vector multiplication function.
    '''
    
    def __init__(self,shape,matvec,dtype=None):
        '''
        Constructor.

        Parameters
        ----------
        shape : 2-tuple
            The shape of the linear operator.
        matvec : callable
            The matrix-vector multiplication function.
        dtype : np.float64, np.complex128, etc, optional
            The data type of the linear operator.
        '''
        super(LinearOperator,self).__init__(dtype=dtype,shape=shape)
        self.count=0
        self._matvec_=matvec

    def _matvec(self,v):
        '''
        Matrix-vector multiplication.
        '''
        self.count+=1
        return self._matvec_(v)

def kron(m1,m2,rcs=None,timers=None):
    '''
    Kronecker product of two matrices.

    Parameters
    ----------
    m1,m2 : 2d ndarray
        The matrices.
    rcs : 1d ndarray or 3-tuple of 1d ndarray
        * When 1d ndarray
            The selected rows and columns of the kronecker product
        * When 3-tuple of 1d ndarray
            * tuple[0]: the selected rows and columns of the first matrix `m1`
            * tuple[1]: the selected rows and columns of the second matrix `m2`
            * tuple[2]: the map between the indices before and after the selection of the rows and columns of the kronecker product.
    timers : Timers, optional
        The timers to record certain procedures of this function.

    Returns
    -------
    csr_matrix
        The product.
    '''
    if rcs is None:
        result=sp.kron(m1,m2,format='csr')
    else:
        assert m1.dtype==m2.dtype and m1.shape[0]==m1.shape[1] and m2.shape[0]==m2.shape[1]
        if isinstance(rcs,np.ndarray):
            rcs1,rcs2=np.divide(rcs,m2.shape[1]),np.mod(rcs,m2.shape[1])
            slices=np.zeros(m1.shape[1]*m2.shape[1],dtype=np.int64)
            slices[rcs]=xrange(len(rcs))
        else:
            rcs1,rcs2,slices=rcs
        def csr(m1,m2):
            return sp.csr_matrix(m1),sp.csr_matrix(m2)
        def fkron(m1,m2):
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
                result=sp.csr_matrix((data,indices,indptr),shape=(len(rcs1),len(rcs1)))
            else:
                result=sp.csr_matrix((len(rcs1),len(rcs1)),dtype=m1.dtype)
            return result
        if timers is None:
            m1,m2=csr(m1,m2)
            result=fkron(m1,m2)
        else:
            with timers.get('csr'):
                m1,m2=csr(m1,m2)
            with timers.get('fkron'):
                result=fkron(m1,m2)
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
    Reorder the elements of an array along some axes accroding to the permutation.

    Parameters
    ----------
    array : ndarray
        The original array.
    axes : list of int, optional
        The axes of the array to be reordered.
    permutation : 1d ndarray of int, optional
        The permutation array applied to the required axes.

    Returns
    -------
    ndarray
        The reordered array.
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

def truncatedsvd(m,nmax=None,tol=None,returnerr=False,**karg):
    '''
    Perform the truncated svd.

    Parameters
    ----------
    m : 2d ndarray
        The matrix to be truncated svded.
    nmax : int, optional
        The maximum number of singular values to be kept. If it is None, it takes no effect.
    tol : float, optional
        The truncation tolerance. If it is None, it takes no effect.
    returnerr : logical, optional
        If it is True, the truncation err will be returned.
    karg : dict
        Please see http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html for details.

    Returns
    -------
    u,s,v : ndarray
        The truncated result.
    err : float, optional
        The truncation error.
    '''
    u,s,v=sl.svd(m,**karg)
    nmax=len(s) if nmax is None else min(nmax,len(s))
    tol=s[nmax-1] if tol is None else max(s[nmax-1],tol)
    indices=(s>=tol)
    if returnerr:
        u,s,v,err=u[:,indices],s[indices],v[indices,:],(s[~indices]**2).sum()
        return u,s,v,err
    else:
        u,s,v=u[:,indices],s[indices],v[indices,:]
        return u,s,v

def eigsh(A,maxtry=6,evon=True,**karg):
    '''
    Find the eigenvalues and eigenvectors of the real symmetric square matrix or complex hermitian matrix A.
    This is a wrapper for scipy.sparse.linalg.eigsh to handle the exceptions it raises.

    Parameters
    ----------
    A : An NxN matrix, array, sparse matrix, or LinearOperator
        The matrix whose eigenvalues and eigenvectors is to be computed.
    maxtry : integer, optional
        The maximum number of tries to do the computation when the computed eigenvalues/eigenvectors do not converge.
    evon : logical, optional
        True for returning the eigenvectors and False for not.
    karg : dict
        Please refer to https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.linalg.eigsh.html for details.
    '''
    if A.shape==(1,1):
        assert 'M' not in karg and 'sigma' not in karg and karg.get('k',1)==1
        if evon:
            return A.dot(np.ones(1)).reshape(-1).real,np.ones((1,1),dtype=A.dtype)
        else:
            return A.dot(np.ones(1)).reshape(-1).real
    else:
        ntry=1
        while True:
            try:
                result=pl.eigsh(A,return_eigenvectors=evon,**karg)
                break
            except pl.ArpackNoConvergence as err:
                if ntry<maxtry:
                    ntry+=1
                else:
                    raise err
        return result

def blockdiag(*ms):
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
    rtol : float
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

def deparallelization(m,mode='R',zero=10**-8,tol=10**-6,returnindices=False):
    '''
    Deparallelize the rows or columns of a matrix.

    Parameters
    ----------
    m : 2d ndarray
        The matrix to be deparallelized.
    mode : 'R' or 'C', optional
        'R' for deparallelization of rows and 'C' for decomposition of columns.
    zero : float, optional
        The absolute value to identity zero vectors.
    tol : float, optional
        The relative tolerance for rows or columns that can be considered as paralleled.
    returnindices : logical, optional
        When True, the indices of the kept rows or columns will be returned.
        Otherwise not.

    Returns
    -------
    M : 2d ndarray
        The deparallelized rows or columns.
    T : 2d ndarray
        The coefficient matrix that satisfies T*M==m('R') or M*T==m('C').
    indices : 1d ndarray of int, optional
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
        return (T,M,indices) if returnindices else (T,M)
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
        return (M,T,indices) if returnindices else (M,T)
