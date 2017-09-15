'''
---------------
Linear algebras
---------------

Linear algebras as a supplement to `numpy.linalg`, `scipy.linalg` and 'scipy.sparse.linalg', including
    * constants: TOL
    * classes: Lanczos, LinearOperator
    * functions: kron, overlap, reorder, dagger, truncated_svd, eigsh, block_diag, solve, deparallelization
'''

__all__=['TOL','Lanczos','LinearOperator','kron','overlap','reorder','dagger','truncated_svd','eigsh','block_diag','solve','deparallelization']

import numpy as np
import numpy.linalg as nl
import scipy.sparse as sp
import scipy.sparse.linalg as pl
import scipy.linalg as sl
from copy import copy
from fkron import *

TOL=5*10**-12

class Lanczos(object):
    '''
    The Lanczos algorithm to deal with csr-formed sparse Hermitian matrices.

    Attributes
    ----------
    matrix : csr_matrix
        The csr-formed sparse Hermitian matrix.
    zero : float
        The precision used to cut off the Lanczos iterations.
    new,old : 1d ndarray
        The new and old vectors updated in the Lanczos iterations.
    a,b : 1d list of floats
        The coefficients calculated in the Lanczos iterations.
    cut : logical
        A flag to tag whether the iteration has been cut off.
    '''

    def __init__(self,matrix,v0=None,check_normalization=True,vtype='rd',zero=10**-10,dtype=np.complex128):
        '''
        Constructor.

        Parameters
        ----------
        matrix : csr_matrix
            The csr-formed sparse Hermitian matrix.
        v0 : 1d ndarray, optional
            The initial vector to begin with the Lanczos iterations. It must be normalized already.
        check_normalization : logical, optional
            When it is True, the input v0 will be check to see whether it is normalized.
        vtype : string, optional
            A flag to tell what type of initial vectors to use when the parameter vector is None.
            'rd' means a random vector while 'sy' means a symmetric vector.
        zero : float, optional
            The precision used to cut off the Lanczos iterations.
        dtype : dtype, optional
            The data type of the iterated vectors.
        '''
        self.matrix=matrix
        self.zero=zero
        if v0 is None:
            if vtype.lower()=='rd':
                self.new=np.zeros(matrix.shape[0],dtype=dtype)
                self.new[:]=np.random.rand(matrix.shape[0])
            else:
                self.new=np.ones(matrix.shape[0],dtype=dtype)
            self.new[:]=self.new[:]/nl.norm(self.new)
        else:
            if check_normalization:
                temp=nl.norm(v0)
                if abs(temp-v0)>zero:
                    raise ValueError('Lanczos constructor error: v0(norm=%s) is not normalized.'%temp)
            self.new=v0
        self.old=copy(self.new)
        self.cut=False
        self.a=[]
        self.b=[]

    def iter(self):
        '''
        The Lanczos iteration.
        '''
        count=len(self.a)
        buff=self.matrix.dot(self.new)
        self.a.append(np.vdot(self.new,buff))
        if count>0:
            buff[:]=buff[:]-self.a[count]*self.new-self.b[count-1]*self.old
        else:
            buff[:]=buff[:]-self.a[count]*self.new
        nbuff=nl.norm(buff)
        if nbuff>self.zero:
            self.b.append(nbuff)
            self.old[:]=self.new[:]
            self.new[:]=buff[:]/nbuff
        else:
            self.cut=True
            self.b.append(0.0)
            self.old[:]=self.new[:]
            self.new[:]=0.0

    def tridiagnoal(self):
        '''
        This method returns the tridiagnoal matrix representation of the original sparse Hermitian matrix.

        Returns
        -------
        2d ndarray
            The tridiagnoal matrix representation of the original sparse Hermitian matrix.
        '''
        nmatrix=len(self.a)
        result=np.zeros((nmatrix,nmatrix))
        for i,(a,b) in enumerate(zip(self.a,self.b)):
            result[i,i]=a.real
            if i<nmatrix-1: 
                result[i+1,i]=b
                result[i,i+1]=b
        return result

    def eig(self,job='n',precision=10**-10):
        '''
        This method returns the ground state energy and optionally the ground state of the original sparse Hermitian matrix.

        Parameters
        ----------
        job : string
            A flag to tag what jobs the method does.
            'n' means ground state energy only and 'v' means ground state energy and ground state both.
        precision : float
            The precision of the calculated ground state energy which is used to terminate the Lanczos iteration.

        Returns
        -------
        gse : float
            the ground state energy.
        gs : 1d ndarray, optional
            The ground state. Present when the parameter job is set to be 'V' or 'v'.
        '''
        if job in ('V','v'):gs=copy(self.new)
        delta=1.0;buff=np.inf;gse=None;v=[]
        while not self.cut and delta>precision:
            self.iter()
            if job in ('V','v'):
                w,vs=sl.eigh(self.tridiagnoal())
                gse=w[0];v=vs[:,0]
            else:
                gse=sl.eigh(self.tridiagnoal(),eigvals_only=True)[0]
            delta=abs(gse-buff)
            buff=gse
        if job in ('V','v'):
            self.a=[];self.b=[]
            for i in xrange(len(v)):
                if i==0:
                    self.new[:]=gs[:]
                    gs[:]=0.0
                gs[:]+=self.new*v[i]
                self.iter()
            return gse,gs
        else:
            return gse

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

def eigsh(A,max_try=6,return_eigenvectors=True,**karg):
    '''
    Find the eigenvalues and eigenvectors of the real symmetric square matrix or complex hermitian matrix A.
    This is a wrapper for scipy.sparse.linalg.eigsh to handle the exceptions it raises.

    Parameters
    ----------
    A : An NxN matrix, array, sparse matrix, or LinearOperator
        The matrix whose eigenvalues and eigenvectors is to be computed.
    max_try : integer, optional
        The maximum number of tries to do the computation when the computed eigenvalues/eigenvectors do not converge.
    return_eigenvectors : logical, optional
        True for returning the eigenvectors and False for not.
    karg : dict
        Please refer to https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.linalg.eigsh.html for details.
    '''
    if A.shape==(1,1):
        assert 'M' not in karg and 'sigma' not in karg and karg.get('k',1)==1
        if return_eigenvectors:
            return A.dot(np.ones(1)).reshape(-1),np.ones((1,1),dtype=A.dtype)
        else:
            return A.dot(np.ones(1)).reshape(-1)
    else:
        ntry=1
        while True:
            try:
                result=pl.eigsh(A,return_eigenvectors=return_eigenvectors,**karg)
                break
            except pl.ArpackNoConvergence as err:
                if ntry<max_try:
                    ntry+=1
                else:
                    raise err
        return result

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
    return_indices : logical, optional
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
