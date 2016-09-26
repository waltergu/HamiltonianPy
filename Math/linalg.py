'''
Linear algebra, including
1) functions: truncated_svd
2) classes: Lanczos
'''

__all__=['truncated_svd','Lanczos']

import numpy as np
import numpy.linalg as nl
import scipy.linalg as sl
from copy import copy

def truncated_svd(m,nmax=None,tol=None,print_truncation_err=False,**karg):
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
            If it is None, it taks no effect.
        print_truncation_err: logical, optional
            If it is True, the truncation err will be printed.
        For other parameters, please see http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html for details.
    Returns:
        u,s,v: ndarray
            The truncated result.
    '''
    u,s,v=sl.svd(m,**karg)
    nmax=len(s) if nmax is None else min(nmax,len(s))
    tol=s[nmax-1] if tol is None else tol
    indices=(s>=tol)
    if print_truncation_err and nmax<len(s): print 'Tensor svd truncation err: %s'%s[~indices].sum()
    return u[:,indices],s[indices],v[indices,:]

class Lanczos:
    '''
    The Lanczos algorithm to deal with csr-formed sparse Hermitian matrices.
    Attributes:
        matrix: csr_matrix
            The csr-formed sparse Hermitian matrix.
        zero: float
            The precision used to cut off the Lanczos iterations.
        new,old: 1D ndarray
            The new and old vectors updated in the Lanczos iterations.
        a,b: 1D list of floats
            The coefficients calculated in the Lanczos iterations.
        cut: logical
            A flag to tag whether the iteration has been cut off.
    '''
    def __init__(self,matrix,v0=None,check_normalization=True,vtype='rd',zero=10**-10,dtype=np.complex128):
        '''
        Constructor.
        Parameters:
            matrix: csr_matrix
                The csr-formed sparse Hermitian matrix.
            v0: 1D ndarray,optional
                The initial vector to begin with the Lanczos iterations. 
                It must be normalized already.
            check_nomalization: logical, optional
                When it is True, the input v0 will be check to see whether it is normalized.
            vtype: string,optional
                A flag to tell what type of initial vectors to use when the parameter vector is None.
                'rd' means a random vector while 'sy' means a symmetric vector.
            zero: float,optional
                The precision used to cut off the Lanczos iterations.
            dtype: dtype,optional
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
        Returns:
            result: 2D ndarray
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
        Parameters:
            job: string
                A flag to tag what jobs the method does.
                'n' means ground state energy only and 'v' means ground state energy and ground state both.
            precision: float
                The precision of the calculated ground state energy which is used to terminate the Lanczos iteration.
        Returns:
            gse: float
                the ground state energy.
            gs: 1D ndarray,optional
                The ground state. Present when the parameter job is set to be 'V' or 'v'.
        '''
        if job in ('V','v'):gs=copy(self.new)
        delta=1.0;buff=np.inf
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
