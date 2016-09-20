'''
Linear algebra, including
1) functions: truncated_svd, kron, block_svd;
2) classes: Lanczos
'''

__all__=['truncated_svd','kron','block_svd','Lanczos']

import numpy as np
import numpy.linalg as nl
import scipy.linalg as sl
import scipy.sparse as sp
from ..Basics import QuantumNumber,QuantumNumberCollection
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

def kron(m1,m2,qns1=None,qns2=None,qns=None,target=None,separate_return=False,format='csr'):
    '''
    Kronecker product of two matrices.
    Parameters:
        m1,m2: 2d ndarray-like
            The matrices.
        qns1,qns2: QuantumNumberCollection, optional
            The corresponding quantum number collections of the two matrices.
        qns: QuantumNumberCollection, optional
            The corresponding quantum number collection of the product.
        target: QuantumNumber/list of QuantumNumber
            The target subspace of the product.
        separate_return: logical, optional
            It takes on effect only when target is not None.
            When it is True, the different block of the target subspaces will be returned separately as a list of sparse matrices.
            When it is False, the different block of the target subspaces will be returned as a single sparse matrix but block diagonally.
        format: string, optional
            The format of the product.
    Returns: sparse matrix whose format is specified by the parameter format
        The product.
    '''
    if qns1 is None and qns2 is None:
        result=sp.kron(m1,m2,format=format)
        if format=='csr':result.eliminate_zeros()
        return result
    elif qns1 is not None and qns2 is not None and qns is not None:
        if m1.shape!=(qns1.n,qns1.n) or m2.shape!=(qns2.n,qns2.n) or qns.n!=qns1.n*qns2.n:
            raise ValueError("kron error: the matrices and the quantum number collections don't match.")
        P=sp.coo_matrix((np.ones(qns.n),(range(qns.n),qns.permutation)),shape=(qns.n,qns.n))
        result=P.dot(sp.kron(m1,m2,format=format).dot(P.T))
        if format=='csr':result.eliminate_zeros()
        if target is not None:
            if isinstance(target,QuantumNumber):
                return result[qns[target],qns[target]]
            else:
                result=[result[qns[qn],qns[qn]] for qn in target]
                if separate_return:
                    return result
                else:
                    return sp.block_diag(result,format=format)
    else:
        raise ValueError('kron error: all of or none of qns1, qns2 and qns should be None.')

def block_svd(Psi,qns1,qns2,qns=None,n=None,return_truncation_error=True):
    '''
    Block svd of the wavefunction Psi according to the bipartition information passed by qns1 and qns2.
    Parameters:
        Psi: 1D ndarray
            The wavefunction to be block svded.
        qns1,qns2: integer or QuantumNumberCollection
            When integers, they are the number of the basis of the two parts of the bipartition.
            When QuantumNumberCollection, they are the quantum number collections of the two parts of the bipartition.
        qns: QuantumNumberCollection, optional
            It takes on effect only when qns1 and quns2 are QuantumNumberCollection.
            The quantum number collection of the wavefunction.
        n: integer, optional
            The maximum number of largest singular values to be kept.
        return_truncation_error: logical, optional
            It only takes on effect when n is not None.
            When it is True, the trancation error is also returned.
    Returns:
        U,S,V: ndarray
            The svd decomposition of Psi
        QNS1,QNS2: QuantumNumberCollection, optional
            The new QuantumNumberCollection after the SVD.
            Only when qns1, qns2 and qns are QuantumNumberCollection are they returned.
        err: float64, optional
            The truncation error.
            Only when n is not None and return_truncation_error is True is it returned.
    '''
    if isinstance(qns1,QuantumNumberCollection) and isinstance(qns2,QuantumNumberCollection) and isinstance(qns,QuantumNumberCollection):
        pairs,Us,Ss,Vs=[],[],[],[]
        for qn in qns:
            count=0
            for qn1,qn2 in qns.map[qn]:
                pairs.append((qn1,qn2))
                s1,s2=qns1[qn1],qns2[qn2]
                n1,n2=s1.stop-s1.start,s2.stop-s2.start
                u,s,v=sl.svd(Psi[count:count+n1*n2].reshape((n1,n2)),full_matrices=False)
                Us.append(u)
                Ss.append(s)
                Vs.append(v)
                count+=n1*n2
        if n is None:
            return sl.block_diag(Us),np.concatenate(Ss),sl.block_diag(Vs),qns1,qns2
        else:
            temp=np.sort(np.concatenate([-s for s in Ss]))
            n=min(n,len(temp))
            U,S,V,para1,para2=[],[],[],[],[]
            for u,s,v,(qn1,qn2) in zip(Us,sS,Vs,pairs):
                cut=np.searchsorted(-s,temp[n-1],side='right')
                U.append(u[:,0:cut])
                S.append(s[0:cut])
                V.append(v[0:cut,:])
                para1.append((qn1,n))
                para2.append((qn2,n))
            U,S,V,QNS1,QNS2=sl.block_diag(U),np.concatenate(S),sl.block_diag(V),QuantumNumberCollection(para1),QuantumNumberCollection(para2)
            if return_truncation_error:
                return U,S,V,QNS1,QNS2,-temp[n:].sum()
            else:
                return U,S,V,QNS1,QNS2
    elif (isinstance(qns1,int) or isinstance(qns1,long)) and (isinstance(qns2,int) or isinstance(qns2,long)):
        u,s,v=sl.svd(Psi.reshape((qns1,qns2)),full_matrices=False)
        if n is None:
            return u,s,v
        else:
            n=min(n,len(s))
            if return_truncation_error:
                return u[:,0:n],s[0:n],v[0:n,:],s[n:].sum()
            else:
                return u[:,0:n],s[0:n],v[0:n,:]
    else:
        raise ValueError("block_svd error: the type of qns1(%s), qns2(%s) and qns(%s) do not match."%(qns1.__class__.__name__,qns2.__class__.__name__,qns.__class__.__name__))

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
