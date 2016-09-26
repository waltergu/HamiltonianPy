'''
Tensors with quantum numbers, including
1) functions: kron, block_svd
2) classes: QNTensor
'''

import numpy as np
import scipy.linalg as sl
import scipy.sparse as sp
from ..Basics import QuantumNumber,QuantumNumberCollection
from Tensor import *

__all__=['kron','block_svd']

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

class QNTensor(Tensor):
    '''
    '''
    
    def __init__(self,):
        pass
