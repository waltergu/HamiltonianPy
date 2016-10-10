'''
Linear algebra, including:
1) functions: kron, block_svd
'''

import numpy as np
import scipy.linalg as sl
import scipy.sparse as sp
from ..Math.linalg import truncated_svd
from ..Basics import QuantumNumber,QuantumNumberCollection

__all__=['kron','block_svd']

def kron(m1,m2,qns1=None,qns2=None,qns=None,target=None,format='csr'):
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
        format: string, optional
            The format of the product.
    Returns: sparse matrix whose format is specified by the parameter format
        The product.
    '''
    if qns1 is None and qns2 is None:
        result=sp.kron(m1,m2,format=format)
        if format=='csr':result.eliminate_zeros()
    elif qns1 is not None and qns2 is not None and qns is not None:
        assert m1.shape==(qns1.n,qns1.n) and m2.shape==(qns2.n,qns2.n) and qns.n==qns1.n*qns2.n
        #P=sp.coo_matrix((np.ones(qns.n),(range(qns.n),qns.permutation)),shape=(qns.n,qns.n))
        #result=P.dot(sp.kron(m1,m2,format=format).dot(P.T))
        result=sp.kron(m1,m2,format=format)[qns.permutation,:][:,qns.permutation]
        if format=='csr':result.eliminate_zeros()
        if target is not None:
            if isinstance(target,QuantumNumber):
                result=result[qns[target],qns[target]]
            else:
                result=[result[qns[qn],qns[qn]] for qn in target]
    else:
        raise ValueError('kron error: all of or none of qns1, qns2 and qns should be None.')
    return result

def block_svd(Psi,qns1,qns2,qns=None,nmax=None,print_truncation_err=False):
    '''
    Block svd of the wavefunction Psi according to the bipartition information passed by qns1 and qns2.
    Parameters:
        Psi: 1D ndarray
            The wavefunction to be block svded.
        qns1,qns2: integer or QuantumNumberCollection
            1) integers
                The number of the basis of the two parts of the bipartition.
            2) QuantumNumberCollection
                The quantum number collections of the two parts of the bipartition.
        qns: QuantumNumberCollection, optional
            The quantum number collection of the wavefunction.
            It takes effect only when qns1 and quns2 are QuantumNumberCollection.
        nmax,print_truncation_err: optional
            For details, please refer to HamiltonianPy.Math.linalg.truncated_svd
    Returns:
        U,S,V: ndarray
            The svd decomposition of Psi
        QNS1,QNS2: QuantumNumberCollection, optional
            The new QuantumNumberCollection after the SVD.
            Only when qns1, qns2 and qns are QuantumNumberCollection are they returned.
    '''
    if isinstance(qns1,QuantumNumberCollection) and isinstance(qns2,QuantumNumberCollection) and isinstance(qns,QuantumNumberCollection):
        pairs,Us,Ss,Vs=[],[],[],[]
        for qn in qns:
            count=0
            for qn1,qn2 in qns.pairs[qn]:
                pairs.append((qn1,qn2))
                s1,s2=qns1[qn1],qns2[qn2]
                n1,n2=s1.stop-s1.start,s2.stop-s2.start
                u,s,v=sl.svd(Psi[count:count+n1*n2].reshape((n1,n2)),full_matrices=False)
                Us.append(u)
                Ss.append(s)
                Vs.append(v)
                count+=n1*n2
        if nmax is None:
            return sl.block_diag(Us),np.concatenate(Ss),sl.block_diag(Vs),qns1,qns2
        else:
            temp=np.sort(np.concatenate([-s for s in Ss]))
            nmax=min(nmax,len(temp))
            U,S,V,para1,para2=[],[],[],[],[]
            for u,s,v,(qn1,qn2) in zip(Us,Ss,Vs,pairs):
                cut=np.searchsorted(-s,temp[nmax-1],side='right')
                U.append(u[:,0:cut])
                S.append(s[0:cut])
                V.append(v[0:cut,:])
                para1.append((qn1,nmax))
                para2.append((qn2,nmax))
            if print_truncation_err:
                print "Tensor svd truncation err: %s."%(-temp[nmax:].sum())
            return sl.block_diag(U),np.concatenate(S),sl.block_diag(V),QuantumNumberCollection(para1),QuantumNumberCollection(para2)
    elif (isinstance(qns1,int) or isinstance(qns1,long)) and (isinstance(qns2,int) or isinstance(qns2,long)):
        return truncated_svd(Psi.reshape((qns1,qns2)),full_matrices=False,nmax=nmax,print_truncation_err=print_truncation_err)
    else:
        n1,n2,n=qns1.__class__.__name__,qns2.__class__.__name__,qns.__class__.__name__
        raise ValueError("block_svd error: the type of qns1(%s), qns2(%s) and qns(%s) do not match."%(n1,n2,n))
