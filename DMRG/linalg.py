'''
Linear algebra, including:
1) functions: kron, kronsum, block_svd
'''

import numpy as np
import scipy.linalg as sl
import scipy.sparse as sp
from ..Math.linalg import truncated_svd
from ..Basics import QuantumNumber,QuantumNumberCollection

__all__=['kron','kronsum','block_svd']

def kron(m1,m2,qnc=None,target=None,format='csr'):
    '''
    Kronecker product of two matrices.
    Parameters:
        m1,m2: 2d ndarray-like
            The matrices.
        qnc: QuantumNumberCollection, optional
            The corresponding quantum number collection of the product.
        target: QuantumNumber
            The target subspace of the product.
        format: string, optional
            The format of the product.
    Returns: sparse matrix whose format is specified by the parameter format
        The product.
    '''
    if isinstance(qnc,QuantumNumberCollection):
        result=qnc.reorder(sp.kron(m1,m2,format=format))
        if target is not None:
            assert isinstance(target,QuantumNumber)
            result=result[qnc[target],qnc[target]]
    else:
        result=sp.kron(m1,m2,format=format)
    #if format in ('csr','csc'):result.eliminate_zeros()
    return result

def kronsum(m1,m2,qnc=None,target=None,format='csr'):
    '''
    Kronecker sum of two matrices.
    Please see scipy.sparse.kronsum for details.
    Parameters:
        m1,m2: 2d ndarray-like
            The matrices.
        qnc: QuantumNumberCollection, optional
            The corresponding quantum number collection of the product.
        target: QuantumNumber
            The target subspace of the product.
        format: string, optional
            The format of the product.
    Returns: sparse matrix whose format is specified by the parameter format
        The Kronecker sum.
    '''
    if isinstance(qnc,QuantumNumberCollection):
        result=qnc.reorder(sp.kronsum(m1,m2,format=format))
        if target is not None:
            assert isinstance(target,QuantumNumber)
            result=result[qnc[target],qnc[target]]
    else:
        result=sp.kronsum(m1,m2,format=format)
    if format in ('csr','csc'):result.eliminate_zeros()
    return result

def block_svd(Psi,qnc1,qnc2,qnc=None,target=None,nmax=None,print_truncation_err=True):
    '''
    Block svd of the wavefunction Psi according to the bipartition information passed by qnc1 and qnc2.
    Parameters:
        Psi: 1D ndarray
            The wavefunction to be block svded.
        qnc1,qnc2: integer or QuantumNumberCollection
            1) integers
                The number of the basis of the two parts of the bipartition.
            2) QuantumNumberCollection
                The quantum number collections of the two parts of the bipartition.
        qnc: QuantumNumberCollection, optional
            The quantum number collection of the wavefunction.
            It takes effect only when qnc1 and qnc2 are QuantumNumberCollection.
        target: QuantumNumber, optional
            The target subspace of the product.
        nmax,print_truncation_err: optional
            For details, please refer to HamiltonianPy.Math.linalg.truncated_svd
    Returns:
        U,S,V: ndarray
            The svd decomposition of Psi
        QNC1,QNC2: integer or QuantumNumberCollection
            Their types coincide with those of qnc1 and qnc2.
            1) integers
                The number of the new singular values after the SVD
            2) QuantumNumberCollection
                The new QuantumNumberCollection after the SVD.
    '''
    if isinstance(qnc1,QuantumNumberCollection) and isinstance(qnc2,QuantumNumberCollection) and isinstance(qnc,QuantumNumberCollection):
        pairs,Us,Ss,Vs=[],[],[],[]
        count=0
        for qn1,qn2 in qnc.pairs(target):
            pairs.append((qn1,qn2))
            s1,s2=qnc1[qn1],qnc2[qn2]
            n1,n2=s1.stop-s1.start,s2.stop-s2.start
            u,s,v=sl.svd(Psi[count:count+n1*n2].reshape((n1,n2)),full_matrices=False)
            Us.append(u)
            Ss.append(s)
            Vs.append(v)
            count+=n1*n2
        if nmax is None:
            return sl.block_diag(*Us),np.concatenate(Ss),sl.block_diag(*Vs),qnc1,qnc2
        else:
            temp=np.sort(np.concatenate([-s for s in Ss]))
            nmax=min(nmax,len(temp))
            U,S,V,para1,para2=[],[],[],[],[]
            for u,s,v,(qn1,qn2) in zip(Us,Ss,Vs,pairs):
                cut=np.searchsorted(-s,temp[nmax-1],side='right')
                U.append(u[:,0:cut])
                S.append(s[0:cut])
                V.append(v[0:cut,:])
                para1.append((qn1,cut))
                para2.append((qn2,cut))
            if print_truncation_err and nmax<len(temp):
                print "Tensor svd truncation err: %s."%((temp[nmax:]**2).sum())
            #print 'Us:',Us
            #print 'Ss:',Ss
            #print 'Vs:',Vs
            #print 'U:\n','\n'.join(str(u) for u in U)
            #print 'S:\n','\n'.join(str(s) for s in S)
            #print 'V:\n','\n'.join(str(v) for v in V)
            #print 'qnc1:',QuantumNumberCollection(para1)
            #print 'qnc2:',QuantumNumberCollection(para2)
            return sl.block_diag(*U),np.concatenate(S),sl.block_diag(*V),QuantumNumberCollection(para1),QuantumNumberCollection(para2)
    elif (isinstance(qnc1,int) or isinstance(qnc1,long)) and (isinstance(qnc2,int) or isinstance(qnc2,long)):
        u,s,v=truncated_svd(Psi.reshape((qnc1,qnc2)),full_matrices=False,nmax=nmax,print_truncation_err=print_truncation_err)
        return u,s,v,len(s),len(s)
    else:
        n1,n2,n=qnc1.__class__.__name__,qnc2.__class__.__name__,qnc.__class__.__name__
        raise ValueError("block_svd error: the type of qnc1(%s), qnc2(%s) and qnc(%s) do not match."%(n1,n2,n))
