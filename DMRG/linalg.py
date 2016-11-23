'''
Linear algebra, including:
1) functions: kron, kronsum, vblock_svd, mblock_svd
'''

import numpy as np
import scipy.linalg as sl
import scipy.sparse as sp
from ..Math.linalg import truncated_svd
from ..Basics import QuantumNumber,QuantumNumberCollection
from linalg_Fortran import *
import time

__all__=['kron','kronsum','vblock_svd','mblock_svd']

def kron(m1,m2,qnc1=None,qnc2=None,qnc=None,target=None,format='csr',**karg):
    '''
    Kronecker product of two matrices.
    Parameters:
        m1,m2: 2d ndarray-like
            The matrices.
        qnc1,qnc2,qnc: QuantumNumberCollection, optional
            The corresponding quantum number collection of the m1/m2/product.
        target: QuantumNumber
            The target subspace of the product.
        format: string, optional
            The format of the product.
    Returns: sparse matrix whose format is specified by the parameter format
        The product.
    '''
    m1,m2=np.asarray(m1),np.asarray(m2)
    if isinstance(qnc,QuantumNumberCollection):
        assert isinstance(qnc1,QuantumNumberCollection)
        assert isinstance(qnc2,QuantumNumberCollection)
        assert isinstance(target,QuantumNumber)
        #logger=karg.get('logger',None)
        #if not logger.has_timer('kron'):logger.add_timer('kron')
        #if not logger.has_timer('k.reorder'):logger.add_timer('k.reorder')
        #logger.proceed('kron')
        result=sp.kron(m1,m2,format=format)
        #logger.suspend('kron')
        #logger.proceed('k.reorder')
        result=qnc.reorder(result,targets=[target])
        #logger.suspend('k.reorder')
    else:
        result=sp.kron(m1,m2,format=format)
    return result

def kronsum(m1,m2,qnc1=None,qnc2=None,qnc=None,target=None,format='csr',**karg):
    '''
    Kronecker sum of two matrices.
    Please see scipy.sparse.kronsum for details.
    Parameters:
        m1,m2: 2d ndarray-like
            The matrices.
        qnc1,qnc2,qnc: QuantumNumberCollection, optional
            The corresponding quantum number collection of the m1/m2/product.
        target: QuantumNumber
            The target subspace of the product.
        format: string, optional
            The format of the product.
    Returns: sparse matrix whose format is specified by the parameter format
        The Kronecker sum.
    '''
    m1,m2=np.asarray(m1),np.asarray(m2)
    if isinstance(qnc,QuantumNumberCollection):
        assert isinstance(qnc1,QuantumNumberCollection)
        assert isinstance(qnc2,QuantumNumberCollection)
        assert isinstance(target,QuantumNumber)
        #logger=karg.get('logger',None)
        #if not logger.has_timer('kronsum'):logger.add_timer('kronsum')
        #if not logger.has_timer('ks.reorder'):logger.add_timer('ks.reorder')
        #logger.proceed('kronsum')
        result=sp.kronsum(m1,m2,format=format)
        #logger.suspend('kronsum')
        #logger.proceed('ks.reorder')
        result=qnc.reorder(result,targets=[target])
        #logger.suspend('ks.reorder')
    else:
        result=sp.kronsum(m1,m2,format=format)
    return result

def vblock_svd(Psi,qnc1,qnc2,mode='L',nmax=None,tol=None,return_truncation_err=False):
    '''
    Block svd of a wavefunction Psi according to the bipartition information passed by qnc1 and qnc2.
    Parameters:
        Psi: 1D ndarray
            The wavefunction to be block svded.
        qnc1,qnc2: QuantumNumberCollection or integer
            1) QuantumNumberCollection
                The quantum number collections of the two parts of the bipartition.
            2) integers
                The number of the basis of the two parts of the bipartition.
        mode: 'L' or 'R'
            When 'L', the renewed qnc1 will be returned;
            When 'R', the renewed qnc2 will be returned.
        nmax,tol,return_truncation_err: optional
            For details, please refer to HamiltonianPy.Math.linalg.truncated_svd
    Returns:
        U,S,V: ndarray
            The svd decomposition of Psi
        QNC: QuantumNumberCollection or integer
            Its type coincides with those of qnc1 and qnc2.
            1) QuantumNumberCollection
                The new QuantumNumberCollection after the SVD.
            2) integer
                The number of the new singular values after the SVD.
        err: float64, optional
            The truncation error.
    '''
    if isinstance(qnc1,QuantumNumberCollection) and isinstance(qnc2,QuantumNumberCollection):
        assert mode in 'LR'
        Us,Ss,Vs=[],[],[]
        count=0
        for qn1,qn2 in zip(qnc1,qnc2):
            s1,s2=qnc1[qn1],qnc2[qn2]
            n1,n2=s1.stop-s1.start,s2.stop-s2.start
            u,s,v=sl.svd(Psi[count:count+n1*n2].reshape((n1,n2)),full_matrices=False)
            Us.append(u)
            Ss.append(s)
            Vs.append(v)
            count+=n1*n2
        temp=np.sort(np.concatenate([-s for s in Ss]))
        nmax=len(temp) if nmax is None else min(nmax,len(temp))
        tol=temp[nmax-1] if tol is None else min(-tol,temp[nmax-1])
        U,S,V,contents=[],[],[],[]
        for u,s,v,qn in zip(Us,Ss,Vs,qnc1 if mode=='L' else qnc2):
            cut=np.searchsorted(-s,tol,side='right')
            U.append(u[:,0:cut])
            S.append(s[0:cut])
            V.append(v[0:cut,:])
            contents.append((qn,cut))
        U,S,V=sl.block_diag(*U),np.concatenate(S),sl.block_diag(*V)
        QNC=QuantumNumberCollection(contents)
        if return_truncation_err:
            err=(temp[nmax:]**2).sum()
            return U,S,V,QNC,err
        else:
            return U,S,V,QNC
    elif (isinstance(qnc1,int) or isinstance(qnc1,long)) and (isinstance(qnc2,int) or isinstance(qnc2,long)):
        temp=truncated_svd(Psi.reshape((qnc1,qnc2)),full_matrices=False,nmax=nmax,tol=tol,return_truncation_err=return_truncation_err)
        U,S,V=temp[0],temp[1],temp[2]
        if return_truncation_err:
            err=temp[3]
            return U,S,V,len(S),err
        else:
            return U,S,V,len(S)
    else:
        n1,n2,n=qnc1.__class__.__name__,qnc2.__class__.__name__,qnc.__class__.__name__
        raise ValueError("block_svd error: the type of qnc1(%s), qnc2(%s) and qnc(%s) do not match."%(n1,n2,n))

def mblock_svd(M,qnc1=None,qnc2=None,mode='L',nmax=None,tol=None,return_truncation_err=False):
    '''
    Block svd of a matrix M according to the information passed by qnc1 and qnc2.
    Parameters:
        M: 2D ndarray
            The matrix to be block svded.
        qnc1,qnc2: QuantumNumberCollection or None, optional
            1) QuantumNumberCollection
                The quantum number collections of the rows and columns of the matrix.
                NOTE: In this case, the quantum numbers of rows and columns should be one incoming and one outgoing.
            2) None
                They play no roles.
        mode: 'L' or 'R'
            When 'L', the renewed qnc1 will be returned;
            When 'R', the renewed qnc2 will be returned.
        nmax,tol,return_truncation_err: optional
            For details, please refer to HamiltonianPy.Math.linalg.truncated_svd
    Returns:
        U,S,V: ndarray
            The svd decomposition of M
        QNC: QuantumNumberCollection or integer
            Two cases,
            1) QuantumNumberCollection
                The new QuantumNumberCollection after the SVD.
                When qnc1 and qnc2 are QuantumNumberCollection.
            2) integer
                The number of the new singular values after the SVD.
                When qnc1 and qnc2 are None.
        err: float64, optional
            The truncation error.
    '''
    if isinstance(qnc1,QuantumNumberCollection) and isinstance(qnc2,QuantumNumberCollection):
        assert mode in 'LR'
        Us,Ss,Vs=[],[],[]
        for key in qnc2 if mode=='L' else qnc1:
            try:
                u,s,v=sl.svd(M[qnc1[key],qnc2[key]],full_matrices=False)
            except KeyError:
                if mode=='L':
                    n,m=0,qnc2[key].stop-qnc2[key].start
                else:
                    n,m=qnc1[key].stop-qnc1[key].start,0
                u=np.zeros((n,0))
                s=np.array([])
                v=np.zeros((0,m))
            Us.append(u)
            Ss.append(s)
            Vs.append(v)
        temp=np.sort(np.concatenate([-s for s in Ss]))
        nmax=len(temp) if nmax is None else min(nmax,len(temp))
        tol=temp[nmax-1] if tol is None else min(-tol,temp[nmax-1])
        U,S,V,contents=[],[],[],[]
        for u,s,v,qn in zip(Us,Ss,Vs,qnc2 if mode=='L' else qnc1):
            cut=np.searchsorted(-s,tol,side='right')
            U.append(u[:,0:cut])
            S.append(s[0:cut])
            V.append(v[0:cut,:])
            contents.append((qn,cut))
        U,S,V=sl.block_diag(*U),np.concatenate(S),sl.block_diag(*V)
        QNC=QuantumNumberCollection(contents)
        if mode=='L':
            permutation=np.argsort(qnc1.subslice(targets=QNC))
            U=U[permutation,:]
        else:
            permutation=np.argsort(qnc2.subslice(targets=QNC))
            V=V[:,permutation]
        if return_truncation_err:
            err=(temp[nmax:]**2).sum()
            return U,S,V,QNC,err
        else:
            return U,S,V,QNC
    else:
        temp=truncated_svd(M,full_matrices=False,nmax=nmax,tol=tol,return_truncation_err=return_truncation_err)
        U,S,V=temp[0],temp[1],temp[2]
        if return_truncation_err:
            err=temp[3]
            return U,S,V,len(S),err
        else:
            return U,S,V,len(S)
