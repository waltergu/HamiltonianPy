'''
Linear algebra, including:
1) functions: bond_qnc_generation,vb_svd,mb_svd,expanded_svd
'''

import numpy as np
import scipy.linalg as sl
import scipy.sparse as sp
from HamiltonianPy import QuantumNumberCollection
from ..Misc import TOL,truncated_svd,block_diag
from itertools import ifilter

__all__=['bond_qnc_generation','vb_svd','mb_svd','expanded_svd']

def bond_qnc_generation(m,bond,site,mode='L',history=False):
    '''
    Generate the qnc of the left/right dimension of m.
    Parameters:
        m: 3d ndarray-like
            The matrix whose qnc is to be generated.
        bond,site: QuantumNumberCollection
            The bond qnc and site qnc of m.
        mode: 'L' or 'R', optional
            When 'L', the left bond qnc is to be generated;
            When 'R', the right bond qnc is to be generated.
        history: True or False, optional
            When True, the permutation information will be recorded;
            Otherwise not.
    Returns: QuantumNumberCollection
        The generated bond qnc.
    '''
    assert m.ndim==3 and mode in 'LR'
    bqnc,sqnc=bond.expansion(),site.expansion()
    if mode=='L':
        indices=sorted(np.argwhere(np.abs(m)>TOL),key=lambda row: row[0])
        result=[None]*m.shape[0]
        for index in indices:
            L,S,R=index
            if result[L] is None:
                result[L]=bqnc[R]-sqnc[S]
            else:
                assert result[L]==bqnc[R]-sqnc[S]
    else:
        indices=sorted(np.argwhere(np.abs(m)>TOL),key=lambda row: row[2])
        result=[None]*m.shape[2]
        for index in indices:
            L,S,R=index
            if result[R] is None:
                result[R]=bqnc[L]+sqnc[S]
            else:
                assert result[R]==bqnc[L]+sqnc[S]
    return QuantumNumberCollection(result,history=history)

def vb_svd(Psi,qnc1,qnc2,mode='L',nmax=None,tol=None,return_truncation_err=False):
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
            u,s,v=sl.svd(Psi[count:count+n1*n2].reshape((n1,n2)),full_matrices=False,lapack_driver='gesvd')
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
        U,S,V=block_diag(*U),np.concatenate(S),block_diag(*V)
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

def mb_svd(M,qnc1=None,qnc2=None,nmax=None,tol=None,return_truncation_err=False):
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
        us,ss,vs,qns=[],[],[],[]
        for qn in ifilter(qnc1.has_key,qnc2):
            u,s,v=sl.svd(M[qnc1[qn],qnc2[qn]],full_matrices=False,lapack_driver='gesvd')
            us.append(u)
            ss.append(s)
            vs.append(v)
            qns.append(qn)
        temp=np.sort(np.concatenate([-s for s in ss]))
        nmax=len(temp) if nmax is None else min(nmax,len(temp))
        tol=temp[nmax-1] if tol is None else min(-tol,temp[nmax-1])
        Us,Ss,Vs,contents=[],[],[],[]
        for u,s,v,qn in zip(us,ss,vs,qns):
            cut=np.searchsorted(-s,tol,side='right')
            if cut>0:
                Us.append(u[:,0:cut])
                Ss.append(s[0:cut])
                Vs.append(v[0:cut,:])
                contents.append((qn,cut))
        S,QNC=np.concatenate(Ss),QuantumNumberCollection(contents)
        U,V=np.zeros((qnc1.n,QNC.n),dtype=M.dtype),np.zeros((QNC.n,qnc2.n),dtype=M.dtype)
        for u,v,qn in zip(Us,Vs,QNC):
            U[qnc1[qn],QNC[qn]]=u
            V[QNC[qn],qnc2[qn]]=v
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

def expanded_svd(m,expansion,L=None,S=None,R=None,mode='vs',nmax=None,tol=None):
    '''
    Expand the physical index of m and perform a sequential svd.
    Parameters:
        m: 3d ndarray
            The matrix to be expanded and svded.
        expansion: list of QuantumNumberCollection or list of integer
            When QuantumNumberCollections, they are the expanded quantum number collections of the physical index of m;
            When integers, they are the expanded dimensions of the physical index of m.
        L,S,R: QuantumNumberCollection, optional
            The quantum number collections of the left, physical and right index of m.
        mode: 'vs' or 'us', optional
            When 'vs', the expanded physical indices will be tied with the v of all svds;
            When 'us', the expanded physical indices will be tied with the u of all svds.
        namx: integer, optional
            The maximum number of singular values to be kept.
        tol: float64
            The tolerance of the singular values.
    Returns: two cases
        1) mode=='vs': u,s,vs,qncs
            u: 2d ndarray
                The final u matrix of the sequential svd.
            s: 1d ndarray
                The final singular values of the sequential svd.
            vs: list of 3d ndarray
                The reshaped v matrices of the sequential svd,
                whose dimensions correspond to the wanted left, physical and right indices.
            qncs: list of QuantumNumberCollection or list of integer
                The quantum number collections or the dimensions of the generated internal legs.
        2) mode=='us': us,s,v,qncs
            us: list of 3d ndarray
                The reshaped u matrices of the sequential svd,
                whose dimensions correspond to the wanted left, physical and right indices.
            s: 1d ndarray
                The final singular values of the sequential svd.
            v: 2d ndarray
                The final v matrix of the sequential svd.
            qncs: list of QuantumNumberCollection or list of integer
                The quantum number collections or the dimensions of the generated internal legs.
    '''
    assert mode in ('vs','us')
    if all([isinstance(qnc,QuantumNumberCollection) for qnc in expansion]):
        assert isinstance(L,QuantumNumberCollection) and isinstance(S,QuantumNumberCollection) and isinstance(R,QuantumNumberCollection)
        if mode=='vs':
            evolution=[None]*len(expansion)
            for i,qnc in enumerate(expansion):
                evolution[i]=qnc if i==0 else evolution[i-1].kron(qnc,action='+',history=True)
            vs,qncs=[],[]
            for i in reversed(xrange(len(expansion))):
                if i<len(expansion)-1:
                    m=np.einsum('ij,j->ij',u,s).reshape((L.n,evolution[i].n,R.n))
                row=L.kron(evolution[i-1],action='+',history=True) if i>0 else L
                col=expansion[i].kron(R,action='-',history=True)
                m=m[:,np.argsort(evolution[i].permutation()),:].reshape((row.n,col.n))
                m=row.reorder(col.reorder(m,axes=[1]),axes=[0])
                u,s,v,qnc,err=mb_svd(m,row,-col,nmax=nmax,tol=tol,return_truncation_err=True)
                vs.insert(0,v[:,np.argsort(col.permutation())].reshape((qnc.n,expansion[i].n,R.n)))
                qncs.insert(0,qnc)
                u,R=u[np.argsort(row.permutation()),:],qnc
                QuantumNumberCollection.clear_history(row,col)
            QuantumNumberCollection.clear_history(*evolution)
            return u,s,vs,qncs
        else:
            # NOT DEBUGED COED
            evolution=[None]*len(expansion)
            for i in reversed(range(len(expansion))):
                evolution[i]=expansion[i].kron(evolution[i+1],action='+',history=True) if i+2<len(expansion) else expansion[i]
            us,qncs=[],[]
            for i in xrange(len(expansion)):
                if i>0:
                    m=np.einsum('i,ij->ij',s,v).reshape((L.n,evolution[i].n,R.n))
                row=L.kron(expansion[i],action='+',history=True)
                col=evolution[i+1].kron(R,action='-',history=True) if i+1<len(expansion) else R
                m=m[:,np.argsort(evolution[i]).permutation(),:].reshape((row.n,col.n))
                m=row.reorder(col.reorder(m,axes=[1]),axes=[0])
                u,s,v,qnc,err=mb_svd(m,row,-col,nmax=nmax,tol=tol,return_truncation_err=True)
                us.append(u[np.argsort(row.permutation()),:].reshape((L.n,expansion[i].n,qnc.n)))
                qncs.append(qnc)
                v,L=v[:,np.argsort(col.permutation())],qnc
                QuantumNumberCollection.clear_history(row,col)
            QuantumNumberCollection.clear_history(*evolution)
            return us,s,v,qncs
    else:
        L,S,R=m.shape
        if mode=='vs':
            vs,qncs=[],[]
            for i in reversed(xrange(len(expansion))):
                if i<len(expansion)-1: m=np.einsum('ij,j->ij',u,s).reshape((L,S,R))
                S=S/expansion[i]
                m=m.reshape((L*S,expansion[i]*R))
                u,s,v,err=truncated_svd(m,full_matrices=False,nmax=nmax,tol=tol,return_truncation_err=True)
                vs.insert(0,v.reshape((len(s),expansion[i],R)))
                qncs.insert(0,len(s))
                R=len(s)
            return u,s,vs,qncs
        else:
            # NOT DEBUGED COED
            us,qncs=[],[]
            for i in xrange(len(expansion)):
                if i>0: m=np.einsum('i,ij->ij',s,v).reshape((L,S,R))
                S=S/expansion[i]
                m=m.reshape((L*expansion[i],S*R))
                u,s,v,err=truncated_svd(m,full_matrices=False,nmax=nmax,tol=tol,return_truncation_err=True)
                us.append(u.reshape((L,expansion[i],len(s))))
                qncs.append(len(s))
                L=len(s)
            return us,s,v,qncs
