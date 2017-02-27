'''
Linear algebra, including:
1) functions: vb_svd
'''

import numpy as np
import scipy.linalg as sl
from HamiltonianPy import QuantumNumbers
from ..Misc import truncated_svd,block_diag

__all__=['vb_svd']

def vb_svd(Psi,qnc1,qnc2,mode='L',nmax=None,tol=None,return_truncation_err=False):
    '''
    Block svd of a wavefunction Psi according to the bipartition information passed by qnc1 and qnc2.
    Parameters:
        Psi: 1D ndarray
            The wavefunction to be block svded.
        qnc1,qnc2: QuantumNumbers or integer
            1) QuantumNumbers
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
        QNC: QuantumNumbers or integer
            Its type coincides with those of qnc1 and qnc2.
            1) QuantumNumbers
                The new QuantumNumbers after the SVD.
            2) integer
                The number of the new singular values after the SVD.
        err: float64, optional
            The truncation error.
    '''
    if isinstance(qnc1,QuantumNumbers) and isinstance(qnc2,QuantumNumbers):
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
        QNC=QuantumNumbers(contents)
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
