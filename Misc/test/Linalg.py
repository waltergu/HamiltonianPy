'''
Linalg test.
'''

__all__=['test_linalg']

import numpy as np
import scipy.linalg as sl
from copy import deepcopy
from scipy.sparse.linalg import eigsh
from HamiltonianPy.Misc import Lanczos

def test_linalg():
    print 'test_linalg'
    test_lanczos()
    print

def test_lanczos():
    print 'test_lanczos'
    N,Nv,Ne,Niter=4000,15,1,750

    np.random.seed(1)
    m=np.random.random((N,N))+1j*np.random.random((N,N))
    m=m+m.T.conjugate()
    v0=np.random.random((Nv,N))
    exacteigs=sl.eigh(m,eigvals_only=True)[:Ne] if N<1000 else eigsh(m,k=Ne,return_eigenvectors=False,which='SA')[::-1]
    print 'Exact eigs:',exacteigs
    print

    a=Lanczos(m,deepcopy(v0),maxiter=Niter,keepstate=True)
    for i in xrange(Niter): a.iter()
    print 'diff from Hermitian: %s.'%sl.norm(a.T-a.T.T.conjugate())
    h=np.zeros((Niter,Niter),dtype=m.dtype)
    for i in xrange(Niter):
        hi=a.matrix.dot(a.vectors[i]).conjugate()
        for j in xrange(Niter):
            h[i,j]=hi.dot(a.vectors[j])
    print 'diff from h: %s.'%sl.norm(a.T-h)
    V=np.asarray(a.vectors)
    print 'vecotrs input diff: %s.'%sl.norm(V[0:Nv,:].T.dot(a.P)-v0.T)
    print 'vectors orthonormal diff: %s.'%sl.norm(np.identity(Niter)-V.dot(V.T.conjugate()))
    Leigs=a.eigs()[:Ne]
    print 'Lanczos eigs:',Leigs
    print 'eigenvalues diff: %s.'%sl.norm(exacteigs-Leigs)
    print
