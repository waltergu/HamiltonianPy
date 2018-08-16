'''
Linalg test (5 tests in total).
'''

__all__=['linalg']

import numpy as np
import scipy.linalg as sl
from copy import deepcopy
from scipy.sparse.linalg import eigsh
from HamiltonianPy.Misc import Lanczos
from unittest import TestCase,TestLoader,TestSuite

class TestLanczos(TestCase):
    def setUp(self):
        np.random.seed(1)
        N,Nv,Niter=4000,15,750
        m=np.random.random((N,N))+1j*np.random.random((N,N))
        m+=m.T.conjugate()
        self.v0=np.random.random((Nv,N))
        self.lanczos=Lanczos(m,deepcopy(self.v0),maxiter=Niter,keepstate=True)
        for _ in range(Niter): self.lanczos.iter()

    def test_hermiticity(self):
        self.assertAlmostEqual(sl.norm(self.lanczos.T-self.lanczos.T.T.conjugate()),0.0)

    def test_deviation(self):
        print()
        h=np.zeros((self.lanczos.niter,self.lanczos.niter),dtype=self.lanczos.matrix.dtype)
        for i in range(self.lanczos.niter):
            hi=self.lanczos.matrix.dot(self.lanczos.vectors[i]).conjugate()
            for j in range(self.lanczos.niter):
                h[i,j]=hi.dot(self.lanczos.vectors[j])
        print('diff from h: %s.'%sl.norm(self.lanczos.T-h))

    def test_reconstruction(self):
        V=np.asarray(self.lanczos.vectors)
        self.assertAlmostEqual(sl.norm(V[0:self.lanczos.nv0,:].T.dot(self.lanczos.P)-self.v0.T)/sl.norm(self.v0),0.0,delta=0.01)

    def test_orthonormality(self):
        V=np.asarray(self.lanczos.vectors)
        self.assertAlmostEqual(sl.norm(np.identity(self.lanczos.niter)-V.dot(V.T.conjugate()))/sl.norm(V),0.0,delta=0.5)

    def test_eigs(self):
        Ne=1
        exacteigs=sl.eigh(self.lanczos.matrix,eigvals_only=True)[:Ne] if self.lanczos.matrix.size<1000**2 else eigsh(self.lanczos.matrix,k=Ne,return_eigenvectors=False,which='SA')[::-1]
        Leigs=self.lanczos.eigs()[:Ne]
        self.assertAlmostEqual(sl.norm(exacteigs-Leigs),0.0)

linalg=TestSuite([
                TestLoader().loadTestsFromTestCase(TestLanczos),
                ])
