'''
Fermionic operator representation test (1 test in total).
'''

__all__=['fockoptrep']

from HamiltonianPy.Basics import *
from unittest import TestCase,TestLoader,TestSuite
import numpy as np
import itertools as it
import time

class Test_foptrep(TestCase):
    def setUp(self):
        m,n=3,4
        point,a1,a2=np.array([0.0,0.0]),np.array([1.0,0.0]),np.array([0.0,1.0])
        self.lattice=Lattice(name="WG",rcoords=tiling([point],vectors=[a1,a2],translations=it.product(xrange(m),xrange(n))))
        self.config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY,map=lambda pid:Fock(norbital=1,nspin=2,nnambu=2),pids=self.lattice.pids)
        self.terms=[Hopping('t',1.0,neighbour=1,indexpacks=sigmaz("SP"))]
        self.generator=Generator(bonds=self.lattice.bonds,config=self.config,table=self.config.table(mask=[]),terms=self.terms)
        self.bases=[FBasis(nstate=2*m*n),FBasis(nstate=2*m*n,nparticle=m*n),FBasis(nstate=2*m*n,nparticle=m*n,spinz=0.0)]

    def test_time(self):
        print
        for operator in self.generator.operators:
            for basis in self.bases:
                stime=time.time()
                matrix=foptrep(operator,basis,transpose=False)
                etime=time.time()
                print '%s mode: shape=%s, nnz=%s, time=%ss.'%(basis.mode,matrix.shape,matrix.nnz,etime-stime)
            print

fockoptrep=TestSuite([
            TestLoader().loadTestsFromTestCase(Test_foptrep),
            ])
