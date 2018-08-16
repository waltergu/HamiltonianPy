'''
TrFED test.
'''

__all__=['trfed']

import time
import numpy as np
from HamiltonianPy import *
from HamiltonianPy.ED import *
from HamiltonianPy.Beta.TrED import *
from scipy.linalg import eigh,norm
from unittest import TestCase,TestLoader,TestSuite

class TestTrFBasis(TestCase):
    def test_time(self):
        print()
        m=16
        stime=time.time()
        nmbasis=FBasis(m*2,m//2,0.0)
        etime=time.time()
        print('NRM basis: %ss.'%(etime-stime))
        stime=time.time()
        TrFBasis(nmbasis,dk=4,nk=8)
        etime=time.time()
        print('TRI basis: %ss'%(etime-stime))

class TestTrFED(TestCase):
    def test_trfed(self):
        t1,U,m=1.0,4.0,4
        lattice=Square('S1')('%sP-1O'%m)
        config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY,pids=lattice.pids,map=lambda pid: Fock(norbital=1,nspin=2,nnambu=1))
        basis=FBasis(m*2,m,0.0)
        ed=FED(name='OneD_%s_%r'%(lattice.name,basis),sectors=[basis],lattice=lattice,config=config,terms=[Hopping('t1',t1),Hubbard('U',U)],dtype=np.complex128)
        eigvals0=eigh(ed.matrix(basis).todense(),eigvals_only=True)
        basis=TrFBasis(FBasis(m*2,m,0.0),dk=2,nk=m)
        ed=TrFED(name='OneD_%s_%r'%(lattice.name,basis),basis=basis,lattice=lattice,config=config,terms=[Hopping('t1',t1),Hubbard('U',U)],dtype=np.complex128)
        eigvals1=[]
        for k in range(m): eigvals1.append(eigh(ed.matrix(sector=k).todense(),eigvals_only=True))
        eigvals1=sorted(np.concatenate(eigvals1))
        self.assertAlmostEqual(norm(eigvals0-eigvals1),0.0)

trfed=TestSuite([
            TestLoader().loadTestsFromTestCase(TestTrFBasis),
            TestLoader().loadTestsFromTestCase(TestTrFED),
            ])

if __name__=='__main__':
    from unittest import main
    main(verbosity=2)
