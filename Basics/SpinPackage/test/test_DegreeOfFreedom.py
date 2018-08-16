'''
Spin degree of freedom test (5 tests in total).
'''

__all__=['sdegreeoffreedom']

import numpy as np
from HamiltonianPy.Basics.Geometry import *
from HamiltonianPy.Basics.DegreeOfFreedom import *
from HamiltonianPy.Basics.SpinPackage import *
from unittest import TestCase,TestLoader,TestSuite

class TestSID(TestCase):
    def test_sid(self):
        print()
        print(SID(orbital=0,S=2))

class TestSpin(TestCase):
    def setUp(self):
        self.spin=Spin(norbital=2,S=2)

    def test_indices(self):
        r1=[Index(pid=PID(scope='WG',site=0),iid=SID(orbital=ob,S=self.spin.S)) for ob in range(self.spin.norbital)]
        r2=[Index(pid=PID(scope='WG',site=None),iid=SID(orbital=ob,S=self.spin.S)) for ob in range(self.spin.norbital)]
        r3=[Index(pid=PID(scope='WG',site=None),iid=SID(orbital=None,S=self.spin.S))]
        self.assertEqual(self.spin.indices(PID(scope='WG',site=0),mask=[]),r1)
        self.assertEqual(self.spin.indices(PID(scope='WG',site=0),mask=['site']),r2)
        self.assertEqual(self.spin.indices(PID(scope='WG',site=0),mask=['site','orbital']),r3)

class TestSpinMatrix(TestCase):
    def test_spinmatrix(self):
        print()
        N=1
        print(SpinMatrix(N,'x',dtype=np.float64))
        print(SpinMatrix(N,'y',dtype=np.complex128))
        print(SpinMatrix(N,'z',dtype=np.float64))
        print(SpinMatrix(N,'p',dtype=np.float64))
        print(SpinMatrix(N,'m',dtype=np.float64))
        print(SpinMatrix(N,'WG',matrix=np.random.random((2*N+1,2*N+1)),dtype=np.float64))

class TestIndexPacks(TestCase):
    def test_algebra(self):
        a=SpinPack(1.0,tags=('x','x'))
        b=SpinPack(-2.0,tags=('W1','W2'),matrices=(np.random.random((2,2)),np.random.random((2,2))))
        self.assertEqual(repr(a*2),"2.0*xx")
        self.assertEqual(a*2,2*a)
        self.assertEqual(str(a+b),"IndexPacks(1.0*xx,-2.0*W1W2)")

    def test_functions(self):
        self.assertEqual(str(Heisenberg()),"IndexPacks(0.5*pm,0.5*mp,1.0*zz)")
        self.assertEqual(str(Ising('x',orbitals=(0,1))),"IndexPacks(1.0*xx*ob01)")
        self.assertEqual(str(Ising('y',orbitals=(0,1))),"IndexPacks(1.0*yy*ob01)")
        self.assertEqual(str(Ising('z',orbitals=(0,1))),"IndexPacks(1.0*zz*ob01)")
        self.assertEqual(str(S('z',orbital=0)),"IndexPacks(1.0*z*ob0)")

sdegreeoffreedom=TestSuite([
                    TestLoader().loadTestsFromTestCase(TestSID),
                    TestLoader().loadTestsFromTestCase(TestSpin),
                    TestLoader().loadTestsFromTestCase(TestSpinMatrix),
                    TestLoader().loadTestsFromTestCase(TestIndexPacks),
                    ])
