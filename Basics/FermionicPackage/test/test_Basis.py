'''
FBasis test (2 tests in total).
'''
__all__=['fbasis']

from HamiltonianPy.Basics.FermionicPackage.Basis import *
from unittest import TestCase,TestLoader,TestSuite

class TestFBasis(TestCase):
    def setUp(self):
        self.nstate=4
        self.nparticle=2
        self.spinz=1

    def test_fbasis(self):
        print
        for basis in [  FBasis(nstate=self.nstate,nparticle=self.nparticle,spinz=self.spinz),
                        FBasis(nstate=self.nstate,nparticle=self.nparticle),
                        FBasis(nstate=self.nstate)
                        ]:
            print '%s\n%s\n'%(basis.rep,basis)

    def test_fbases(self):
        print
        for basis in FBases(mode='FS',nstate=self.nstate,select=lambda n,sz: True if n%2==0 and sz==0 else False):
            print '%s\n%s\n'%(basis.rep,basis)
        for basis in FBases(mode='FP',nstate=self.nstate):
            print '%s\n%s\n'%(basis.rep,basis)
        for basis in FBases(mode='FG',nstate=self.nstate):
            print '%s\n%s\n'%(basis.rep,basis)

fbasis=TestSuite([
            TestLoader().loadTestsFromTestCase(TestFBasis),
            ])
