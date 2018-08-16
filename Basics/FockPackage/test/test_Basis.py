'''
FBasis test (2 tests in total).
'''
__all__=['fockbasis']

from HamiltonianPy.Basics.FockPackage.Basis import *
from unittest import TestCase,TestLoader,TestSuite

class TestFBasis(TestCase):
    def setUp(self):
        self.nstate=4
        self.nparticle=2
        self.spinz=1

    def test_fbasis(self):
        print()
        for basis in [  FBasis(nstate=self.nstate,nparticle=self.nparticle,spinz=self.spinz),
                        FBasis(nstate=self.nstate,nparticle=self.nparticle),
                        FBasis(nstate=self.nstate)
                        ]:
            print('%r\n%s\n'%(basis,basis))

    def test_fbases(self):
        print()
        for basis in FBases(mode='FS',nstate=self.nstate,select=lambda n,sz: True if n%2==0 and sz==0 else False):
            print('%r\n%s\n'%(basis,basis))
        for basis in FBases(mode='FP',nstate=self.nstate):
            print('%r\n%s\n'%(basis,basis))
        for basis in FBases(mode='FG',nstate=self.nstate):
            print('%r\n%s\n'%(basis,basis))

fockbasis=TestSuite([
            TestLoader().loadTestsFromTestCase(TestFBasis),
            ])
