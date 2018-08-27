'''
Structure test (4 tests in total).
'''

__all__=['structure']

from HamiltonianPy import *
from HamiltonianPy.TensorNetwork.Structure import *
from unittest import TestCase,TestLoader,TestSuite

class TestDegFreTree(TestCase):
    def setUp(self):
        self.flayers=DEGFRE_FERMIONIC_LAYERS
        fmap=lambda pid: Fock(norbital=1,nspin=2,nnambu=1)
        self.fconfig=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY,pids=[PID(scope,site) for site in (0,1) for scope in (0,1)],map=fmap)
        self.slayers=DEGFRE_SPIN_LAYERS
        smap=lambda pid: Spin(S=0.5)
        self.sconfig=IDFConfig(priority=DEFAULT_SPIN_PRIORITY,pids=[PID(scope,site) for site in (0,1) for scope in (0,1)],map=smap)

    def test_fermi_int(self):
        print()
        degmap=lambda index: 2
        tree=DegFreTree(layers=self.flayers,leaves=self.fconfig.table(mask=['nambu']),map=degmap)
        for layer in self.flayers:
            print('layer: %s'%(layer,))
            for i,index in enumerate(tree.indices(layer)):
                print('i,index,tree[index]: %s, %s, %s'%(i,repr(index),tree[index]))
            print()

    def test_fermi_qns(self):
        print()
        degmap=lambda index: SzPQNS(-0.5 if index.spin==0 else 0.5)
        tree=DegFreTree(layers=self.flayers,leaves=self.fconfig.table(mask=['nambu']),map=degmap)
        for layer in self.flayers:
            print('layer: %s'%(layer,))
            for i,index in enumerate(tree.indices(layer)):
                print('i,index,len(tree[index]),tree[index]: %s, %s, %s, %s'%(i,repr(index),len(tree[index]),tree[index]))
            print()

    def test_spin_int(self):
        print()
        degmap=lambda index: 2
        tree=DegFreTree(layers=self.slayers,leaves=self.sconfig.table(mask=[]),map=degmap)
        for layer in self.slayers:
            print('layer: %s'%(layer,))
            for i,index in enumerate(tree.indices(layer)):
                print('i,index,tree[index]: %s, %s, %s'%(i,repr(index),tree[index]))
            print()

    def test_spin_qns(self):
        print()
        degmap=lambda index: SQNS(0.5)
        tree=DegFreTree(layers=self.slayers,leaves=self.sconfig.table(mask=[]),map=degmap)
        for layer in self.slayers:
            print('layer: %s'%(layer,))
            for i,index in enumerate(tree.indices(layer)):
                print('i,index,len(tree[index]),tree[index]: %s, %s, %s, %s'%(i,repr(index),len(tree[index]),tree[index]))
            print()

structure=TestSuite([
            TestLoader().loadTestsFromTestCase(TestDegFreTree),
            ])
