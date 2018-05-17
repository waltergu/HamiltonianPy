'''
Fermionic term test (3 tests in total).
'''

__all__=['fockterm']

from HamiltonianPy.Basics.Geometry import *
from HamiltonianPy.Basics.DegreeOfFreedom import *
from HamiltonianPy.Basics.Operator import *
from HamiltonianPy.Basics.FockPackage import *
from unittest import TestCase,TestLoader,TestSuite

class TestQuadratic(TestCase):
    def setUp(self):
        p1=Point(pid=PID(scope='WG',site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0])
        p2=Point(pid=PID(scope='WG',site=1),rcoord=[1.0,0.0],icoord=[0.0,0.0])
        self.lattice=Lattice.compose(name="WG",points=[p1,p2])
        self.config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY)
        self.config[p1.pid]=Fock(atom=0,norbital=2,nspin=2,nnambu=2)
        self.config[p2.pid]=Fock(atom=1,norbital=2,nspin=2,nnambu=2)
        self.hopping=Hopping('t',1.0,neighbour=1,indexpacks=sigmax("SL")*sigmax("SP"))
        self.onsite=Onsite('mu',1.0,indexpacks=sigmaz("SP")*sigmay("OB"))
        self.pairing=Pairing('delta',1.0,neighbour=1,indexpacks=sigmaz("SP")+sigmay("OB"))

    def test_operators(self):
        print
        opts=Operators()
        for bond in self.lattice.bonds:
            opts+=self.hopping.operators(bond,self.config,self.config.table(mask=[]))
            opts+=self.onsite.operators(bond,self.config,self.config.table(mask=[]))
            opts+=self.pairing.operators(bond,self.config,self.config.table(mask=[]))
        self.assertEqual(len(opts),24)
        print repr(opts)

class TestHubbard(TestCase):
    def setUp(self):
        p=Point(PID(scope="WG",site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0])
        self.lattice=Lattice.compose(name="WG",points=[p],neighbours=0)
        self.config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY)
        self.config[p.pid]=Fock(norbital=2,nspin=2,nnambu=1)
        self.hubbard=Hubbard('UUJJ',[20.0,12.0,5.0,5.0])

    def test_operators(self):
        print
        opts=Operators()
        for bond in self.lattice.bonds:
            opts+=self.hubbard.operators(bond,self.config,self.config.table(mask=['nambu']))
        self.assertEqual(len(opts),8)
        print repr(opts)

class TestCoulomb(TestCase):
    def setUp(self):
        p1=Point(PID(scope='WG',site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0])
        p2=Point(PID(scope='WG',site=1),rcoord=[1.0,0.0],icoord=[0.0,0.0])
        self.lattice=Lattice.compose(name='WG',points=[p1,p2],neighbours=1)
        self.config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY,pids=self.lattice.pids,map=lambda pid: Fock(norbital=1,nspin=2,nnambu=1))
        self.U=Coulomb('U',1.0,neighbour=0,indexpacks=(sigmap('sp'),sigmam('sp')))
        self.V=Coulomb('V',8.0,neighbour=1,indexpacks=(sigmaz('sp'),sigmaz('sp')))

    def test_operators(self):
        print 
        opts=Operators()
        for bond in self.lattice.bonds:
            opts+=self.U.operators(bond,self.config,self.config.table(mask=['nambu']))
            opts+=self.V.operators(bond,self.config,self.config.table(mask=['nambu']))
        self.assertEqual(len(opts),6)
        print repr(opts)

fockterm=TestSuite([
            TestLoader().loadTestsFromTestCase(TestQuadratic),
            TestLoader().loadTestsFromTestCase(TestHubbard),
            TestLoader().loadTestsFromTestCase(TestCoulomb),
            ])