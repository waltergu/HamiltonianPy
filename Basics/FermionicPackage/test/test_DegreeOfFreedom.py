'''
Fermionic degree of freedom test (7 tests in total).
'''

__all__=['fdegreeoffreedom']

from HamiltonianPy.Basics.Geometry import *
from HamiltonianPy.Basics.DegreeOfFreedom import *
from HamiltonianPy.Basics.FermionicPackage import *
from unittest import TestCase,TestLoader,TestSuite

class TestFID(TestCase):
    def setUp(self):
        self.fid=FID(orbital=1,spin=2,nambu=CREATION)

    def test_dagger(self):
        result=FID(orbital=1,spin=2,nambu=ANNIHILATION)
        self.assertEqual(self.fid.dagger,result)
        self.assertEqual(result.dagger,self.fid)

class TestIndex(TestCase):
    def setUp(self):
        self.index=Index(pid=PID(scope='WG',site=1),iid=FID(orbital=2,spin=3,nambu=ANNIHILATION))

    def test_replace(self):
        result=Index(pid=PID(scope='WG',site=1),iid=FID(orbital=2,spin=3,nambu=CREATION))
        self.assertEqual(self.index.replace(nambu=CREATION),result)

    def test_to_tuple(self):
        result=(1,2,3,0,'WG')
        self.assertEqual(self.index.to_tuple(['site','orbital','spin','nambu','scope']),result)

class TestFermi(TestCase):
    def setUp(self):
        self.fermi=Fermi(atom=0,norbital=2,nspin=2,nnambu=2)

    def test_indices(self):
        r1=[Index(pid=PID(scope='WG',site=0),iid=FID(orbital=ob,spin=sp,nambu=nb)) for nb in xrange(self.fermi.nnambu) for sp in xrange(self.fermi.nspin) for ob in xrange(self.fermi.norbital)]
        r2=[Index(pid=PID(scope='WG',site=0),iid=FID(orbital=ob,spin=sp,nambu=None))for sp in xrange(self.fermi.nspin) for ob in xrange(self.fermi.norbital)]
        r3=[Index(pid=PID(scope='WG',site=None),iid=FID(orbital=ob,spin=sp,nambu=None))for sp in xrange(self.fermi.nspin) for ob in xrange(self.fermi.norbital)]
        self.assertEqual(self.fermi.indices(PID(scope='WG',site=0),mask=[]),r1)
        self.assertEqual(self.fermi.indices(PID(scope='WG',site=0),mask=['nambu']),r2)
        self.assertEqual(self.fermi.indices(PID(scope='WG',site=0),mask=['nambu','site']),r3)

class TestIDFConfig(TestCase):
    def setUp(self):
        self.config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY)
        self.config[PID(scope='WG',site=0)]=Fermi(atom=0,norbital=1,nspin=2,nnambu=2)
        self.config[PID(scope='WG',site=1)]=Fermi(atom=1,norbital=1,nspin=2,nnambu=2)

    def test_table(self):
        r1=Table([
            Index(PID('WG',0),FID(0,0,0)),Index(PID('WG',0),FID(0,1,0)),
            Index(PID('WG',1),FID(0,0,0)),Index(PID('WG',1),FID(0,1,0)),
            Index(PID('WG',0),FID(0,0,1)),Index(PID('WG',0),FID(0,1,1)),
            Index(PID('WG',1),FID(0,0,1)),Index(PID('WG',1),FID(0,1,1)),
            ])
        r2=Table([
            Index(PID('WG',0),FID(0,0,None)),Index(PID('WG',0),FID(0,1,None)),
            Index(PID('WG',1),FID(0,0,None)),Index(PID('WG',1),FID(0,1,None)),
            ])
        self.assertEqual(self.config.table(mask=[]),r1)
        self.assertEqual(self.config.table(mask=['nambu']),r2)

class TestIndexPacks(TestCase):
    def test_algebra(self):
        a=FermiPack(1.0,orbitals=[0,0])
        b=FermiPack(2.0,atoms=[0,0])
        c=FermiPack(3.0,spins=[0,0])
        self.assertEqual(str(a+b+c),"IndexPacks(1.0*ob00,2.0*sl00,3.0*sp00)")
        self.assertEqual((a+b)+c,a+(b+c))
        self.assertEqual(repr(a*b*c),"6.0*sl00*ob00*sp00")
        self.assertEqual((a*b)*c,a*(b*c))

    def test_functions(self):
        self.assertEqual(str(sigma0('sp')),"IndexPacks(1.0*sp00,1.0*sp11)")
        self.assertEqual(str(sigma0('ob')),"IndexPacks(1.0*ob00,1.0*ob11)")
        self.assertEqual(str(sigma0('sl')),"IndexPacks(1.0*sl00,1.0*sl11)")
        self.assertEqual(str(sigma0('ph')),"IndexPacks(1.0*ph01,1.0*ph10)")
        self.assertEqual(str(sigmax('sp')),"IndexPacks(1.0*sp01,1.0*sp10)")
        self.assertEqual(str(sigmax('ob')),"IndexPacks(1.0*ob01,1.0*ob10)")
        self.assertEqual(str(sigmax('sl')),"IndexPacks(1.0*sl01,1.0*sl10)")
        self.assertEqual(str(sigmax('ph')),"IndexPacks(1.0*ph00,1.0*ph11)")
        self.assertEqual(str(sigmay('sp')),"IndexPacks(1.0j*sp01,-1.0j*sp10)")
        self.assertEqual(str(sigmay('ob')),"IndexPacks(1.0j*ob01,-1.0j*ob10)")
        self.assertEqual(str(sigmay('sl')),"IndexPacks(1.0j*sl01,-1.0j*sl10)")
        self.assertEqual(str(sigmay('ph')),"IndexPacks(1.0j*ph00,-1.0j*ph11)")
        self.assertEqual(str(sigmaz('sp')),"IndexPacks(-1.0*sp00,1.0*sp11)")
        self.assertEqual(str(sigmaz('ob')),"IndexPacks(-1.0*ob00,1.0*ob11)")
        self.assertEqual(str(sigmaz('sl')),"IndexPacks(-1.0*sl00,1.0*sl11)")
        self.assertEqual(str(sigmaz('ph')),"IndexPacks(-1.0*ph01,1.0*ph10)")
        self.assertEqual(str(sigmap('sp')),"IndexPacks(1.0*sp10)")
        self.assertEqual(str(sigmap('ob')),"IndexPacks(1.0*ob10)")
        self.assertEqual(str(sigmap('sl')),"IndexPacks(1.0*sl10)")
        self.assertEqual(str(sigmap('ph')),"IndexPacks(1.0*ph11)")
        self.assertEqual(str(sigmam('sp')),"IndexPacks(1.0*sp01)")
        self.assertEqual(str(sigmam('ob')),"IndexPacks(1.0*ob01)")
        self.assertEqual(str(sigmam('sl')),"IndexPacks(1.0*sl01)")
        self.assertEqual(str(sigmam('ph')),"IndexPacks(1.0*ph00)")

fdegreeoffreedom=TestSuite([
                    TestLoader().loadTestsFromTestCase(TestFID),
                    TestLoader().loadTestsFromTestCase(TestIndex),
                    TestLoader().loadTestsFromTestCase(TestFermi),
                    TestLoader().loadTestsFromTestCase(TestIDFConfig),
                    TestLoader().loadTestsFromTestCase(TestIndexPacks),
                    ])
