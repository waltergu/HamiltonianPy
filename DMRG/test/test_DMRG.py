'''
DMRG test.
'''

__all__=['dmrg']

import mkl
import numpy as np
from HamiltonianPy import *
from HamiltonianPy.TensorNetwork import *
from HamiltonianPy.DMRG import *
from unittest import TestCase,TestLoader,TestSuite

mkl.set_num_threads(1)
Engine.DEBUG=True
savedata,qnon=False,True
mode='QN' if qnon else 'NB'

class TestSpin(TestCase):
    def setUp(self):
        J,spin=1.0,0.5
        priority,layers=DEGFRE_SPIN_PRIORITY,DEGFRE_SPIN_LAYERS
        self.dmrg=fDMRG(
                name=       'spin-%s(%s)'%(spin,mode),
                lattice=    Cylinder(name='WG',block=[np.array([0.0,0.0])],translation=np.array([1.0,0.0]),neighbours=1),
                terms=      [SpinTerm('J',J,neighbour=1,indexpacks=Heisenberg())],
                config=     IDFConfig(priority=priority,map=lambda pid: Spin(S=spin)),
                degfres=    DegFreTree(mode=mode,layers=layers,priority=priority,map=lambda index: SQNS(S=spin) if qnon else int(spin*2+1)),
                mask=       [],
                dtype=      np.float64
                )

    def test_idmrg(self):
        print
        N=200
        targets=[SQN(0.0)]*(N/2) if qnon else [None]*(N/2)
        self.dmrg.register(TSG(name='iGROWTH',targets=targets,nmax=200,savedata=savedata,run=fDMRGTSG))
        self.dmrg.summary()

    def test_fdmrg(self):
        print
        N=20
        targets=[SQN(0.0)]*(N/2) if qnon else [None]*(N/2)
        self.dmrg.add(TSG(name='fGROWTH',targets=targets,nmax=200,savedata=savedata,plot=False,run=fDMRGTSG))
        self.dmrg.register(TSS(name='SWEEP',target=targets[-1],nsite=N*self.dmrg.nspb,nmaxs=[200,200],dependences=['fGROWTH'],savedata=savedata,run=fDMRGTSS))
        self.dmrg.summary()

class TestSpinlessFermion(TestCase):
    def setUp(self):
        t=-0.5
        priority,layers=DEGFRE_FERMIONIC_PRIORITY,DEGFRE_FERMIONIC_LAYERS
        self.dmrg=fDMRG(
                name=       'fermion-spinless(%s)'%mode,
                lattice=    Cylinder(name='WG',block=[np.array([0.0,0.0])],translation=np.array([1.0,0.0])),
                terms=      [Hopping('t',t,neighbour=1)],
                config=     IDFConfig(priority=priority,map=lambda pid: Fock(atom=0,norbital=1,nspin=1,nnambu=1)),
                degfres=    DegFreTree(mode=mode,layers=layers,priority=priority,map=lambda index: PQNS(1) if qnon else 2),
                mask=       ['nambu'],
                dtype=      np.float64
                )

    def test_idmrg(self):
        print
        N=200
        targets=[PQN(num) for num in xrange(1,N/2+1)] if qnon else [None]*(N/2)
        self.dmrg.register(TSG(name='iGROWTH',targets=targets,nmax=200,savedata=savedata,run=fDMRGTSG))
        self.dmrg.summary()

    def test_fdmrg(self):
        print
        N=20
        targets=[PQN(num) for num in xrange(1,N/2+1)] if qnon else [None]*(N/2)
        self.dmrg.add(TSG(name='fGROWTH',targets=targets,nmax=200,savedata=savedata,plot=False,run=fDMRGTSG))
        self.dmrg.register(TSS(name='SWEEP',target=targets[-1],nsite=N*self.dmrg.nspb,nmaxs=[200,200],dependences=['fGROWTH'],savedata=savedata,run=fDMRGTSS))
        self.dmrg.summary()

class TestSpinfulFermion(TestCase):
    def setUp(self):
        t,U=-1.0,1.0
        priority,layers=DEGFRE_FERMIONIC_PRIORITY,DEGFRE_FERMIONIC_LAYERS
        self.dmrg=fDMRG(
                name=       'fermion-spinful(%s)'%mode,
                lattice=    Cylinder(name='WG',block=[np.array([0.0,0.0])],translation=np.array([1.0,0.0])),
                terms=      [Hopping('t',t,neighbour=1),Hubbard('U',U)],
                config=     IDFConfig(priority=priority,map=lambda pid: Fock(atom=0,norbital=1,nspin=2,nnambu=1)),
                degfres=    DegFreTree(mode=mode,layers=layers,priority=priority,map=lambda index: SzPQNS(index.spin-0.5) if qnon else 2),
                mask=       ['nambu'],
                dtype=      np.float64
                )

    def test_idmrg(self):
        print
        N=200
        targets=[SPQN((num*2,0.0)) for num in xrange(1,N/2+1)] if qnon else [None]*(N/2)
        self.dmrg.register(TSG(name='iGROWTH',targets=targets,nmax=200,savedata=savedata,run=fDMRGTSG))
        self.dmrg.summary()

    def test_fdmrg(self):
        print
        N=20
        targets=[SPQN((num*2,0.0)) for num in xrange(1,N/2+1)] if qnon else [None]*(N/2)
        self.dmrg.add(TSG(name='fGROWTH',targets=targets,nmax=200,savedata=savedata,plot=False,run=fDMRGTSG))
        self.dmrg.register(TSS(name='SWEEP',target=targets[-1],nsite=N*self.dmrg.nspb,nmaxs=[200,200],dependences=['fGROWTH'],savedata=savedata,run=fDMRGTSS))
        self.dmrg.summary()

class TestHoneycombHeisenberg(TestCase):
    def setUp(self):
        J=1.0
        h4,priority,layers=Hexagon(name='H4'),DEGFRE_SPIN_PRIORITY,DEGFRE_SPIN_LAYERS
        self.dmrg=fDMRG(
                name=       'honeycomb-heisenberg(%s)'%mode,
                lattice=    Cylinder(name='WG',block=h4.rcoords,translation=h4.vectors[0],vectors=[h4.vectors[1]],neighbours=1),
                terms=      [SpinTerm('J',J,neighbour=1,indexpacks=Heisenberg())],
                config=     IDFConfig(priority=priority,map=lambda pid: Spin(S=0.5)),
                degfres=    DegFreTree(mode=mode,layers=layers,priority=priority,map=lambda index: SQNS(S=0.5) if qnon else 2),
                mask=       [],
                dtype=      np.float64
                )

    def test_idmrg(self):
        print
        N=20
        targets=[SQN(0.0)]*(N/2) if qnon else [None]*(N/2)
        self.dmrg.register(TSG(name='iGROWTH',targets=targets,nmax=100,savedata=savedata,run=fDMRGTSG))
        self.dmrg.summary()

    def test_fdmrg(self):
        print
        N=10
        targets=[SQN(0.0)]*(N/2) if qnon else [None]*(N/2)
        self.dmrg.add(TSG(name='fGROWTH',targets=targets,nmax=100,savedata=savedata,plot=False,run=fDMRGTSG))
        self.dmrg.register(TSS(name='SWEEP',target=targets[-1],nsite=N*self.dmrg.nspb,nmaxs=[100,100],dependences=['fGROWTH'],savedata=savedata,run=fDMRGTSS))
        self.dmrg.summary()

dmrg=TestSuite([
            TestLoader().loadTestsFromTestCase(TestSpin),
            TestLoader().loadTestsFromTestCase(TestSpinlessFermion),
            TestLoader().loadTestsFromTestCase(TestSpinfulFermion),
            TestLoader().loadTestsFromTestCase(TestHoneycombHeisenberg),
            ])
