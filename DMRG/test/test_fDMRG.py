'''
fDMRG test.
'''

__all__=['fdmrg']

import mkl
import numpy as np
from HamiltonianPy import *
from HamiltonianPy.TensorNetwork import *
from HamiltonianPy.DMRG import *
from unittest import TestCase,TestLoader,TestSuite

mkl.set_num_threads(1)
Engine.DEBUG=True
ttype,savedata,qnon='S',True,True

class TestSpin(TestCase):
    def test_fdmrg(self):
        print()
        J,spin,N=1.0,0.5,30
        priority,layers=DEFAULT_SPIN_PRIORITY,DEGFRE_SPIN_LAYERS
        target=lambda niter: SQN(0.0) if qnon else None
        dmrg=fDMRG(
                name=       'spin-%s'%spin,
                tsg=        TSG(name='GROWTH',target=target,maxiter=N//2,nmax=200,savedata=savedata,plot=False,run=fDMRGTSG),
                tss=        TSS(name='SWEEP',target=target(N//2-1),nsite=N,nmaxs=[200,200],savedata=savedata,run=fDMRGTSS),
                lattice=    Cylinder(name='WG',block=[np.array([0.0,0.0])],translation=np.array([1.0,0.0]),neighbours=1),
                terms=      [SpinTerm('J',J,neighbour=1,indexpacks=Heisenberg())],
                config=     IDFConfig(priority=priority,map=lambda pid: Spin(S=spin)),
                degfres=    DegFreTree(layers=layers,map=lambda index: SQNS(S=spin) if qnon else int(spin*2+1)),
                mask=       [],
                ttype=      ttype,
                dtype=      np.float64
                )
        dmrg.run('SWEEP')
        dmrg.summary()

class TestSpinlessFermion(TestCase):
    def test_fdmrg(self):
        print()
        t,N=-0.5,30
        priority,layers=DEFAULT_FERMIONIC_PRIORITY,DEGFRE_FERMIONIC_LAYERS
        target=lambda niter: PQN(niter+1) if qnon else None
        dmrg=fDMRG(
                name=       'fermion-spinless',
                tsg=        TSG(name='GROWTH',target=target,maxiter=N//2,nmax=200,savedata=savedata,plot=False,run=fDMRGTSG),
                tss=        TSS(name='SWEEP',target=target(N//2-1),nsite=N,nmaxs=[200,200],savedata=savedata,run=fDMRGTSS),
                lattice=    Cylinder(name='WG',block=[np.array([0.0,0.0])],translation=np.array([1.0,0.0])),
                terms=      [Hopping('t',t,neighbour=1)],
                config=     IDFConfig(priority=priority,map=lambda pid: Fock(atom=0,norbital=1,nspin=1,nnambu=1)),
                degfres=    DegFreTree(layers=layers,map=lambda index: PQNS(1) if qnon else 2),
                mask=       ['nambu'],
                ttype=      ttype,
                dtype=      np.float64
                )
        dmrg.run('SWEEP')
        dmrg.summary()

class TestSpinfulFermion(TestCase):
    def test_fdmrg(self):
        print()
        t,U,N=-1.0,1.0,20
        priority,layers=DEFAULT_FERMIONIC_PRIORITY,DEGFRE_FERMIONIC_LAYERS
        target=lambda niter: SPQN((2.0*(niter+1),0.0)) if qnon else None
        dmrg=fDMRG(
                name=       'fermion-spinful',
                tsg=        TSG(name='GROWTH',target=target,maxiter=N//2,nmax=200,savedata=savedata,plot=False,run=fDMRGTSG),
                tss=        TSS(name='SWEEP',target=target(N//2-1),nsite=N,nmaxs=[200,200],savedata=savedata,run=fDMRGTSS),
                lattice=    Cylinder(name='WG',block=[np.array([0.0,0.0])],translation=np.array([1.0,0.0])),
                terms=      [Hopping('t',t,neighbour=1),Hubbard('U',U)],
                config=     IDFConfig(priority=priority,map=lambda pid: Fock(atom=0,norbital=1,nspin=2,nnambu=1)),
                degfres=    DegFreTree(layers=layers,map=lambda index: SzPQNS(index.spin-0.5) if qnon else 2),
                mask=       ['nambu'],
                ttype=      ttype,
                dtype=      np.float64
                )
        dmrg.run('SWEEP')
        dmrg.summary()

class TestHoneycombHeisenberg(TestCase):
    def test_fdmrg(self):
        print()
        J,N=1.0,10
        h4,priority,layers=Hexagon(name='H4'),DEFAULT_SPIN_PRIORITY,DEGFRE_SPIN_LAYERS
        target=lambda niter: SQN(0.0) if qnon else None
        dmrg=fDMRG(
                name=       'honeycomb-heisenberg',
                tsg=        TSG(name='fGROWTH',target=target,maxiter=N//2,nmax=100,savedata=savedata,plot=False,run=fDMRGTSG),
                tss=        TSS(name='SWEEP',target=target(N//2-1),nsite=N*4,nmaxs=[100,100],savedata=savedata,run=fDMRGTSS),
                lattice=    h4.cylinder(0,'1O-1P',nneighbour=1),
                terms=      [SpinTerm('J',J,neighbour=1,indexpacks=Heisenberg())],
                config=     IDFConfig(priority=priority,map=lambda pid: Spin(S=0.5)),
                degfres=    DegFreTree(layers=layers,map=lambda index: SQNS(S=0.5) if qnon else 2),
                mask=       [],
                ttype=      ttype,
                dtype=      np.float64
                )
        dmrg.run('SWEEP')
        dmrg.summary()

fdmrg=TestSuite([
            TestLoader().loadTestsFromTestCase(TestSpin),
            TestLoader().loadTestsFromTestCase(TestSpinlessFermion),
            TestLoader().loadTestsFromTestCase(TestSpinfulFermion),
            TestLoader().loadTestsFromTestCase(TestHoneycombHeisenberg),
            ])
