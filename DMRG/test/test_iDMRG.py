'''
iDMRG test.
'''

__all__=['idmrg']

import mkl
import numpy as np
from HamiltonianPy import *
from HamiltonianPy.TensorNetwork import *
from HamiltonianPy.DMRG import *
from unittest import TestCase,TestLoader,TestSuite

mkl.set_num_threads(1)
Engine.DEBUG=True
savedata,qnon=True,True
mode='QN' if qnon else 'NB'

class TestSpin(TestCase):
    def test_idmrg(self):
        print
        J,spin,N=1.0,0.5,50
        priority,layers=DEGFRE_SPIN_PRIORITY,DEGFRE_SPIN_LAYERS
        self.dmrg=iDMRG(
                name=       'spin-%s(%s)'%(spin,mode),
                tsg=        TSG(name='ITER',target=lambda niter: SQN(0.0) if qnon else None,maxiter=N,nmax=200,savedata=savedata,run=iDMRGTSG),
                lattice=    Cylinder(name='WG',block=[np.array([0.0,0.0])],translation=np.array([1.0,0.0]),neighbours=1),
                terms=      [SpinTerm('J',J,neighbour=1,indexpacks=Heisenberg())],
                config=     IDFConfig(priority=priority,map=lambda pid: Spin(S=spin)),
                degfres=    DegFreTree(mode=mode,layers=layers,priority=priority,map=lambda index: SQNS(S=spin) if qnon else int(spin*2+1)),
                mask=       [],
                dtype=      np.float64
                )
        self.dmrg.run('ITER')
        self.dmrg.summary()

class TestSpinlessFermion(TestCase):
    def test_idmrg(self):
        print
        t,N=-0.5,50
        priority,layers=DEGFRE_FERMIONIC_PRIORITY,DEGFRE_FERMIONIC_LAYERS
        self.dmrg=iDMRG(
                name=       'fermion-spinless(%s)'%mode,
                tsg=        TSG(name='ITER',target=lambda niter: PQN(niter+1) if qnon else None,maxiter=N,nmax=200,savedata=savedata,run=iDMRGTSG),
                lattice=    Cylinder(name='WG',block=[np.array([0.0,0.0])],translation=np.array([1.0,0.0])),
                terms=      [Hopping('t',t,neighbour=1)],
                config=     IDFConfig(priority=priority,map=lambda pid: Fock(atom=0,norbital=1,nspin=1,nnambu=1)),
                degfres=    DegFreTree(mode=mode,layers=layers,priority=priority,map=lambda index: PQNS(1) if qnon else 2),
                mask=       ['nambu'],
                dtype=      np.float64
                )
        self.dmrg.run('ITER')
        self.dmrg.summary()

class TestSpinfulFermion(TestCase):
    def test_idmrg(self):
        print
        t,U,N=-1.0,1.0,50
        priority,layers=DEGFRE_FERMIONIC_PRIORITY,DEGFRE_FERMIONIC_LAYERS
        self.dmrg=iDMRG(
                name=       'fermion-spinful(%s)'%mode,
                tsg=        TSG(name='ITER',target=lambda niter: SPQN((2.0*(niter+1),0.0)) if qnon else None,maxiter=N,nmax=200,savedata=savedata,run=iDMRGTSG),
                lattice=    Cylinder(name='WG',block=[np.array([0.0,0.0])],translation=np.array([1.0,0.0])),
                terms=      [Hopping('t',t,neighbour=1),Hubbard('U',U)],
                config=     IDFConfig(priority=priority,map=lambda pid: Fock(atom=0,norbital=1,nspin=2,nnambu=1)),
                degfres=    DegFreTree(mode=mode,layers=layers,priority=priority,map=lambda index: SzPQNS(index.spin-0.5) if qnon else 2),
                mask=       ['nambu'],
                dtype=      np.float64
                )
        self.dmrg.run('ITER')
        self.dmrg.summary()

class TestHoneycombHeisenberg(TestCase):
    def test_idmrg(self):
        print
        J,N=1.0,10
        h4,priority,layers=Hexagon(name='H4'),DEGFRE_SPIN_PRIORITY,DEGFRE_SPIN_LAYERS
        self.dmrg=iDMRG(
                name=       'honeycomb-heisenberg(%s)'%mode,
                tsg=        TSG(name='ITER',target=lambda niter: SQN(0.0) if qnon else None,maxiter=N,nmax=100,savedata=savedata,run=iDMRGTSG),
                lattice=    h4.cylinder(0,'1O-1P',nneighbour=1),
                terms=      [SpinTerm('J',J,neighbour=1,indexpacks=Heisenberg())],
                config=     IDFConfig(priority=priority,map=lambda pid: Spin(S=0.5)),
                degfres=    DegFreTree(mode=mode,layers=layers,priority=priority,map=lambda index: SQNS(S=0.5) if qnon else 2),
                mask=       [],
                dtype=      np.float64
                )
        self.dmrg.run('ITER')
        self.dmrg.summary()

idmrg=TestSuite([
            TestLoader().loadTestsFromTestCase(TestSpin),
            TestLoader().loadTestsFromTestCase(TestSpinlessFermion),
            TestLoader().loadTestsFromTestCase(TestSpinfulFermion),
            TestLoader().loadTestsFromTestCase(TestHoneycombHeisenberg),
            ])
