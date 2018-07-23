'''
MPO test (12 tests in total).
'''

__all__=['mpo']

import numpy as np
import HamiltonianPy.Misc as hm
from numpy.linalg import norm
from HamiltonianPy import *
from HamiltonianPy.TensorNetwork import *
from unittest import TestCase,TestLoader,TestSuite

class TestOptStr(TestCase):
    def setUp(self):
        self.init()
        self.overlaps=[hm.overlap(self.mps1.state,mopt,self.mps2.state) for mopt in self.mopts]
        self.optstrs=[OptStr.fromoperator(opt,self.degfres,self.degfres.layers[-1]) for opt in self.generator.operators.itervalues()]

    def init(self):
        raise NotImplementedError()

    def test_overlap(self):
        for optstr,overlap in zip(self.optstrs,self.overlaps):
            self.assertAlmostEqual(norm(overlap-optstr.overlap(self.mps1,self.mps2)),0.0)

    def test_algebra(self):
        for optstr,overlap in zip(self.optstrs,self.overlaps):
            self.assertAlmostEqual(norm(2.0*overlap-(optstr*2.0).overlap(self.mps1,self.mps2)),0.0)
            self.assertAlmostEqual(norm(2.0*overlap-(2.0*optstr).overlap(self.mps1,self.mps2)),0.0)
            self.assertAlmostEqual(norm(overlap/2.0-(optstr/2.0).overlap(self.mps1,self.mps2)),0.0)

    def test_relayer(self):
        nmps1=self.mps1.relayer(self.degfres,self.degfres.layers[0])
        nmps2=self.mps2.relayer(self.degfres,self.degfres.layers[0])
        for optstr,overlap in zip(self.optstrs,self.overlaps):
            self.assertAlmostEqual(norm(overlap-optstr.relayer(self.degfres,self.degfres.layers[0]).overlap(nmps1,nmps2)),0.0)

class TestMPO(TestCase):
    def setUp(self):
        self.init()
        self.mopt=sum(self.mopts)
        self.overlap=hm.overlap(self.mps1.state,self.mopt,self.mps2.state)
        optstrs=[OptStr.fromoperator(opt,self.degfres,self.degfres.layers[-1]) for opt in self.generator.operators.itervalues()]
        self.mpo=OptMPO(optstrs,self.degfres).tompo()

    def init(self):
        raise NotImplementedError()

    def test_overlap(self):
        self.assertAlmostEqual(norm(self.overlap-self.mpo.overlap(self.mps1,self.mps2)),0.0)

    def test_algebra(self):
        self.assertAlmostEqual(norm(self.overlap*2.0-(self.mpo*2.0).overlap(self.mps1,self.mps2)),0.0)
        self.assertAlmostEqual(norm(self.overlap*2.0-(2.0*self.mpo).overlap(self.mps1,self.mps2)),0.0)
        self.assertAlmostEqual(norm(self.overlap/2.0-(self.mpo/2.0).overlap(self.mps1,self.mps2)),0.0)
        self.assertAlmostEqual(norm(self.overlap-hm.overlap(self.mps1.state,(self.mpo*self.mps2).state)),0.0)
        self.assertAlmostEqual(norm(hm.overlap(self.mps1.state,self.mopt.dot(self.mopt),self.mps2.state)-(self.mpo*self.mpo).overlap(self.mps1,self.mps2)),0.0)
        another,overlap=self.mpo/2.0,self.overlap/2.0
        self.assertAlmostEqual(norm((self.overlap+overlap)-(self.mpo+another).overlap(self.mps1,self.mps2)),0.0)
        self.assertAlmostEqual(norm((self.overlap-overlap)-(self.mpo-another).overlap(self.mps1,self.mps2)),0.0)

    def test_relayer(self):
        nmps1=self.mps1.relayer(self.degfres,self.degfres.layers[0])
        nmps2=self.mps2.relayer(self.degfres,self.degfres.layers[0])
        nmpo0=self.mpo.relayer(self.degfres,self.degfres.layers[0])
        nmpo1=nmpo0.relayer(self.degfres,self.degfres.layers[1])
        self.assertAlmostEqual(norm(self.overlap-nmpo0.overlap(nmps1,nmps2)),0.0)
        self.assertAlmostEqual(norm(self.overlap-nmpo1.overlap(self.mps1,self.mps2)),0.0)

class SpinBase(object):
    def init(self):
        S,priority,layers=0.5,['scope','site','orbital','S'],[('scope',),('site','orbital','S')]
        self.lattice=Cylinder(name='WG',block=[np.array([0.0,0.0]),np.array([0.0,1.0])],translation=np.array([1.0,0.0]))([0,1])
        self.terms=[SpinTerm('J',1.0,neighbour=1,indexpacks=Heisenberg())]
        self.config=IDFConfig(priority=priority,pids=self.lattice.pids,map=lambda pid: Spin(S=S))
        self.degfres=DegFreTree(mode='QN',layers=layers,priority=priority,leaves=self.config.table(mask=[]).keys(),map=lambda index:SQNS(S))
        self.generator=Generator(self.lattice.bonds,self.config,terms=self.terms,dtype=np.float64)
        self.mopts=[soptrep(opt,self.config.table(mask=[])) for opt in self.generator.operators.itervalues()]
        np.random.seed()
        sites,bonds=self.degfres.labels(mode='S',layer=layers[-1]),self.degfres.labels(mode='B',layer=layers[-1])
        bonds[+0]=Label(bonds[+0],qns=QuantumNumbers.mono(SQN(0.0)),flow=None)
        bonds[-1]=Label(bonds[-1],qns=QuantumNumbers.mono(SQN(0.0)),flow=None)
        cut=np.random.randint(0,len(self.lattice)+1)
        self.mps1=MPS.random(sites,bonds,cut=cut,nmax=20)
        self.mps2=MPS.random(sites,bonds,cut=cut,nmax=20)

class TestSpinOptStr(TestOptStr,SpinBase):
    init=SpinBase.init

class TestSpinMPO(TestMPO,SpinBase):
    init=SpinBase.init

class FermiBase(object):
    def init(self):
        priority,layers=['scope','site','orbital','spin','nambu'],[('scope',),('site','orbital','spin')]
        self.lattice=Cylinder(name='WG',block=[np.array([0.0,0.0]),np.array([0.0,1.0])],translation=np.array([1.0,0.0]))([0,1])
        self.terms=[Hopping('t',1.0,neighbour=1)]
        self.config=IDFConfig(priority=priority,pids=self.lattice.pids,map=lambda pid: Fock(norbital=1,nspin=1,nnambu=1))
        self.degfres=DegFreTree(mode='QN',layers=layers,priority=priority,leaves=self.config.table(mask=['nambu']).keys(),map=lambda index:PQNS(1))
        self.generator=Generator(self.lattice.bonds,self.config,terms=self.terms,dtype=np.complex128)
        self.mopts=[soptrep(JWBosonization(opt,self.config.table(mask=['nambu'])),self.config.table(mask=['nambu'])) for opt in self.generator.operators.itervalues()]
        np.random.seed()
        sites,bonds=self.degfres.labels(mode='S',layer=layers[-1]),self.degfres.labels(mode='B',layer=layers[-1])
        bonds[+0]=Label(bonds[+0],qns=QuantumNumbers.mono(PQN(0)),flow=None)
        bonds[-1]=Label(bonds[-1],qns=QuantumNumbers.mono(PQN(len(self.lattice)/2)),flow=None)
        cut=np.random.randint(0,len(self.lattice)+1)
        self.mps1=MPS.random(sites,bonds,cut=cut,nmax=20)
        self.mps2=MPS.random(sites,bonds,cut=cut,nmax=20)

class TestFermiOptStr(TestOptStr,FermiBase):
    init=FermiBase.init

class TestFermiMPO(TestMPO,FermiBase):
    init=FermiBase.init

mpo=TestSuite([
            TestLoader().loadTestsFromTestCase(TestSpinOptStr),
            TestLoader().loadTestsFromTestCase(TestSpinMPO),
            TestLoader().loadTestsFromTestCase(TestFermiOptStr),
            TestLoader().loadTestsFromTestCase(TestFermiMPO),
            ])
