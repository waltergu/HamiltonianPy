'''
MPS test (4 tests in total).
'''

__all__=['mps']

import numpy as np
from HamiltonianPy import *
from numpy.linalg import norm
from HamiltonianPy.TensorNetwork import *
from unittest import TestCase,TestLoader,TestSuite

class TestMPS(TestCase):
    def test_ordinary(self):
        N=4
        np.random.seed()
        state,target=np.zeros((2,)*N),SQN(0.0)
        for index in QuantumNumbers.decomposition([SQNS(0.5)]*N,signs=[1]*N,target=target):
            state[index]=np.random.random()
        state=state.reshape((-1,))
        sites=[Label('S%s'%i,qns=SQNS(0.5),flow=1) for i in xrange(N)]
        bonds=[Label('B%s'%i,qns=None,flow=None) for i in xrange(N+1)]
        bonds[+0]=bonds[+0].replace(qns=SQNS(0.0),flow=+1)
        bonds[-1]=bonds[-1].replace(qns=QuantumNumbers.mono(target),flow=-1)
        for cut in xrange(N+1):
            mps=MPS.fromstate(state,sites,bonds,mode='D',cut=cut)
            self.assertTrue(all(mps.iscanonical()))
            self.assertAlmostEqual(norm(state-mps.state),0.0)
        for cut in xrange(N+1):
            mps.canonicalize(cut)
            self.assertTrue(all(mps.iscanonical()))
        for cut in xrange(N+1):
            mps=MPS.fromstate(state,sites,bonds,mode='S',cut=cut)
            self.assertTrue(all(mps.iscanonical()))
            self.assertAlmostEqual(norm(state-mps.state),0.0)
        for cut in xrange(N+1):
            mps.canonicalize(cut)
            self.assertTrue(all(mps.iscanonical()))

    def test_random(self):
        N=20
        np.random.seed()
        sites=[SQNS(0.5) for _ in xrange(N)]
        bonds=[SQN(0.0),SQN(0.0)]
        mps=MPS.random(sites,bonds,mode='D',cut=np.random.randint(0,N+1),nmax=20)
        self.assertTrue(all(mps.iscanonical()))
        mps=MPS.random(sites,bonds,mode='S',cut=np.random.randint(0,N+1),nmax=20)
        self.assertTrue(all(mps.iscanonical()))

    def test_algebra(self):
        N=8
        np.random.seed()
        sites=[SQNS(0.5) for _ in xrange(N)]
        bonds=[SQN(0.0),SQN(0.0)]
        cut=np.random.randint(0,N+1)
        mps1=MPS.random(sites,bonds,mode='D',cut=cut,nmax=10)
        mps2=MPS.random(sites,bonds,mode='D',cut=cut,nmax=10)
        self.assertAlmostEqual(norm((mps1+mps2).state-(mps1.state+mps2.state)),0.0)
        self.assertAlmostEqual(norm((mps1-mps2).state-(mps1.state-mps2.state)),0.0)
        self.assertAlmostEqual(norm((mps1*2.0).state-mps1.state*2.0),0.0)
        self.assertAlmostEqual(norm((2.0*mps1).state-2.0*mps1.state),0.0)
        self.assertAlmostEqual(norm((mps1/2.0).state-mps1.state/2.0),0.0)
        self.assertAlmostEqual(MPS.overlap(mps1,mps2)-np.dot(mps1.state,mps2.state),0.0)
        mps1=MPS.random(sites,bonds,mode='S',cut=cut,nmax=10)
        mps2=MPS.random(sites,bonds,mode='S',cut=cut,nmax=10)
        self.assertAlmostEqual(norm((mps1+mps2).state-(mps1.state+mps2.state)),0.0)
        self.assertAlmostEqual(norm((mps1-mps2).state-(mps1.state-mps2.state)),0.0)
        self.assertAlmostEqual(norm((mps1*2.0).state-mps1.state*2.0),0.0)
        self.assertAlmostEqual(norm((2.0*mps1).state-2.0*mps1.state),0.0)
        self.assertAlmostEqual(norm((mps1/2.0).state-mps1.state/2.0),0.0)
        self.assertAlmostEqual(MPS.overlap(mps1,mps2)-np.dot(mps1.state,mps2.state),0.0)

    def test_relayer(self):
        Nsite,Nscope,S=2,2,1.0
        priority,layers=['scope','site','orbital','S'],[('scope',),('site','orbital','S')]
        config=IDFConfig(priority=priority,pids=[PID(scope,site) for scope in xrange(Nscope) for site in xrange(Nsite)],map=lambda pid: Spin(S=S))
        tree=DegFreTree(mode='QN',layers=layers,priority=priority,leaves=config.table(mask=[]).keys(),map=lambda index: SQNS(S))
        sites=tree.labels(mode='S',layer=layers[-1])
        bonds=tree.labels(mode='B',layer=layers[-1])
        bonds[+0]=Label(bonds[+0],QuantumNumbers.mono(SQN(0.0)),None)
        bonds[-1]=Label(bonds[-1],QuantumNumbers.mono(SQN(0.0)),None)
        mps0=MPS.random(sites,bonds,cut=0,nmax=10)
        mps1=mps0.relayer(tree,layers[0])
        mps2=mps1.relayer(tree,layers[1])
        self.assertAlmostEqual(norm(mps1.state-mps0.state),0.0)
        self.assertAlmostEqual(norm(mps2.state-mps1.state),0.0)

mps=TestSuite([
            TestLoader().loadTestsFromTestCase(TestMPS),
            ])
