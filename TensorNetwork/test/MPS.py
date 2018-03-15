'''
MPS test.
'''

__all__=['test_mps']

import numpy as np
from HamiltonianPy import *
from numpy.linalg import norm
from HamiltonianPy.TensorNetwork import *

def test_mps():
    print 'test_mps'
    test_mps_ordinary()
    test_mps_random()
    test_mps_algebra()
    test_mps_relayer()

def test_mps_ordinary():
    print 'test_mps_ordinary'
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
        mps=MPS.from_state(state,sites,bonds,cut=cut)
        print 'mps.cut,mps.is_canonical,diff: %s, %s, %s.'%(mps.cut,mps.is_canonical(),norm(state-mps.state))
    for cut in xrange(N+1):
        mps.canonicalize(cut)
        print 'mps.cut, mps.is_canonical: %s, %s.'%(mps.cut,mps.is_canonical())
    print

def test_mps_random():
    print 'test_mps_random'
    N=20
    np.random.seed()
    sites=[SQNS(0.5) for i in xrange(N)]
    bonds=[SQN(0.0),SQN(0.0)]
    mps=MPS.random(sites,bonds,cut=np.random.randint(0,N+1),nmax=20)
    print 'mps.bonds:\n%s'%'\n'.join(repr(bond) for bond in mps.bonds)
    print 'mps.cut, mps.is_canonical: %s, %s.'%(mps.cut,mps.is_canonical())
    print

def test_mps_algebra():
    print 'test_mps_algebra'
    N=8
    np.random.seed()
    sites=[SQNS(0.5) for i in xrange(N)]
    bonds=[SQN(0.0),SQN(0.0)]
    cut=np.random.randint(0,N+1)
    print 'cut: %s'%cut
    mps1=MPS.random(sites,bonds,cut=cut,nmax=10)
    mps2=MPS.random(sites,bonds,cut=cut,nmax=10)
    print 'Addition diff: %s.'%(norm((mps1+mps2).state-(mps1.state+mps2.state)))
    print 'Subtraction diff: %s.'%(norm((mps1-mps2).state-(mps1.state-mps2.state)))
    print 'Left multiplication diff: %s.'%(norm((mps1*2.0).state-mps1.state*2.0))
    print 'Right multiplication diff: %s.'%(norm((mps1*2.0).state-mps1.state*2.0))
    print 'Division diff: %s.'%(norm((mps1/2.0).state-mps1.state/2.0))
    print 'overlap diff: %s.'%(MPS.overlap(mps1,mps2)-np.dot(mps1.state,mps2.state))
    print

def test_mps_relayer():
    print 'test_mps_relayer'
    Nsite,Nscope,S=2,2,1.0
    priority,layers=['scope','site','orbital','S'],[('scope',),('site','orbital','S')]
    config=IDFConfig(priority=priority)
    for scope in xrange(Nscope):
        for site in xrange(Nsite):
            config[PID(scope=scope,site=site)]=Spin(S=S)
    leaves=config.table(mask=[]).keys()
    tree=DegFreTree(mode='QN',layers=layers,priority=priority,leaves=leaves,map=lambda index: SQNS(S))
    sites=tree.labels(mode='S',layer=layers[-1])
    bonds=tree.labels(mode='B',layer=layers[-1])
    bonds[+0]=bonds[+0].replace(qns=QuantumNumbers.mono(SQN(0.0)))
    bonds[-1]=bonds[-1].replace(qns=QuantumNumbers.mono(SQN(0.0)))
    mps0=MPS.random(sites,bonds,cut=0,nmax=10)
    mps1=mps0.relayer(tree,layers[0])
    print 'relayer(1->0) diff: %s'%norm(mps1.state-mps0.state)
    mps2=mps1.relayer(tree,layers[1])
    print 'relayer(0->1) diff: %s'%norm(mps2.state-mps1.state)
    print
