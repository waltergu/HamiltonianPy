'''
MPS test.
'''

__all__=['test_mps']

import numpy as np
from HamiltonianPy import *
from numpy.linalg import norm
from HamiltonianPy.TensorNetwork import Label,Tensor,MPS,DegFreTree
from copy import copy,deepcopy

def test_mps():
    print 'test_mps'
    test_mps_ordinary()
    test_mps_algebra()
    test_mps_relayer()

def test_mps_ordinary():
    'test_mps_ordinary'
    N=4
    np.random.seed()
    state,target=np.zeros((2,)*N),SQN(0.0)
    for index in QuantumNumbers.decomposition([SQNS(0.5)]*N,signs='+'*N,target=target):
        state[index]=np.random.random()
    state=state.reshape((-1,))
    sites=[Label('S%s'%i,qns=SQNS(0.5)) for i in xrange(N)]
    bonds=[Label('B%s'%i,qns=SQNS(0.0) if i==0 else (QuantumNumbers.mono(target) if i==N else None)) for i in xrange(N+1)]
    for cut in xrange(N+1):
        mps=MPS.from_state(state,sites,bonds,cut=cut)
        print 'mps.cut,mps.is_canonical,diff: %s, %s, %s.'%(mps.cut,mps.is_canonical(),norm(state-mps.state))
    for cut in xrange(N+1):
        mps.canonicalization(cut)
        print 'mps.cut, mps.is_canonical: %s, %s.'%(mps.cut,mps.is_canonical())
    print

def test_mps_algebra():
    print 'test_mps_algebra'
    N=8
    np.random.seed()
    target=SQN(0.0)
    sites=[Label('S%s'%i,qns=SQNS(0.5)) for i in xrange(N)]
    bonds=[Label('B%s'%i,qns=SQNS(0.0) if i==0 else (QuantumNumbers.mono(target) if i==N else None)) for i in xrange(N+1)]
    cut=np.random.randint(0,N+1)
    print 'cut: %s'%cut
    mps1=MPS.random(sites,bonds,cut=cut)
    mps2=MPS.random(sites,bonds,cut=cut)
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
    priority,layers=['scope','site','S'],[('scope',),('site','S')]
    config=IDFConfig(priority=priority)
    for scope in xrange(Nscope):
        for site in xrange(Nsite):
            config[PID(scope=scope,site=site)]=Spin(S=S)
    leaves=config.table(mask=[]).keys()
    tree=DegFreTree(mode='QN',layers=layers,priority=priority,leaves=leaves,map=lambda index: SQNS(S))
    sites=tree.labels(layer=layers[-1],mode='S')
    bonds=tree.labels(layer=layers[-1],mode='B')
    bonds[+0]=bonds[+0].replace(qns=QuantumNumbers.mono(SQN(0.0)))
    bonds[-1]=bonds[-1].replace(qns=QuantumNumbers.mono(SQN(0.0)))
    mps0=MPS.random(sites,bonds,cut=0)
    mps1=mps0.relayer(tree,layers[0]).relayer(tree,layers[-1])
    print 'relayer diff: %s'%norm(mps1.state-mps0.state)
    print
