'''
QuantumNumber test.
'''

__all__=['test_quantum_number']

import numpy as np
import HamiltonianPy.Misc as hm
from HamiltonianPy.Basics.QuantumNumber import *
from numpy.linalg import norm
from copy import copy,deepcopy
from time import time

def test_quantum_number():
    test_quantumnumber()
    test_quantumnumbers_ordinary()
    test_quantumnumbers_time()
    test_quantumnumbers_kron()
    test_quantumnumbers_decomposition()

def test_quantumnumber():
    print 'test_quantumnumber'
    a,b=SPQN((2,-2)),SPQN((1,-1))
    print 'a: %s'%(a,)
    print 'b: %s'%(b,)
    print 'a+b: %s'%(a+b,)
    print

def test_quantumnumbers_ordinary():
    print 'test_quantumnumbers_ordinary'
    a=QuantumNumbers('C',([SQN(-1.0),SQN(0.0),SQN(1.0)],[1,1,1]),protocol=QuantumNumbers.COUNTS)
    print 'a: %s'%a
    print 'copy(a): %s'%copy(a)
    print 'deepcopy(a): %s'%deepcopy(a)

    b,permutation=QuantumNumbers.kron([a]*2).sort(history=True)
    print 'b: ',b
    print 'permutation of b:%s'%permutation
    print 'b.reorder(permutation,protocol="EXPANSION"): \n%s'%b.reorder(permutation,protocol="EXPANSION")
    print 'b.reorder([4,3,2,1,0],protocol="CONTENTS"): \n%s'%b.reorder([4,3,2,1,0],protocol="CONTENTS")

    c=b.to_ordereddict(protocol=QuantumNumbers.COUNTS)
    print 'c(b.to_ordereddict(protocol=QuantumNumbers.COUNTS)):\n%s'%('\n'.join('%s: %s'%(key,value) for key,value in c.iteritems()))
    print 'QuantumNumbers.from_ordereddict(SQN,c,protocol=QuantumNumbers.COUNTS):\n%s'%QuantumNumbers.from_ordereddict(SQN,c,protocol=QuantumNumbers.COUNTS)

    d=b.to_ordereddict(protocol=QuantumNumbers.INDPTR)
    print 'd(b.to_ordereddict(protocol=QuantumNumbers.INDPTR)):\n%s'%('\n'.join('%s: %s'%(key,value)for key,value in d.iteritems()))
    print 'QuantumNumbers.from_ordereddict(SQN,d,protocol=QuantumNumbers.INDPTR):\n%s'%QuantumNumbers.from_ordereddict(SQN,d,protocol=QuantumNumbers.INDPTR)
    print

def test_quantumnumbers_time():
    print 'test_quantumnumbers_time'
    N=6
    t1=time()
    b=QuantumNumbers.kron([SQNS(1.0)]*N).sort()
    t2=time()
    print 'Summation form 1 to %s: %ss.'%(N,t2-t1)
    t3=time()
    QuantumNumbers.kron([b,b]).sort(history=True)
    t4=time()
    print 'Summation of %s and %s: %ss.'%(N,N,t4-t3)
    print

def test_quantumnumbers_kron():
    print 'test_quantumnumbers_kron'
    N,S=2,0.5
    a,b,c,d=np.random.random((N,N)),np.random.random((N,N)),np.random.random((N,N)),np.random.random((N,N))

    s211=QuantumNumbers.kron([SQNS(S)]*2,signs='+-')
    s422,p422=QuantumNumbers.kron([s211,s211],signs='++').sort(history=True)
    tmp1,tmp2=np.kron(a,b),np.kron(c,d)
    m1=hm.reorder(np.kron(tmp1,tmp2),permutation=p422)
    print 'p422: %s'%p422

    s41111,p41111=QuantumNumbers.kron([SQNS(S)]*4,signs='+-+-').sort(history=True)
    m2=hm.reorder(np.kron(np.kron(np.kron(a,b),c),d),permutation=p41111)
    print 'p41111: %s'%p41111

    print 'diff: %s.'%norm(m1-m2)
    print

def test_quantumnumbers_decomposition():
    print 'test_quantumnumbers_decomposition'
    qnses,signs=[SQNS(0.5)]*4,'+-+-'
    print 'Exhaustion method:'
    for index in QuantumNumbers.decomposition(qnses,signs=signs,target=SQN(0.0),method='exhaustion',nmax=None):
        print index
    print
    print 'Monte carlo method:'
    qnses,signs=[SQNS(0.5)]*40,None
    for index in QuantumNumbers.decomposition(qnses,signs=signs,target=SQN(1.0),method='monte carlo',nmax=10):
        print index
        assert index.count(1)==21
    print
    print
