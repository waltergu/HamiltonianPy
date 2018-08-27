'''
QuantumNumber test (9 tests in total).
'''

__all__=['quantumnumber']

import numpy as np
import numpy.linalg as nl
import HamiltonianPy.Misc as hm
from HamiltonianPy.Basics.QuantumNumber import *
from collections import OrderedDict
from unittest import TestCase,TestLoader,TestSuite
from time import time

class TestQuantumNumber(TestCase):
    def setUp(self):
        self.a=SPQN((2,-2))
        self.b=SPQN((1,-1))

    def test_add(self):
        result=SPQN((3,-3))
        self.assertEqual(self.a+self.b,result)

    def test_subtraction(self):
        result=SPQN((1,-1))
        self.assertEqual(self.a-self.b,result)

class TestQuantumNumbers(TestCase):
    def test_kron(self):
        qns,permutation=QuantumNumbers.kron([SQNS(1.0)]*2).sorted(history=True)
        rqns=QuantumNumbers('C',([SQN(-2.0),SQN(-1.0),SQN(0.0),SQN(1.0),SQN(2.0)],[1,2,3,2,1]),protocol=QuantumNumbers.COUNTS)
        rpermutation=np.array([0,1,3,2,4,6,5,7,8])
        self.assertEqual(qns,rqns)
        self.assertEqual(nl.norm(permutation-rpermutation),0.0)

    def test_associativity(self):
        N,S=2,0.5
        a,b,c,d=np.random.random((N,N)),np.random.random((N,N)),np.random.random((N,N)),np.random.random((N,N))
        p4_2_2=QuantumNumbers.kron([QuantumNumbers.kron([SQNS(S)]*2,signs=(+1,-1))]*2,signs=(+1,+1)).sorted(history=True)[1]
        p41111=QuantumNumbers.kron([SQNS(S)]*4,signs=(+1,-1,+1,-1)).sorted(history=True)[1]
        tmp1,tmp2=np.kron(a,b),np.kron(c,d)
        m1=hm.reorder(np.kron(tmp1,tmp2),permutation=p4_2_2)
        m2=hm.reorder(np.kron(np.kron(np.kron(a,b),c),d),permutation=p41111)
        self.assertAlmostEqual(nl.norm(m2-m1),0.0,delta=10**-14)

    def test_reorder(self):
        qns=QuantumNumbers('C',([SQN(-1.0),SQN(0.0),SQN(1.0)],[1,2,1]),protocol=QuantumNumbers.COUNTS)
        p1=np.array([0,1,3,2])
        r1=QuantumNumbers('C',([SQN(-1.0),SQN(0.0),SQN(1.0),SQN(0.0)],[1,1,1,1]),protocol=QuantumNumbers.COUNTS)
        self.assertEqual(qns.reorder(p1,protocol="EXPANSION"),r1)
        p2=np.array([0,2,1])
        r2=QuantumNumbers('C',([SQN(-1.0),SQN(1.0),SQN(0.0)],[1,1,2]),protocol=QuantumNumbers.COUNTS)
        self.assertEqual(qns.reorder(p2,protocol="CONTENTS"),r2)

    def test_toordereddict(self):
        qns=SQNS(1.0)
        r1=OrderedDict([(SQN(-1.0),slice(0,1)),(SQN(0.0),slice(1,2)),(SQN(1.0),slice(2,3))])
        self.assertEqual(qns.toordereddict(QuantumNumbers.INDPTR),r1)
        r2=OrderedDict([(SQN(-1.0),1),(SQN(0.0),1),(SQN(1.0),1)])
        self.assertEqual(qns.toordereddict(QuantumNumbers.COUNTS),r2)

    def test_fromordereddict(self):
        od1=OrderedDict([(SQN(-1.0),slice(0,1)),(SQN(0.0),slice(1,2)),(SQN(1.0),slice(2,3))])
        od2=OrderedDict([(SQN(-1.0),1),(SQN(0.0),1),(SQN(1.0),1)])
        result=QuantumNumbers('U',([SQN(-1.0),SQN(0.0),SQN(1.0)],[1,1,1]),protocol=QuantumNumbers.COUNTS)
        self.assertEqual(QuantumNumbers.fromordereddict(od1,protocol=QuantumNumbers.INDPTR),result)
        self.assertEqual(QuantumNumbers.fromordereddict(od2,protocol=QuantumNumbers.COUNTS),result)

    def test_time(self):
        print()
        N=6
        stime=time()
        qns=QuantumNumbers.kron([SQNS(1.0)]*N).sorted()
        etime=time()
        print('Summation form 1 to %s: %ss.'%(N,etime-stime))
        stime=time()
        QuantumNumbers.kron([qns,qns]).sorted(history=True)
        etime=time()
        print('Summation of %s and %s: %ss.'%(N,N,etime-stime))

    def test_decomposition(self):
        qnses,signs,indices=[SQNS(0.5)]*4,(+1,-1,+1,-1),[(0,0,0,0),(0,0,1,1),(0,1,1,0),(1,0,0,1),(1,1,0,0),(1,1,1,1)]
        for component,index in zip(sorted(QuantumNumbers.decomposition(qnses,signs=signs,target=SQN(0.0),method='exhaustion')),indices):
            self.assertEqual(component,index)
        qnses,signs=[SQNS(0.5)]*40,None
        for index in QuantumNumbers.decomposition(qnses,signs=signs,target=SQN(1.0),method='monte carlo',nmax=10):
            self.assertEqual(index.count(1),21)

quantumnumber=TestSuite([
                    TestLoader().loadTestsFromTestCase(TestQuantumNumber),
                    TestLoader().loadTestsFromTestCase(TestQuantumNumbers),
                    ])
