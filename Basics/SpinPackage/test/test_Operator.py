'''
SOperator test (3 tests in total).
'''

__all__=['soperator']

import numpy as np
from HamiltonianPy.Basics.Geometry import *
from HamiltonianPy.Basics.DegreeOfFreedom import *
from HamiltonianPy.Basics.Operator import *
from HamiltonianPy.Basics.SpinPackage import *
from unittest import TestCase,TestLoader,TestSuite

class TestOperator(TestCase):
    def setUp(self):
        self.operator=SOperator(
                    value=      0.5,
                    indices=    [Index(pid=PID('WG',0),iid=SID(0,0.5)),Index(pid=PID('WG',1),iid=SID(0,0.5))],
                    spins=      [SpinMatrix(0.5,'p',dtype=np.float64),SpinMatrix(0.5,'m',dtype=np.float64)],
                    seqs=       (0,1)
                )

    def test_id(self):
        self.assertEqual(self.operator.id,(Index(PID('WG',0),SID(0,0.5)),Index(PID('WG',1),SID(0,0.5)),'p','m'))

    def test_rank(self):
        self.assertEqual(self.operator.rank,2)

class TestOperators(TestCase):
    def setUp(self):
        self.a=SOperator(
                    value=      0.5,
                    indices=    [Index(pid=PID('WG',0),iid=SID(0,0.5)),Index(pid=PID('WG',1),iid=SID(0,0.5))],
                    spins=      [SpinMatrix(0.5,'p',dtype=np.float64),SpinMatrix(0.5,'m',dtype=np.float64)],
                    seqs=       (0,1)
                )
        self.b=SOperator(
                    value=      0.5,
                    indices=    [Index(pid=PID('WG',0),iid=SID(0,0.5)),Index(pid=PID('WG',1),iid=SID(0,0.5))],
                    spins=      [SpinMatrix(0.5,'m',dtype=np.float64),SpinMatrix(0.5,'p',dtype=np.float64)],
                    seqs=       (0,1)
                )
        self.c=SOperator(
                    value=      1.0,
                    indices=    [Index(pid=PID('WG',0),iid=SID(0,0.5)),Index(pid=PID('WG',1),iid=SID(0,0.5))],
                    spins=      [SpinMatrix(0.5,'z',dtype=np.float64),SpinMatrix(0.5,'z',dtype=np.float64)],
                    seqs=       (0,1)
                )

    def test_algebra(self):
        opts=Operators()
        opts+=self.a*2
        opts+=self.b*2
        opts+=self.c*3
        self.assertEqual(len(opts),3)
        self.assertEqual(opts[self.a.id],self.a*2)
        self.assertEqual(opts[self.b.id],self.b*2)
        self.assertEqual(opts[self.c.id],self.c*3)

soperator=TestSuite([
                TestLoader().loadTestsFromTestCase(TestOperator),
                TestLoader().loadTestsFromTestCase(TestOperators),
                ])

