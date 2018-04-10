'''
Utilities test (3 tests in total).
'''

__all__=['utilities']

from HamiltonianPy.Basics.Utilities import *
from unittest import TestCase,TestLoader,TestSuite
from time import sleep
import numpy as np

class TestTimers(TestCase):
    def setUp(self):
        self.nrecord=4
        self.keys=['Preparation','Diagonalization','Truncation']
        self.timers=Timers(*self.keys)

    def tearDown(self):
        self.timers.close()

    def test_timers(self):
        print
        np.random.seed()
        for _ in xrange(self.nrecord):
            for key in self.keys:
                with self.timers.get(key): sleep(np.random.random())
            self.timers.record()
            print '%s\n'%self.timers
            self.timers.graph()

class TestSheet(TestCase):
    def setUp(self):
        self.info=Sheet(rows=('nnz','gse','overlap','nbasis'),cols=('value',))
        self.info['nnz']=10
        self.info['gse']=-0.12345667
        self.info['overlap']=0.99999899
        self.info['nbasis']=200

    def test_sheet(self):
        print
        print self.info

class Test_mpirun(TestCase):
    def setUp(self):
        def f(n):
            with open('test_mpirun_%s.dat'%n,'w+') as fout:
                fout.write(str(np.array(xrange(4))+n))
        self.f=f
        self.np=4

    def test_mpirun(self):
        mpirun(self.f,[(i,) for i in range(self.np)])

utilities=TestSuite([
                    TestLoader().loadTestsFromTestCase(TestTimers),
                    TestLoader().loadTestsFromTestCase(TestSheet),
                    TestLoader().loadTestsFromTestCase(Test_mpirun),
                    ])
