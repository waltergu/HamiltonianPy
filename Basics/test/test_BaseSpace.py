'''
BaseSpace test (7 tests in total).
'''

__all__=['basespace']

from HamiltonianPy.Basics.BaseSpace import *
from unittest import TestCase,TestLoader,TestSuite
from collections import OrderedDict
import numpy as np
import numpy.linalg as nl
import time

class TestSquare(TestCase):
    def setUp(self):
        self.a1=np.array([2*np.pi,0.0])
        self.a2=np.array([0.0,2*np.pi])
        self.nk=100

    def test_kspace(self):
        square=KSpace(reciprocals=[self.a1,self.a2],nk=self.nk)
        square.plot(name='square')
        self.assertEqual(1.0,square.volume('k')/(2*np.pi)**2)

    def test_fbz(self):
        square=FBZ(reciprocals=[self.a1,self.a2],nks=self.nk)
        square.plot(name='square(fbz)')
        self.assertEqual(1.0,square.volume('k')/(2*np.pi)**2)

    def test_path(self):
        square=FBZ(reciprocals=[self.a1,self.a2],nks=self.nk)
        t1=time.time()
        path=square.path([(0,self.a1/2),(self.a1/2,(self.a1+self.a2)/2),((self.a1+self.a2)/2,-(self.a1+self.a2)/2),(-(self.a1+self.a2)/2,-self.a2/2),(-self.a2/2,0)])
        t2=time.time()
        print('time,rank: %1.2fs,%s'%(t2-t1,path.rank('k')))
        path.plot(name='square(path)')

class TestHexagon(TestCase):
    def setUp(self):
        self.a1=np.array([1.0,0.0])
        self.a2=np.array([0.5,np.sqrt(3.0)/2])
        self.nk=100

    def test_kspace(self):
        hexagon=KSpace(reciprocals=[self.a1,self.a2],nk=self.nk)
        hexagon.plot(name='hexagon')
        self.assertEqual(np.sqrt(3.0)/2,hexagon.volume('k'))

    def test_fbz(self):
        hexagon=FBZ(reciprocals=[self.a1,self.a2],nks=self.nk)
        self.assertEqual(np.sqrt(3.0)/2,hexagon.volume('k'))
        hexagon.plot(name='hexagon(fbz)')

    def test_path(self):
        hexagon=FBZ(reciprocals=[self.a1,self.a2],nks=self.nk)
        t1=time.time()
        path=hexagon.path([(0,self.a1/2),(self.a1/2,(self.a1+self.a2)/3),((self.a1+self.a2)/3,-(self.a1+self.a2)/3),(-(self.a1+self.a2)/3,-self.a2/2),(-self.a2/2,0)])
        t2=time.time()
        print('time,rank: %1.2fs,%s'%(t2-t1,path.rank('k')))
        path.plot(name='hexagon(path)')

class TestBaseSpace(TestCase):
    def setUp(self):
        self.basespace=BaseSpace(('k',np.array([1,2,3,4])),('t',np.array([11,12,13,14])))

    def test_callable(self):
        for i,paras in enumerate(self.basespace('*')):
            self.assertEqual(paras,OrderedDict([('k',i//4+1),('t',i%4+11)]))
        for i,paras in enumerate(self.basespace('+')):
            self.assertEqual(paras,OrderedDict([('k',i+1),('t',i+11)]))

basespace=TestSuite([
                    TestLoader().loadTestsFromTestCase(TestSquare),
                    TestLoader().loadTestsFromTestCase(TestHexagon),
                    TestLoader().loadTestsFromTestCase(TestBaseSpace),
                    ])
