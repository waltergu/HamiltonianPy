'''
KSpacePack test (6 tests in total).
'''

__all__=['kspace']

import numpy as np
from HamiltonianPy.Basics.Extensions.KSpacePack import *
from unittest import TestCase,TestLoader,TestSuite

class TestKMap(TestCase):
    def test_kmap(self):
        KMap.view('S',name='Square_KMap')
        KMap.view('H',name='Hexagon_KMap')

class Test_functions(TestCase):
    def test_square_bz(self):
        kspace=square_bz(reciprocals=[np.array([1.0,1.0]),np.array([1.0,-1.0])],nk=100)
        kspace.plot(name='diamond')
        self.assertAlmostEqual(kspace.volume('k'),2.0)

    def test_rectangle_bz(self):
        kspace=rectangle_bz(nk=100)
        kspace.plot(name='rectangle')
        self.assertAlmostEqual(kspace.volume('k')/(2*np.pi)**2,1.0)

    def test_hexagon_bz(self):
        for vh in ('v','h'):
            kspace=hexagon_bz(nk=100,vh=vh)
            kspace.plot(name='hexagon_%s'%vh)
            self.assertEqual(kspace.volume('k')/(2*np.pi)**2,np.sqrt(3.0)*2/3)

    def test_square_gxm(self):
        square_gxm(nk=100).plot(name='square_gxm')

    def test_hexagon_gkm(self):
        hexagon_gkm(nk=100).plot(name='hexagon_gkm')

kspace=TestSuite([
            TestLoader().loadTestsFromTestCase(TestKMap),
            TestLoader().loadTestsFromTestCase(Test_functions),
            ])
