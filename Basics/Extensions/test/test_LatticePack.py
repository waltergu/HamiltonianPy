'''
LatticePack test (5 tests in totoal).
'''

__all__=['lattice']

from HamiltonianPy.Basics.Extensions.LatticePack import *
from unittest import TestCase,TestLoader,TestSuite

class TestLine(TestCase):
    def test_line(self):
        for name in ['L1','L2']:
            lattice=Line(name)('1P')
            lattice.plot(pidon=True)

class TestSquare(TestCase):
    def test_square(self):
        for name in ['S1','S2x','S2y','S2xxy','S2yxy','S4','S4B4','S4B8','S8','S10','S12','S13']:
            lattice=Square(name)('1P-1P')
            lattice.plot(pidon=True)

class TestHexagon(TestCase):
    def test_hexagon(self):
        for name in ['H2','H2B4','H4','H6','H6B6','H8O','H8P','H10','H24','H4C','H4CB6C']:
            lattice=Hexagon(name)('1P-1P')
            lattice.plot(pidon=True)

class TestTriangle(TestCase):
    def test_triangle(self):
        for name in ['T1','T3','T12']:
            lattice=Triangle(name)('1P-1P')
            lattice.plot(pidon=True)

class TestKagome(TestCase):
    def test_kagome(self):
        for name in ['K3','K9','K12']:
            lattice=Kagome(name)('1P-1P')
            lattice.plot(pidon=True)

lattice=TestSuite([
            TestLoader().loadTestsFromTestCase(TestLine),
            TestLoader().loadTestsFromTestCase(TestSquare),
            TestLoader().loadTestsFromTestCase(TestHexagon),
            TestLoader().loadTestsFromTestCase(TestTriangle),
            TestLoader().loadTestsFromTestCase(TestKagome),
            ])
