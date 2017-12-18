'''
LatticePack test.
'''

__all__=['test_extensions_lattice']

from HamiltonianPy.Basics.Extensions.LatticePack import *

def test_extensions_lattice():
    print 'test_extensions_lattice'
    test_extensions_line()
    test_extensions_square()
    test_extensions_hexagon()
    test_extensions_triangle()
    test_extensions_kagome()

def test_extensions_line():
    print 'test_extensions_line'
    for name in ['L1','L2']:
        lattice=Line(name)('1P')
        lattice.plot(pidon=True)
    print

def test_extensions_square():
    print 'test_extensions_square'
    for name in ['S1','S2x','S2y','S4','S4B8','S10','S12','S13']:
        lattice=Square(name)('1P-1P')
        lattice.plot(pidon=True)
    print

def test_extensions_hexagon():
    print 'test_extensions_hexagon'
    for name in ['H2','H4','H6','H6B6','H8O','H8P','H10','H24','H4C','H4CB6C']:
        lattice=Hexagon(name)('1P-1P')
        lattice.plot(pidon=True)
    print

def test_extensions_triangle():
    print 'test_extensions_triangle'
    for name in ['T1','T3','T12']:
        lattice=Triangle(name)('1P-1P')
        lattice.plot(pidon=True)
    print

def test_extensions_kagome():
    print 'test_extensions_kagome'
    for name in ['K3','K9','K12']:
        lattice=Kagome(name)('1P-1P')
        lattice.plot(pidon=True)
    print
