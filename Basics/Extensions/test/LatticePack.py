'''
LatticePack test.
'''

__all__=['test_extensions_lattice']

from HamiltonianPy.Basics.Extensions.LatticePack import *

def test_extensions_lattice():
    print 'test_extensions_lattice'
    test_extensions_hexagon()
    test_extensions_triangle()
    test_extensions_kagome()

def test_extensions_hexagon():
    print 'test_extensions_hexagon'
    for name in ['H2','H4','H6','H8O','H8P','H10']:
        lattice=Hexagon(name)('1P-1P')
        lattice.plot(pid_on=True)
    print

def test_extensions_triangle():
    print 'test_extensions_triangle'
    for name in ['T1','T12']:
        lattice=Triangle(name)('1P-1P')
        lattice.plot(pid_on=True)
    print

def test_extensions_kagome():
    print 'test_extensions_kagome'
    for name in ['K3']:
        lattice=Kagome(name)('1P-1P')
        lattice.plot(pid_on=True)
    print
