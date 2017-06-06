'''
LatticePack test.
'''

__all__=['test_database_lattice']

from HamiltonianPy.Basics.DataBase.LatticePack import *

def test_database_lattice():
    print 'test_database_lattice'
    test_database_hexagon()
    test_database_triangle()
    test_database_kagome()

def test_database_hexagon():
    print 'test_database_hexagon'
    for name in ['H2','H4','H6','H8O','H8P','H10']:
        lattice=Hexagon(name)('1P-1P')
        lattice.plot(pid_on=True)
    print

def test_database_triangle():
    print 'test_database_triangle'
    for name in ['T1','T12']:
        lattice=Triangle(name)('1P-1P')
        lattice.plot(pid_on=True)
    print

def test_database_kagome():
    print 'test_database_kagome'
    for name in ['K3']:
        lattice=Kagome(name)('1P-1P')
        lattice.plot(pid_on=True)
    print
