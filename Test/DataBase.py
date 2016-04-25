from HamiltonianPP.Basics import Lattice
from HamiltonianPP.DataBase import *
def test_database():
    test_hexagon()
    test_triangle()

def test_hexagon():
    print 'test_hexagon'
    for name in ['H2','H4','H6','H8O','H8P','H10']:
        buff=HexagonDataBase(name=name,scope=name)
        l=Lattice(name=name,points=buff.points,vectors=buff.vectors)
        l.plot(pid_on=True)
    print

def test_triangle():
    print 'test_triangle'
    for name in ['T1','T12']:
        buff=TriangleDataBase(name=name,scope=name)
        l=Lattice(name=name,points=buff.points,vectors=buff.vectors,max_coordinate_number=8)
        l.plot(pid_on=True)
    print
