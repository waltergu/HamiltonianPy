from HamiltonianPP.Basics import Lattice
from HamiltonianPP.DataBase.Hexagon import *
def test_hexagon():
    print 'test_hexagon'
    for name in ['H2','H4','H6','H8P','H8O','H10']:
        buff=HexagonDataBase(name=name,scope=name)
        print buff.vectors
        l=Lattice(name=name,points=buff.points,vectors=buff.vectors)
        l.plot(pid_on=True)
    print
