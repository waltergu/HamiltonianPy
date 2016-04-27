'''
HexagonDataBase test.
'''

__all__=['test_hexagon']

from HamiltonianPP.Basics import Lattice
from HamiltonianPP.DataBase import *

def test_hexagon():
    print 'test_hexagon'
    for name in ['H2','H4','H6','H8O','H8P','H10']:
        buff=HexagonDataBase(name=name,scope=name)
        l=Lattice(name=name,points=buff.points,vectors=buff.vectors)
        l.plot(pid_on=True)
    print
