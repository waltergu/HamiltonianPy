'''
HexagonDataBase test.
'''

__all__=['test_hexagon']

from HamiltonianPy.Basics import Lattice
from HamiltonianPy.DataBase import *

def test_hexagon():
    print 'test_hexagon'
    for name in ['H2','H4','H6','H8O','H8P','H10']:
        buff=HexagonDataBase(name=name)
        l=Lattice(name=name,rcoords=buff.rcoords,vectors=buff.vectors)
        l.plot(pid_on=True)
    print
