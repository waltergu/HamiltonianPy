'''
TriangleDataBase test.
'''

__all__=['test_triangle']

from HamiltonianPy.Basics import Lattice
from HamiltonianPy.DataBase import *
def test_triangle():
    print 'test_triangle'
    for name in ['T1','T12']:
        buff=TriangleDataBase(name=name,scope=name)
        l=Lattice(name=name,points=buff.points,vectors=buff.vectors,max_coordinate_number=8)
        l.plot(pid_on=True)
    print
