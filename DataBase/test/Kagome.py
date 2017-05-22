'''
KagomeDataBase test.
'''

__all__=['test_kagome']

from HamiltonianPy.Basics import Lattice
from HamiltonianPy.DataBase import *

def test_kagome():
    print 'test_kagome'
    for name in ['K3']:
        buff=KagomeDataBase(name=name)
        l=Lattice(name=name,rcoords=buff.rcoords,vectors=buff.vectors)
        l.plot(pid_on=True,save=True)
    print

