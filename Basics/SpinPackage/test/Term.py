'''
Spin term test.
'''

__all__=['test_spin_term']

from numpy import *
from HamiltonianPy.Basics.Geometry import *
from HamiltonianPy.Basics.DegreeOfFreedom import *
from HamiltonianPy.Basics.Operator import *
from HamiltonianPy.Basics.Term import *
from HamiltonianPy.Basics.Generator import *
from HamiltonianPy.Basics.SpinPackage import *

def test_spin_term():
    print 'test_spin_term'
    J,h,N=1.0,5.0,2
    lattice=Lattice(name='WG',rcoords=tiling([array([0.0,0.0])],vectors=[array([1.0,0.0])],translations=xrange(N)),vectors=[array([1.0,0.0])*N],nneighbour=1)
    config=IDFConfig(priority=DEFAULT_SPIN_PRIORITY,map=lambda pid: Spin(S=0.5),pids=lattice.pids)
    table=config.table()
    terms=[ SpinTerm('J',J,neighbour=1,indexpacks=Heisenberg()),
            SpinTerm('h',h,neighbour=0,indexpacks=S('WG',matrix=array([[0.5,0],[0,-0.5]])))
          ]
    generator=Generator(lattice.bonds,config,terms=terms,dtype=float64)
    matrix=0
    for opt in generator.operators.values():
        matrix+=soptrep(opt,table)
    print matrix.todense()
    print
