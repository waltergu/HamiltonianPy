'''
Spin term test.
'''

__all__=['test_spin_term']

from numpy import *
from HamiltonianPy.Basics.Geometry import *
from HamiltonianPy.Basics.DegreeOfFreedom import *
from HamiltonianPy.Basics.Operator import *
from HamiltonianPy.Basics.Term import *
from HamiltonianPy.Basics.SpinPackage import *

def test_spin_term():
    print 'test_spin_term'
    J,h,N=1.0,5.0,2
    p1=Point(pid=PID(scope='WG',site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0])
    a1=array([1.0,0.0])
    points=tiling(cluster=[p1],vectors=[a1],indices=xrange(N))
    l=Lattice(name='WG',points=points,vectors=[a1*N],nneighbour=1)
    config=Configuration(priority=DEFAULT_SPIN_PRIORITY)
    for point in points:
        config[point.pid]=Spin(S=0.5)
    opts=OperatorCollection()
    table=config.table()
    a=SpinTermList(SpinTerm('J',J,neighbour=1,indexpacks=Heisenberg()),SpinTerm('h',h,neighbour=0,indexpacks=IndexPackList(SpinPack(1.0,('z')))))
    print 'a: %s'%a
    opts=OperatorCollection()
    for bond in l.bonds:
        opts+=a.operators(bond,table,config,dtype=float64)
    print opts
    temp=None
    for opt in opts.values():
        if temp is None:
            temp=s_opt_rep(opt,table)
        else:
            temp+=s_opt_rep(opt,table)
    print temp.todense()
    print
