'''
OperatorS test.
'''

__all__=['test_operators']

from numpy import *
from HamiltonianPy.Basics.Geometry import *
from HamiltonianPy.Basics.DegreeOfFreedom import *
from HamiltonianPy.Basics.Operator import *
from HamiltonianPy.Basics.SpinPackage import *

def test_operators():
    print 'test_operators'
    J,N=1.0,2
    p1=Point(pid=PID(scope='WG',site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0])
    a1=array([1.0,0.0])
    points=tiling(cluster=[p1],vectors=[a1],indices=xrange(N))
    l=Lattice(name='WG',points=points,vectors=[a1*N],nneighbour=1)

    config=Configuration(priority=DEFAULT_SPIN_PRIORITY)
    for point in points:
        config[point.pid]=Spin(S=0.5)
    opts=OperatorCollection()
    table=config.table()

    for bond in l.bonds:
        if bond.neighbour==1:
            spid,epid=bond.spoint.pid,bond.epoint.pid
            sS,eS=config[spid].S,config[epid].S
            sindex,eindex=Index(pid=spid,iid=SID(S=sS)),Index(pid=epid,iid=SID(S=eS))
            opts+=OperatorS(
                value=      J/2,
                indices=    [sindex,eindex],
                spins=      [SpinMatrix(id=(sS,'+'),dtype=float64),SpinMatrix(id=(eS,'-'),dtype=float64)],
                rcoords=     [bond.spoint.rcoord,bond.epoint.rcoord],
                icoords=     [bond.spoint.icoord,bond.epoint.icoord],
                seqs=       (table[sindex],table[eindex])
            )
            opts+=OperatorS(
                value=      J/2,
                indices=    [sindex,eindex],
                spins=      [SpinMatrix(id=(sS,'z'),dtype=float64),SpinMatrix(id=(eS,'z'),dtype=float64)],
                rcoords=    [bond.spoint.rcoord,bond.epoint.rcoord],
                icoords=    [bond.spoint.icoord,bond.epoint.icoord],
                seqs=       (table[sindex],table[eindex])
                )
    print opts
    print
