'''
Spin operator representation test.
'''

__all__=['test_s_opt_rep']

from numpy import *
from HamiltonianPy.Basics.Geometry import *
from HamiltonianPy.Basics.DegreeOfFreedom import *
from HamiltonianPy.Basics.Operator import *
from HamiltonianPy.Basics.SpinPackage import *

def test_s_opt_rep():
    print 'test_s_opt_rep'
    J,N=1.0,2
    a1=array([1.0,0.0])
    rcoords=tiling([array([0.0,0.0])],vectors=[a1],translations=xrange(N))
    l=Lattice(name='WG',rcoords=rcoords,vectors=[a1*N],nneighbour=1)

    config=IDFConfig(priority=DEFAULT_SPIN_PRIORITY)
    for pid in l.pids:
        config[pid]=Spin(S=0.5)
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
    temp=None
    for opt in opts.values():
        if temp is None:
            temp=s_opt_rep(opt,table)
        else:
            temp+=s_opt_rep(opt,table)
    temp+=temp.T.conjugate()
    print temp.todense()
    print
