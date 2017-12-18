'''
Spin operator representation test.
'''

__all__=['test_soptrep']

from numpy import *
from HamiltonianPy.Basics.Geometry import *
from HamiltonianPy.Basics.DegreeOfFreedom import *
from HamiltonianPy.Basics.Operator import *
from HamiltonianPy.Basics.SpinPackage import *

def test_soptrep():
    print 'test_soptrep'
    J,N=1.0,2
    a1=array([1.0,0.0])
    rcoords=tiling([array([0.0,0.0])],vectors=[a1],translations=xrange(N))
    lattice=Lattice(name='WG',rcoords=rcoords,vectors=[a1*N],neighbours=1)

    config=IDFConfig(priority=DEFAULT_SPIN_PRIORITY,pids=lattice.pids,map=lambda pid: Spin(S=0.5))
    opts,table=Operators(),config.table()

    for bond in lattice.bonds:
        if bond.neighbour==1:
            spid,epid=bond.spoint.pid,bond.epoint.pid
            sS,eS=config[spid].S,config[epid].S
            sindex,eindex=Index(pid=spid,iid=SID(S=sS)),Index(pid=epid,iid=SID(S=eS))
            opts+=SOperator(
                value=      J/2,
                indices=    [sindex,eindex],
                spins=      [SpinMatrix(sS,'+',dtype=float64),SpinMatrix(eS,'-',dtype=float64)],
                seqs=       (table[sindex],table[eindex])
            )
            opts+=SOperator(
                value=      J/2,
                indices=    [sindex,eindex],
                spins=      [SpinMatrix(sS,'z',dtype=float64),SpinMatrix(eS,'z',dtype=float64)],
                seqs=       (table[sindex],table[eindex])
                )
    temp=None
    for opt in opts.values():
        if temp is None:
            temp=soptrep(opt,table)
        else:
            temp+=soptrep(opt,table)
    temp+=temp.T.conjugate()
    print temp.todense()
    print
