'''
SED test.
'''

__all__=['test_sed']

import numpy as np
from HamiltonianPy import *
from HamiltonianPy.ED import *

def test_sed():
    print 'test_sed'
    J,m,n,target=1.0,6,1,SQN(0.0)
    lattice=Hexagon('H4')('%sO-%sP'%(m,n),1)
    config=IDFConfig(pids=lattice.pids,priority=DEFAULT_SPIN_PRIORITY,map=lambda pid: Spin(S=0.5))
    qnses=QNSConfig(indices=config.table().keys(),priority=DEFAULT_SPIN_PRIORITY,map=lambda index: SQNS(0.5))
    sed=SED(
        name=       'WG-%s-%s'%(lattice.name,repr(target)),
        lattice=    lattice,
        config=     config,
        qnses=      qnses,
        terms=[     SpinTerm('J',J,neighbour=1,indexpacks=Heisenberg())
                    ],
        target=     target,
        dtype=      np.float64
    )
    sed.register(EIGS(name='GSE',ne=4,run=EDEIGS))
    sed.summary()
    print
