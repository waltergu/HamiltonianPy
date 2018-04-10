'''
SED test (1 test in total).
'''

__all__=['sed']

import numpy as np
from HamiltonianPy import *
from HamiltonianPy.ED import *
from unittest import TestCase,TestLoader,TestSuite

class TestSED(TestCase):
    def test_sed(self):
        print
        J,h,m,n=1.0,0.0,4,1
        lattice=Hexagon('H4')('%sO-%sP'%(m,n),1)
        config=IDFConfig(pids=lattice.pids,priority=DEFAULT_SPIN_PRIORITY,map=lambda pid: Spin(S=0.5))
        qnses=QNSConfig(indices=config.table().keys(),priority=DEFAULT_SPIN_PRIORITY,map=lambda index: SQNS(0.5))
        sed=SED(
            name=       'WG-%s'%lattice.name,
            lattice=    lattice,
            config=     config,
            qnses=      qnses,
            sectors=    [SQN(0.5*i) for i in xrange(-4*m*n,4*m*n+1,2)],
            terms=[     SpinTerm('J',J,neighbour=1,indexpacks=Heisenberg()),
                        SpinTerm('h',h,neighbour=0,indexpacks=S('z'),modulate=True)
                        ],
            dtype=      np.float64
        )
        sed.register(EIGS(name='GSE',parameters={'h':0.0},sector=SQN(0.0),ne=1,run=EDEIGS))
        sed.register(EIGS(name='GSE',parameters={'h':0.0},sector=SQN(1.0),ne=1,run=EDEIGS))
        sed.register(EL(name='EL',path=BaseSpace(('h',np.linspace(0.4,0.8,41))),ns=2,nder=0,savedata=False,run=EDEL))
        sed.summary()

sed=TestSuite([
            TestLoader().loadTestsFromTestCase(TestSED),
            ])