'''
FED test (1 test in total).
'''

__all__=['fed']

import numpy as np
from HamiltonianPy import *
from HamiltonianPy.ED import *
from unittest import TestCase,TestLoader,TestSuite

class TestFED(TestCase):
    def test_fed(self):
        print
        t,U,m,n=-1.0,8.0,2,5
        basis=FBasis(2*m*n,m*n,0.0)
        lattice=Square('S1')('%sO-%sO'%(m,n))
        config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY,pids=lattice.pids,map=lambda pid: Fermi(atom=0,norbital=1,nspin=2,nnambu=1))
        fed=FED(
                name=       'WG-%s-%s'%(lattice.name,basis.rep),
                sectors=    [basis],
                lattice=    lattice,
                config=     config,
                terms=[     Hopping('t',t,neighbour=1),
                            Hubbard('U',U,modulate=True)
                            ],
                dtype=      np.float64
            )
        fed.register(EL(name='EL',path=BaseSpace(('U',np.linspace(0.0,5.0,11))),ns=4,savedata=False,run=EDEL))
        fed.add(FGF(name='GF',method='S',operators=fspoperators(config.table(),lattice),nstep=200,savedata=False,np=None,prepare=EDGFP,run=EDGF))
        fed.register(DOS(name='DOS-1',parameters={'U':0.0},mu=0.0,emin=-10,emax=10,ne=501,eta=0.05,savedata=False,run=EDDOS,dependences=['GF']))
        fed.register(DOS(name='DOS-2',parameters={'U':8.0},mu=4.0,emin=-10,emax=10,ne=501,eta=0.05,savedata=False,run=EDDOS,dependences=['GF']))
        fed.summary()

fed=TestSuite([
            TestLoader().loadTestsFromTestCase(TestFED),
            ])
