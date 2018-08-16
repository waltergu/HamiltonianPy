'''
TBA test (2 tests in total).
'''

__all__=['tba']

import numpy as np
from HamiltonianPy.Basics import *
from HamiltonianPy.FreeSystem.TBA import *
from unittest import TestCase,TestLoader,TestSuite

class TestTBA(TestCase):
    def tbaconstruct(self,bc='op',t1=-1.0,t2=-0.5,mu=0.0,delta=0.4):
        p1,p2,v=np.array([0.0,0.0]),np.array([0.5,0.0]),np.array([1.0,0.0])
        if bc=='op':
            lattice=Lattice(name='WG',rcoords=tiling(cluster=[p1,p2],vectors=[v],translations=range(20)))
        else:
            lattice=Lattice(name='WG',rcoords=[p1,p2],vectors=[v])
        config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY,pids=lattice.pids,map=lambda pid: Fock(norbital=1,nspin=1,nnambu=2))
        result=TBA(
            name=       'WG(%s)'%bc,
            lattice=    lattice,
            config=     config,
            terms=[     Hopping('t1',t1),
                        Hopping('t2',t2,amplitude=lambda bond: 1 if (bond.spoint.pid.site%2==1 and bond.rcoord[0]>0) or (bond.spoint.pid.site%2==0 and bond.rcoord[0]<0) else -1),
                        Onsite('mu',mu,modulate=lambda **karg:karg.get('mu',None)),
                        Pairing('delta',delta,neighbour=1,amplitude=lambda bond: 1 if bond.rcoord[0]>0 else -1)
                        ],
            mask=       []
        )
        return result

    def test_open(self):
        print()
        op=self.tbaconstruct(bc='op',t1=-1.0,t2=-0.5,mu=0.0,delta=0.4)
        op.register(EB(name='EB-1',path=BaseSpace(('mu',np.linspace(-3,3,num=201))),savedata=False,run=TBAEB))
        op.register(EB(name='EB-2',parameters={'mu':0.0},savedata=False,run=TBAEB))
        op.register(DOS(name='DOS',parameters={'mu':0.0},ne=400,eta=0.01,savedata=False,run=TBADOS))
        op.summary()

    def test_periodic(self):
        print()
        pd=self.tbaconstruct(bc='pd',t1=-1.0,t2=-0.5,mu=0.0,delta=0.4)
        pd.register(EB(name='EB',parameters={'mu':0.2},path=KSpace(reciprocals=pd.lattice.reciprocals,nk=200),savedata=False,run=TBAEB))
        pd.register(DOS(name='DOS',parameters={'mu':0.0},BZ=KSpace(reciprocals=pd.lattice.reciprocals,nk=10000),eta=0.01,ne=400,savedata=False,run=TBADOS))
        pd.summary()

tba=TestSuite([
            TestLoader().loadTestsFromTestCase(TestTBA),
            ])
