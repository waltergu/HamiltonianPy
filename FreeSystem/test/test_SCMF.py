'''
SCMF test (1 test in total).
'''

__all__=['scmf']

from HamiltonianPy.Basics import *
from HamiltonianPy.FreeSystem.SCMF import *
from HamiltonianPy.FreeSystem.TBA import *
from unittest import TestCase,TestLoader,TestSuite

class TsetSCMF(TestCase):
    def test_scmf(self):
        print()
        U,t1,t2=3.13,-1.0,0.1
        lattice=Hexagon(name='H2')('1P-1P',2)
        config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY,pids=lattice.pids,map=lambda pid: Fock(atom=pid.site%2,norbital=1,nspin=2,nnambu=1))
        def haldane_hopping(bond):
            theta=azimuthd(bond.rcoord)
            return 1 if abs(theta)<RZERO or abs(theta-120)<RZERO or abs(theta-240)<RZERO else -1
        scmf=SCMF(
            name=       'H2_SCMF',
            parameters= {'U':U},
            lattice=    lattice,
            config=     config,
            filling=    0.5,
            terms=[     Hopping('t1',t1),
                        Hopping('t2',t2*1j,indexpacks=sigmaz('sl'),amplitude=haldane_hopping,neighbour=2),
                        ],
            orders=[    Onsite('afm',0.2,indexpacks=sigmaz('sp')*sigmaz('sl'),modulate=lambda **karg: -U*karg['afm']/2 if 'afm' in karg else None)
                        ],
            mask=       ['nambu']
            )
        scmf.iterate(KSpace(reciprocals=scmf.lattice.reciprocals,nk=100),tol=10**-5,maxiter=400)
        scmf.register(EB(name='EB',path=hexagon_gkm(nk=100),savedata=False,plot=True,show=True,run=TBAEB))
        scmf.register(BC(name='BC',BZ=KSpace(reciprocals=scmf.lattice.reciprocals,nk=200),mu=scmf.mu,d=10**-6,savedata=False,run=TBABC))
        scmf.summary()

scmf=TestSuite([
            TestLoader().loadTestsFromTestCase(TsetSCMF),
            ])
