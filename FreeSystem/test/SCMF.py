'''
SCMF test.
'''

__all__=['test_scmf']

from HamiltonianPy.Basics import *
from HamiltonianPy.FreeSystem.SCMF import *
from HamiltonianPy.FreeSystem.TBA import *

def haldane_hopping(bond):
    theta=azimuthd(bond.rcoord)
    if abs(theta)<RZERO or abs(theta-120)<RZERO or abs(theta-240)<RZERO: 
        result=1
    else:
        result=-1
    return result

def test_scmf():
    print 'test_scmf'
    U,t1,t2=3.13,-1.0,0.1
    H2=Hexagon(name='H2')
    lattice=Lattice(name='H2_SCMF',rcoords=H2.rcoords,vectors=H2.vectors,nneighbour=2)
    config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY)
    for pid in lattice.pids:
        config[pid]=Fermi(atom=pid.site%2,norbital=1,nspin=2,nnambu=1)
    h2=SCMF(
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
    h2.iterate(KSpace(reciprocals=h2.lattice.reciprocals,nk=100),tol=10**-5,maxiter=400)
    h2.register(EB(name='EB',path=hexagon_gkm(nk=100),savedata=False,plot=True,show=True,run=TBAEB))
    h2.register(BC(name='BC',BZ=KSpace(reciprocals=h2.lattice.reciprocals,nk=200),mu=h2.mu,d=10**-6,savedata=False,run=TBABC))
    h2.summary()
    print
