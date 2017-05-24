'''
FLQT test.
'''

__all__=['test_flqt']

from numpy import *
from HamiltonianPy.Basics import *
from HamiltonianPy.FreeSystem.TBA import TBAEB
from HamiltonianPy.FreeSystem.FLQT import *

def test_flqt():
    print 'test_flqt'
    N,mu1,mu2=50,0.0,3.0
    rcoords=tiling([array([0.0])],vectors=[array([1.0])],translations=xrange(N))
    lattice=Lattice(name='flqt',rcoords=rcoords)
    config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY)
    for pid in lattice.pids:
        config[pid]=Fermi(norbital=1,nspin=1,nnambu=2)
    a=FLQT(
        name=       'flqt',
        lattice=    lattice,
        config=     config,
        terms=[     Hopping('t1',-1.0),
                    Onsite('mu',0.0,modulate=lambda **karg: mu1 if karg.get('t',0.0)<0.5 else mu2),
                    Pairing('delta',0.5,neighbour=1,amplitude=lambda bond: 1 if bond.rcoord[0]>0 else -1)
                    ],
        mask=       []
        )
    a.register(EB(name='EB',path=BaseSpace(('t',linspace(0,1,100))),save_data=False,run=TBAEB))
    a.register(QEB(name='QEB',ts=TSpace(array([0,0.5,1])),save_data=False,run=FLQTQEB))
    a.summary()
    print
