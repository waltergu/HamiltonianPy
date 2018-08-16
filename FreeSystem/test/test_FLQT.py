'''
FLQT test (1 test in total).
'''

__all__=['flqt']

import numpy as np
from collections import OrderedDict
from HamiltonianPy.Basics import *
from HamiltonianPy.FreeSystem.TBA import TBAEB
from HamiltonianPy.FreeSystem.FLQT import *
from unittest import TestCase,TestLoader,TestSuite

class TestFLQT(TestCase):
    def test_flqt(self):
        print()
        N,mu1,mu2=50,0.0,3.0
        lattice=Lattice(name='flqt',rcoords=tiling([np.array([0.0])],vectors=[np.array([1.0])],translations=range(N)))
        config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY,pids=lattice.pids,map=lambda pid: Fock(norbital=1,nspin=1,nnambu=2))
        flqt=FLQT(
            name=       'flqt',
            parameters= OrderedDict([('t1',-1.0),('delta',0.5),('mu1',mu1),('mu2',mu2),('t',None)]),
            map=        lambda parameters: {'mu': parameters['mu1'] if parameters['t']<0.5 else parameters['mu2']},
            lattice=    lattice,
            config=     config,
            terms=[     Hopping('t1',-1.0),
                        Pairing('delta',0.5,neighbour=1,amplitude=lambda bond: 1 if bond.rcoord[0]>0 else -1),
                        Onsite('mu',0.0,modulate=True),
                        ],
            mask=       []
            )
        flqt.register(EB(name='EB',path=BaseSpace(('t',np.linspace(0,1,100))),savedata=False,run=TBAEB))
        flqt.register(QEB(name='QEB',ts=TSpace(np.array([0,0.5,1])),savedata=False,run=FLQTQEB))
        flqt.summary()

flqt=TestSuite([
            TestLoader().loadTestsFromTestCase(TestFLQT),
            ])
