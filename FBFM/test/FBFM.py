'''
FBFM test.
'''

__all__=['test_fbfm']

import numpy as np
import HamiltonianPy.FBFM as FB
from HamiltonianPy import *

def test_fbfm():
    print 'test_fbfm'
    factor=1.16
    t,sd,dt,Us,Ud=1.0,factor,factor**2-2,1.0,1.0
    S2x=Square('S2x')('1P-1O',nneighbour=2)
    fbfm=FB.FBFM(
        basis=          FB.FBFMBasis(BZ=FBZ(S2x.reciprocals,nks=(50,)),polarization='up'),
        lattice=        S2x,
        config=         IDFConfig(priority=FB.FBFM_PRIORITY,pids=S2x.pids,map=lambda pid: Fermi(atom=(pid.site+1)%2,norbital=1,nspin=2,nnambu=1)),
        terms=[         Hopping('t',t,neighbour=2,atoms=[0,0]),
                        Hopping('sd',sd,neighbour=1),
                        Onsite('dt',dt,atoms=[1,1])
                        ],
        interactions=[  Hubbard('Us',Us,atom=0),
                        Hubbard('Ud',Ud,atom=1)
                        ],
        dtype=          np.complex128
    )
    fbfm.register(FB.EB(name='EB',path='L:G1-G2',ne=4,save_data=False,run=FB.FBFMEB))
    fbfm.summary()
    print
