'''
FBFM test (2 tests in total).
'''

__all__=['fbfm']

import numpy as np
import HamiltonianPy.FBFM as FB
from HamiltonianPy import *
from fractions import Fraction
from unittest import TestCase,TestLoader,TestSuite

class TestFBFM(TestCase):
    def fbfmconstruct(self,t,sd,Us,Ud,basis,lattice):
        dt=sd**2/t-2*t
        result=FB.FBFM(
            name=           'fbfm_%s'%lattice.name,
            basis=          basis,
            lattice=        lattice,
            config=         IDFConfig(priority=FB.FBFM_PRIORITY,pids=lattice.pids,map=lambda pid: Fermi(atom=(pid.site+1)%2,norbital=1,nspin=2,nnambu=1)),
            terms=[         Hopping('t',t,neighbour=2,atoms=[0,0]),
                            Hopping('sd',sd,neighbour=1,modulate=True),
                            Onsite('dt',dt,atoms=[1,1],modulate=True)
                            ],
            interactions=[  Hubbard('Us',Us,atom=0,modulate=True),
                            Hubbard('Ud',Ud,atom=1,modulate=True)
                            ],
            dtype=          np.complex128
        )
        return result

    def test_open(self):
        print
        t,sd,Us,Ud,m=1.0,1.4,0.1,0.1,30
        S2x=Square('S2x')('%sO-1O'%m,nneighbour=2)
        fbfm=self.fbfmconstruct(t,sd,Us,Ud,FB.FBFMBasis(BZ=None,polarization='up',filling=Fraction(m-1,m*4)),S2x)
        fbfm.register(POS(name='POS',ns=[0]+[m+n for n in xrange(-2,5)],savedata=False,run=FB.FBFMPOS))
        fbfm.register(FB.EB(name='EB2',path=BaseSpace(('sd',np.linspace(1.0,1.32,33)),('dt',(np.linspace(1.0,1.32,33))**2-2)),ne=m/2*4,savedata=False,run=FB.FBFMEB))
        fbfm.summary()

    def test_periodic(self):
        print
        t,sd,Us,Ud=1.0,1.4,0.1,0.1
        S2x=Square('S2x')('1P-1O',nneighbour=2)
        fbfm=self.fbfmconstruct(t,sd,Us,Ud,FB.FBFMBasis(BZ=FBZ(S2x.reciprocals,nks=(60,)),polarization='up'),S2x)
        fbfm.register(FB.EB(name='EB1',path='L:G1-G2',ne=4,savedata=False,run=FB.FBFMEB))
        fbfm.register(BP(name='BP',path='L:G1-G2',ns=(0,1),savedata=False,run=FB.FBFMBP))
        fbfm.summary()

fbfm=TestSuite([
            TestLoader().loadTestsFromTestCase(TestFBFM),
            ])
