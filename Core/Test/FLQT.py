from Hamiltonian.Core.CoreAlgorithm.FLQTPy import *
from Hamiltonian.Core.BasicClass.BaseSpacePy import *
from Hamiltonian.Core.BasicClass.LatticePy import *
def test_flqt():
    N=50
    mu1=0.0
    mu2=3.0
    name='FLQT'
    p1=Point(scope=name,site=0,rcoord=[0.0],icoord=[0.0],struct=Fermi(norbital=1,nspin=1,nnambu=2))
    a1=array([1.0])
    a=FLQT(
        name=       name,
        lattice=    Lattice(name=name,points=[p1],translations=[(a1,N)]),
        #lattice=    Lattice(name=name,points=[p1],vectors=[a1]),
        terms=[     Hopping('t1',-1.0),
                    Onsite('mu',0.0,modulate=lambda **karg: mu1 if karg['t']<0.5 else mu2),
                    Pairing('delta',0.5,neighbour=1,amplitude=lambda bond: 1 if bond.rcoord[0]>0 else -1)
                    ],
        nambu=True
        )
    #a.addapps('EB',EB(path=BaseSpace({'tag':'t','mesh':array([0,1])}),save_data=False,run=TBAEB))
    a.addapps('EB',EB(ts=TSpace(array([0,0.5,1])),save_data=False,run=FLQTEB))
    a.runapps()
