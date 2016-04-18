from Hamiltonian.Core.CoreAlgorithm.TBAPy import *
from Hamiltonian.Core.BasicClass.BaseSpacePy import *
from Hamiltonian.Core.BasicClass.LatticePy import *
def test_tba():
    p1=Point(scope='WG',site=0,rcoord=[0.0],icoord=[0.0],struct=Fermi(norbital=1,nspin=1,nnambu=2))
    p2=Point(scope='WG',site=1,rcoord=[0.5],icoord=[0.0],struct=Fermi(norbital=1,nspin=1,nnambu=2))
    a1=array([1.0])
    a=TBA(
        name=       'WG',
        lattice=    Lattice(name='WG',points=[p1,p2],translations=[(a1,20)]),
        #lattice=    Lattice(name='WG',points=[p1,p2],vectors=[a1]),
        terms=[     Hopping('t1',-1.0),
                    Hopping('t2',-0.1,amplitude=lambda bond: 1 if (bond.spoint.site%2==1 and bond.rcoord[0]>0) or (bond.spoint.site%2==0 and bond.rcoord[0]<0) else -1),
                    Onsite('mu',0.0,modulate=lambda **karg:karg['mu']),
                    Pairing('delta',0.5,neighbour=1,amplitude=lambda bond: 1 if bond.rcoord[0]>0 else -1)
                    ],
        nambu=      True
        )
    #a.addapps('EB',EB(save_data=False,run=TBAEB))
    #a.addapps('DOS',DOS(ne=400,eta=0.01,save_data=False,run=TBADOS))
    #a.addapps('EB',EB(path=line_1d(nk=200),save_data=False,run=TBAEB))
    #a.addapps('DOS',DOS(BZ=line_1d(nk=10000),eta=0.01,ne=400,save_data=False,run=TBADOS))
    a.addapps('EB',EB(path=BaseSpace({'tag':'mu','mesh':linspace(-3,3,num=201)}),run=TBAEB,save_data=False))
    a.runapps()
