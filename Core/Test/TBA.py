from HamiltonianPP.Core.CoreAlgorithm.TBAPy import *
from HamiltonianPP.Core.BasicClass import *
from numpy import *
def test_tba():
    p1=Point(pid=PID(scope='WG',site=(0,0)),rcoord=[0.0],icoord=[0.0])
    p2=Point(pid=PID(scope='WG',site=(0,1)),rcoord=[0.5],icoord=[0.0])
    a1=array([1.0])

    #points=[p1,p2]
    points=tiling(cluster=[p1,p2],vectors=[a1],indices=[(i,) for i in xrange(20)])
    config=Configuration(priority=DEFAULT_FERMIONIC_PRIORITY)
    for point in points:
        config[point.pid]=Fermi(norbital=1,nspin=1,nnambu=2)

    a=TBA(
        name=       'WG',
        lattice=    Lattice(name='WG',points=points),
        #lattice=    Lattice(name='WG',points=points,vectors=[a1]),
        config=     config,
        terms=[     Hopping('t1',-1.0),
                    Hopping('t2',-0.1,amplitude=lambda bond: 1 if (bond.spoint.pid.site[1]%2==1 and bond.rcoord[0]>0) or (bond.spoint.pid.site[1]%2==0 and bond.rcoord[0]<0) else -1),
                    Onsite('mu',0.0,modulate=lambda **karg:karg['mu'] if 'mu' in karg else None),
                    Pairing('delta',0.5,neighbour=1,amplitude=lambda bond: 1 if bond.rcoord[0]>0 else -1)
                    ],
        nambu=      True
        )
    #a.addapps('EB',EB(save_data=False,run=TBAEB))
    #a.addapps('DOS',DOS(ne=400,eta=0.01,save_data=False,run=TBADOS))
    #a.addapps('EB',EB(path=line_1d(nk=200),save_data=False,run=TBAEB))
    #a.addapps('DOS',DOS(BZ=line_1d(nk=10000),eta=0.01,ne=400,save_data=False,run=TBADOS))
    a.addapps('EB',EB(path=BaseSpace({'tag':'mu','mesh':linspace(-3,3,num=201)}),run=TBAEB,save_data=False,plot=True))
    a.runapps()
