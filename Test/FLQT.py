from numpy import *
from HamiltonianPP.Basics import *
from HamiltonianPP.FreeSystem.TBAPy import TBAEB
from HamiltonianPP.FreeSystem.FLQTPy import *
def test_flqt():
    print 'test_flqt'
    N,mu1,mu2=50,0.0,3.0

    p1=Point(pid=PID(scope='flqt',site=0),rcoord=[0.0],icoord=[0.0])
    a1=array([1.0])

    #points=[p1]
    points=tiling(cluster=[p1],vectors=[a1],indices=[(i,) for i in xrange(N)])
    config=Configuration(priority=DEFAULT_FERMIONIC_PRIORITY)
    for point in points:
        config[point.pid]=Fermi(norbital=1,nspin=1,nnambu=2)

    a=FLQT(
        name=       'flqt',
        #lattice=    Lattice(name='flqt',points=points,vectors=[a1]),
        lattice=    Lattice(name='flqt',points=points),
        config=     config,
        terms=[     Hopping('t1',-1.0),
                    Onsite('mu',0.0,modulate=lambda **karg: mu1 if karg['t']<0.5 else mu2),
                    Pairing('delta',0.5,neighbour=1,amplitude=lambda bond: 1 if bond.rcoord[0]>0 else -1)
                    ],
        nambu=True
        )
    #a.addapps('EB',EB(path=BaseSpace({'tag':'t','mesh':array([0,1])}),save_data=False,run=TBAEB))
    a.addapps('EB',EB(ts=TSpace(array([0,0.5,1])),save_data=False,run=FLQTEB))
    a.runapps()
    print
