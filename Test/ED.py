from numpy import *
from HamiltonianPP.Basics import *
from HamiltonianPP.ED.EDPy import *
import itertools
def test_ed():
    U=8.0
    t=-1.0
    m=2;n=5
    p1=Point(pid=PID(scope='WG'+str(m)+str(n),site=(0,0,0)),rcoord=[0.0,0.0],icoord=[0.0,0.0])
    a1=array([1.0,0.0])
    a2=array([0.0,1.0])
    points=tiling(cluster=[p1],vectors=[a1,a2],indices=itertools.product(xrange(m),xrange(n)))
    config=Configuration(priority=DEFAULT_FERMIONIC_PRIORITY)
    for point in points:
        config[point.pid]=Fermi(atom=0,norbital=1,nspin=2,nnambu=1)
    a=ED(
            name=       'WG'+str(m)+str(n),
            ensemble=   'c',
            filling=    0.5,
            mu=         U/2,
            basis=      BasisF(up=(m*n,m*n/2),down=(m*n,m*n/2)),
            #basis=      BasisE((2*m*n,m*n)),
            nspin=      1,
            lattice=    Lattice(name='WG'+str(m)+str(n),points=points),
            config=     config,
            terms=[     Hopping('t',t,neighbour=1),
                        Hubbard('U',U,modulate=lambda **karg:karg['U'])
                        ]
        )
    a.addapps('GFC',GFC(nstep=100,save_data=False,vtype='RD',run=EDGFC))
    a.addapps('DOS',DOS(emin=-5,emax=5,ne=401,eta=0.05,save_data=False,run=EDDOS,plot=True,show=True))
    #a.addapps('EB',EB(path=BaseSpace({'tag':'U','mesh':linspace(0.0,5.0,100)}),ns=6,save_data=False,run=EDEB))
    a.runapps()
