'''
VCA test.
'''

__all__=['test_vca']

from numpy import *
from HamiltonianPP.Basics import *
from HamiltonianPP.ED import *
from HamiltonianPP.VCA import *
import itertools

def test_vca():
    print 'test_vca'
    U=8.0
    t=-1.0
    m=2;n=2
    name='%s%s%s'%('WG',m,n)
    p1=Point(pid=PID(scope=name,site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0])
    a1=array([1.0,0.0])
    a2=array([0.0,1.0])
    points=tiling(cluster=[p1],vectors=[a1,a2],indices=itertools.product(xrange(m),xrange(n)))
    config=Configuration(priority=DEFAULT_FERMIONIC_PRIORITY)
    for point in points:
        config[point.pid]=Fermi(atom=0,norbital=1,nspin=2,nnambu=1)

    a=VCA(
            name=       name,
            ensemble=   'c',
            filling=    0.5,
            mu=         U/2,
            basis=      BasisF(up=(m*n,m*n/2),down=(m*n,m*n/2)),
            #basis=      BasisF((2*m*n,m*n)),
            #basis=      BasisF(nstate=2*m*n),
            nspin=      2,
            cell=       Lattice(name=name,points=[points[0]],vectors=[a1,a2]),
            lattice=    Lattice(name=name,points=points,vectors=[a1*m,a2*n]),
            config=     config,
            terms=[     Hopping('t',t,neighbour=1),
                        #Onsite('mu',-U/2),
                        Hubbard('U',U)
                        ],
            nambu=      False,
            weiss=[     Onsite('afm',0.2,indexpackages=sigmaz('sp'),amplitude=lambda bond: 1 if bond.spoint.pid.site in (0,3) else -1,modulate=lambda **karg:karg['afm'])]
            )
    #a.addapps(app=GFC(nstep=200,save_data=False,vtype='RD',run=EDGFC))
    #a.addapps(app=GP(BZ=square_bz(reciprocals=a.lattice.reciprocals,nk=100),rank1=128,n=64,run=VCAGP))
    #a.addapps('GPS',GPS(BS=BaseSpace({'tag':'afm','mesh':linspace(0.0,0.3,16)}),save_data=True,plot=True,run=VCAGPS))
    a.addapps('GFC',GFC(nstep=200,save_data=False,method='python',vtype='RD',error=10**-10,run=EDGFC))
    a.addapps('EB',EB(path=square_gxm(nk=100),emax=6.0,emin=-6.0,eta=0.05,ne=400,save_data=False,plot=True,show=True,run=VCAEB))
    #a.addapps('DOS',DOS(BZ=square_bz(nk=50),emin=-6,emax=6,ne=400,eta=0.05,save_data=False,plot=True,show=True,run=VCADOS))
    #a.addapps('FS',FS(BZ=square_bz(nk=100),save_data=False,run=VCAFS))
    #a.addapps('CP',CP(BZ=square_bz(reciprocals=a.lattice.reciprocals,nk=100),error=10**-6,run=VCACP))
    #a.addapps('OP',OP(terms=a.weiss,BZ=square_bz(reciprocals=a.lattice.reciprocals,nk=100),run=VCAOP))
    #a.addapps('FF',FF(BZ=square_bz(reciprocals=a.lattice.reciprocals,nk=100),p=0.5,run=VCAFF))
    a.runapps()
    print
