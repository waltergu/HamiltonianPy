'''
VCA test.
'''

__all__=['test_vca']

from numpy import *
from HamiltonianPy.Basics import *
from HamiltonianPy.ED import *
from HamiltonianPy.VCA import *
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
            weiss=[     Onsite('afm',0.2,indexpacks=sigmaz('sp'),amplitude=lambda bond: 1 if bond.spoint.pid.site in (0,3) else -1,modulate=lambda **karg:karg.get('afm',None))]
            )
    gfc=GFC(nstep=200,save_data=False,vtype='RD',run=EDGFC)
    a.register(
        app=        GPM(id='afm',fout='afm.dat',BS={'afm':0.1},method='BFGS',options={'disp':True},save_data=False,run=VCAGPM),
        dependence= [   gfc,
                        GP(BZ=square_bz(reciprocals=a.lattice.reciprocals,nk=100),run=VCAGP)
                    ]
        )
    a.register(
        app=       GPM(id='afm_curve',BS=BaseSpace({'tag':'afm','mesh':linspace(0.0,0.3,16)}),save_data=False,plot=True,run=VCAGPM),
        dependence= [   gfc,
                        GP(BZ=square_bz(reciprocals=a.lattice.reciprocals,nk=100),run=VCAGP)
                    ]
        )
    a.register(
        app=        EB(id='EB',paras={'afm':0.20},path=square_gxm(nk=100),emax=6.0,emin=-6.0,eta=0.05,ne=400,save_data=False,run=VCAEB),
        dependence= [gfc]
        )
    a.register(
        app=        DOS(id='DOS',paras={'afm':0.20},BZ=square_bz(reciprocals=a.lattice.reciprocals,nk=20),emin=-10,emax=10,ne=400,eta=0.05,save_data=False,plot=True,show=True,run=VCADOS),
        dependence= [gfc]
        )
    a.register(
        app=        FS(id='FS',paras={'afm':0.20},BZ=square_bz(nk=100),save_data=False,run=VCAFS),
        dependence= [gfc]
        )
    a.register(
        app=        OP(id='OP',paras={'afm':0.20},terms=a.weiss,BZ=square_bz(reciprocals=a.lattice.reciprocals,nk=100),run=VCAOP),
        dependence= [gfc]
        )
    a.register(
        app=        FF(id='FF',paras={'afm':0.20},BZ=square_bz(reciprocals=a.lattice.reciprocals,nk=100),p=0.5,run=VCAFF),
        dependence= [gfc]
    )
    #a.register(
    #    app=        CP(id='CP',BZ=square_bz(reciprocals=a.lattice.reciprocals,nk=100),error=10**-6,run=VCACP),
    #    dependence= [gfc]
    #    )
    a.runapps()
    print
