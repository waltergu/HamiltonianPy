'''
ED test.
'''

__all__=['test_ed']

from numpy import *
from HamiltonianPy.Basics import *
from HamiltonianPy.ED.ED import *
import itertools

def test_ed():
    print 'test_ed'
    U=0.0
    t=-1.0
    m=2;n=2
    name='%s%s%s'%('WG',m,n)
    p1=Point(pid=PID(scope=name,site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0])
    a1=array([1.0,0.0])
    a2=array([0.0,1.0])
    points=tiling(cluster=[p1],vectors=[a1,a2],indices=itertools.product(xrange(m),xrange(n)))
    config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY)
    for point in points:
        config[point.pid]=Fermi(atom=0,norbital=1,nspin=2,nnambu=1)
    a=ED(
            name=       name,
            ensemble=   'c',
            filling=    0.5,
            mu=         U/2,
            basis=      BasisF(up=(m*n,m*n/2),down=(m*n,m*n/2)),
            #basis=      BasisF((2*m*n,m*n)),
            nspin=      1,
            lattice=    Lattice(name=name,points=points),
            config=     config,
            terms=[     Hopping('t',t,neighbour=1),
                        Hubbard('U',U,modulate=lambda **karg:karg.get('U',None))
                        ]
        )
    a.register(EB(id='EB',path=BaseSpace({'tag':'U','mesh':linspace(0.0,5.0,100)}),ns=6,save_data=False,run=EDEB))
    a.register(DOS(id='DOS',emin=-5,emax=5,ne=501,eta=0.05,save_data=False,run=EDDOS,plot=True,show=True),dependence=[GFC(id='GFC',nstep=4,save_data=False,vtype='RD',paras={'U':0.0},run=EDGFC)])
    a.runapps()
    print
