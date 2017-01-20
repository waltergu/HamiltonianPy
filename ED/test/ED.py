'''
ED test.
'''

__all__=['test_ed']

from numpy import *
from HamiltonianPy import *
from HamiltonianPy.ED.ED import *
import itertools as it

def test_ed():
    print 'test_ed'
    t,U=-1.0,8.0
    m,n=2,2
    name='%s%s%s'%('WG',m,n)
    p=Point(pid=PID(scope=name,site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0])
    a1=array([1.0,0.0])
    a2=array([0.0,1.0])
    points=tiling(cluster=[p],vectors=[a1,a2],indices=it.product(xrange(m),xrange(n)))
    config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY)
    for point in points:
        config[point.pid]=Fermi(atom=0,norbital=1,nspin=2,nnambu=1)
    a=ED(
            name=       name,
            filling=    0.5,
            mu=         U/2,
            basis=      BasisF(up=(m*n,m*n/2),down=(m*n,m*n/2)),
            lattice=    Lattice(name=name,points=points),
            config=     config,
            terms=[     Hopping('t',t,neighbour=1),
                        Hubbard('U',U,modulate=lambda **karg:karg.get('U',None))
                        ]
        )
    a.register(EL(name='EL',path=BaseSpace({'tag':'U','mesh':linspace(0.0,5.0,100)}),ns=6,save_data=False,run=EDEL))
    gf=GF(name='GF',nspin=2,nstep=100,save_data=False,prepare=EDGFP,run=EDGF)
    a.register(DOS(name='DOS-1',parameters={'U':0.0},emin=-5,emax=5,ne=501,eta=0.05,save_data=False,run=EDDOS,plot=True,show=True,dependences=[gf]))
    a.register(DOS(name='DOS-2',parameters={'U':8.0},emin=-5,emax=5,ne=501,eta=0.05,save_data=False,run=EDDOS,plot=True,show=True,dependences=[gf]))
    a.summary()
    print
