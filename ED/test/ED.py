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
    a1,a2=array([1.0,0.0]),array([0.0,1.0])
    lattice=Lattice(name=name,rcoords=tiling([array([0.0,0.0])],vectors=[a1,a2],translations=it.product(xrange(m),xrange(n))))
    config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY)
    for pid in lattice.pids:
        config[pid]=Fermi(atom=0,norbital=1,nspin=2,nnambu=1)
    a=ED(
            name=       name,
            basis=      BasisF(up=(m*n,m*n/2),down=(m*n,m*n/2)),
            lattice=    lattice,
            config=     config,
            terms=[     Hopping('t',t,neighbour=1),
                        Hubbard('U',U,modulate=lambda **karg:karg.get('U',None))
                        ]
        )
    a.register(EL(name='EL',path=BaseSpace({'tag':'U','mesh':linspace(0.0,5.0,100)}),ns=6,save_data=False,run=EDEL))
    gf=GF(name='GF',nspin=2,nstep=100,save_data=False,prepare=EDGFP,run=EDGF)
    a.register(DOS(name='DOS-1',parameters={'U':0.0},mu=0.0,emin=-10,emax=10,ne=501,eta=0.05,save_data=False,run=EDDOS,plot=True,show=True,dependences=[gf]))
    a.register(DOS(name='DOS-2',parameters={'U':8.0},mu=4.0,emin=-10,emax=10,ne=501,eta=0.05,save_data=False,run=EDDOS,plot=True,show=True,dependences=[gf]))
    a.summary()
    print
