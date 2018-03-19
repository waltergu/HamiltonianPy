'''
TBA test.
'''

__all__=['test_tba']

from numpy import *
from HamiltonianPy.Basics import *
from HamiltonianPy.FreeSystem.TBA import *

def test_tba():
    print 'test_tba'
    print 'open boundary conditions'
    print '------------------------'
    op=tbaconstruct(bc='op')
    op.register(EB(name='EB-1',path=BaseSpace(('mu',linspace(-3,3,num=201))),savedata=False,run=TBAEB))
    op.register(EB(name='EB-2',parameters={'mu':0.0},savedata=False,run=TBAEB))
    op.register(DOS(name='DOS',parameters={'mu':0.0},ne=400,eta=0.01,savedata=False,run=TBADOS))
    op.summary()
    print 'periodic boundary conditions'
    print '----------------------------'
    pd=tbaconstruct(bc='pd')
    pd.register(EB(name='EB',parameters={'mu':0.2},path=KSpace(reciprocals=pd.lattice.reciprocals,nk=200),savedata=False,run=TBAEB))
    pd.register(DOS(name='DOS',parameters={'mu':0.0},BZ=KSpace(reciprocals=pd.lattice.reciprocals,nk=10000),eta=0.01,ne=400,savedata=False,run=TBADOS))
    pd.summary()
    print

def tbaconstruct(bc='op'):
    p1,p2=array([0.0,0.0]),array([0.5,0.0])
    a1=array([1.0,0.0])
    if bc in ('op',):
        lattice=Lattice(name='WG',rcoords=tiling(cluster=[p1,p2],vectors=[a1],translations=xrange(20)))
    else:
        lattice=Lattice(name='WG',rcoords=[p1,p2],vectors=[a1])
    config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY)
    for pid in lattice.pids:
        config[pid]=Fermi(norbital=1,nspin=1,nnambu=2)
    result=TBA(
        name=       'WG(%s)'%bc,
        lattice=    lattice,
        config=     config,
        terms=[     Hopping('t1',-1.0),
                    Hopping('t2',-0.5,amplitude=lambda bond: 1 if (bond.spoint.pid.site%2==1 and bond.rcoord[0]>0) or (bond.spoint.pid.site%2==0 and bond.rcoord[0]<0) else -1),
                    Onsite('mu',0.0,modulate=lambda **karg:karg.get('mu',None)),
                    Pairing('delta',0.4,neighbour=1,amplitude=lambda bond: 1 if bond.rcoord[0]>0 else -1)
                    ],
        mask=      []
        )
    return result
