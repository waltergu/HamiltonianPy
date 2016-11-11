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
    op=tba_construct(bc='op')
    op.register(EB(path=BaseSpace({'tag':'mu','mesh':linspace(-3,3,num=201)}),run=TBAEB,save_data=False,plot=True))
    op.register(EB(paras={'mu':0.0},save_data=False,run=TBAEB))
    op.register(DOS(paras={'mu':0.0},ne=400,eta=0.01,save_data=False,run=TBADOS))
    op.runapps()
    print 'periodic boundary conditions'
    print '------------------------'
    pd=tba_construct(bc='pd')
    pd.register(EB(paras={'mu':0.2},path=line_1d(nk=200),save_data=False,run=TBAEB))
    pd.register(DOS(paras={'mu':0.0},BZ=line_1d(nk=10000),eta=0.01,ne=400,save_data=False,run=TBADOS))
    pd.runapps()
    print

def tba_construct(bc='op'):
    p1=Point(pid=PID(scope='WG',site=0),rcoord=[0.0],icoord=[0.0])
    p2=Point(pid=PID(scope='WG',site=1),rcoord=[0.5],icoord=[0.0])
    a1=array([1.0])
    if bc in ('op',):
        points=tiling(cluster=[p1,p2],vectors=[a1],indices=xrange(20))
        lattice=Lattice(name='WG',points=points)
    else:
        points=[p1,p2]
        lattice=Lattice(name='WG',points=points,vectors=[a1])
    config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY)
    for point in points:
        config[point.pid]=Fermi(norbital=1,nspin=1,nnambu=2)
    result=TBA(
        name=       'WG',
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
