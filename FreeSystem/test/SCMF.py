'''
SCMF test.
'''

__all__=['test_scmf']

from HamiltonianPy.Basics import *
from HamiltonianPy.DataBase.Hexagon import *
from HamiltonianPy.FreeSystem.SCMF import *
from HamiltonianPy.FreeSystem.TBA import *

def haldane_hopping(bond):
    theta=azimuthd(bond.rcoord)
    if abs(theta)<RZERO or abs(theta-120)<RZERO or abs(theta-240)<RZERO: 
        result=1
    else:
        result=-1
    return result

def test_scmf():
    print 'test_scmf'
    U,t1,t2=3.13,-1.0,0.1
    H2=HexagonDataBase(name='H2',scope='H2_SCMF')
    h2=SCMF(
        parameters= {'U':U},
        name=       'H2_SCMF',
        lattice=    Lattice(name='H2_SCMF',points=H2.points,vectors=H2.vectors,nneighbour=2),
        config=     IDFConfig(
                        pids=[p.pid for p in H2.points],
                        map=lambda pid: Fermi(atom=pid.site%2,norbital=1,nspin=2,nnambu=1),
                        priority=DEFAULT_FERMIONIC_PRIORITY
                        ),
        mu=         0,
        filling=    0.5,
        terms=      [
                    Hopping('t1',t1),
                    Hopping('t2',t2*1j,indexpacks=sigmaz('sl'),amplitude=haldane_hopping,neighbour=2),
                    #Onsite('stagger',0.2,indexpacks=sigmaz('sl'))
                    ],
        orders=     [
                    Onsite('afm',0.2,indexpacks=sigmaz('sp')*sigmaz('sl'),modulate=lambda **karg: -U*karg['afm']/2 if 'afm' in karg else None)
                    ],
        mask=      ['nambu']
        )
    h2.iterate(KSpace(reciprocals=h2.lattice.reciprocals,nk=100),tol=10**-5,maxiter=400)
    h2.register(EB(name='EB',path=hexagon_gkm(nk=100),save_data=False,plot=True,show=True,run=TBAEB))
    h2.register(BC(name='BC',BZ=KSpace(reciprocals=h2.lattice.reciprocals,nk=200),d=10**-6,save_data=False,plot=False,show=True,run=TBABC))
    h2.summary()
    print
