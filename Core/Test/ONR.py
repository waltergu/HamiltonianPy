from Hamiltonian.Core.CoreAlgorithm.ONRPy import *
from Hamiltonian.Core.BasicClass.LatticePy import *
from Hamiltonian.Core.BasicClass.BaseSpacePy import *
def test_onr():
    U=0.0
    t=-1.0
    m=2;n=2
    p1=Point(scope='WG'+str(m)+str(n),site=0,rcoord=[0.0,0.0],icoord=[0.0,0.0],struct=Fermi(atom=0,norbital=1,nspin=2,nnambu=1))
    a1=array([1.0,0.0])
    a2=array([0.0,1.0])
    a=ONR(
            name=       'WG'+str(m)+str(n),
            ensemble=   'c',
            filling=    0.5,
            mu=         U/2,
            basis=      BasisE(up=(m*n,m*n/2),down=(m*n,m*n/2)),
            #basis=      BasisE((2*m*n,m*n)),
            nspin=      2,
            lattice=    Lattice(name='WG'+str(m)+str(n),points=[p1],translations=[(a1,m),(a2,n)]),
            terms=[     Hopping('t',t,neighbour=1),
                        Hubbard('U',U,modulate=lambda **karg:karg['U'])
                        ]
        )
    a.addapps('GFC',GFC(nstep=200,save_data=False,vtype='RD',run=ONRGFC))
    a.addapps('DOS',DOS(emin=-5,emax=5,ne=401,eta=0.05,save_data=False,run=ONRDOS,show=True))
    #a.addapps('EB',EB(path=BaseSpace({'tag':'U','mesh':linspace(0.0,5.0,100)}),ns=6,save_data=False,run=ONREB))
    a.runapps()
