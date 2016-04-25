from Hamiltonian.Core.CoreAlgorithm.VCACCTPy import *
from Hamiltonian.Core.BasicClass.LatticePy import *
from Hamiltonian.Core.BasicClass.BaseSpacePy import *
def test_vcacct():
    t1,U=-1.0,0.0
    p1=Point(scope='PA',site=0,rcoord=[0.0,0.0],icoord=[0.0,0.0],struct=Fermi(nspin=2,atom=1))
    p2=Point(scope='PA',site=1,rcoord=[0.0,-sqrt(3)/3],icoord=[0.0,0.0],struct=Fermi(nspin=2,atom=2))
    p3=Point(scope='PA',site=2,rcoord=[-0.5,sqrt(3)/6],icoord=[0.0,0.0],struct=Fermi(nspin=2,atom=2))
    p4=Point(scope='PA',site=3,rcoord=[0.5,sqrt(3)/6],icoord=[0.0,0.0],struct=Fermi(nspin=2,atom=2))
    p5=Point(scope='PB',site=0,rcoord=[0.0,2*sqrt(3)/3],icoord=[0.0,0.0],struct=Fermi(nspin=2,atom=2))
    p6=Point(scope='PB',site=1,rcoord=[0.0,sqrt(3)],icoord=[0.0,0.0],struct=Fermi(nspin=2,atom=1))
    p7=Point(scope='PB',site=2,rcoord=[0.5,sqrt(3)/2],icoord=[0.0,0.0],struct=Fermi(nspin=2,atom=1))
    p8=Point(scope='PB',site=3,rcoord=[-0.5,sqrt(3)/2],icoord=[0.0,0.0],struct=Fermi(nspin=2,atom=1))
    a1=array([1.0,0.0])
    a2=array([0.5,sqrt(3)/2])
    b1=array([1.0,sqrt(3)])
    b2=array([1.5,-sqrt(3)/2])
    PA=Lattice(name='PA',points=[p1,p2,p3,p4])
    PB=Lattice(name='PB',points=[p5,p6,p7,p8])
    a=VCACCT(
        name=       'H4_concat',
        ensemble=   'c',
        filling=    0.5,
        mu=         U/2,
        nspin=      1,
        cell=       Lattice(name='Hexagon',points=[p1,p3],vectors=[a1,a2]),
        lattice=    SuperLattice(name='H4_concat',sublattices=[PA,PB],vectors=[b1,b2]),
        subsystems= [
                    {'basis':BasisE(up=(4,2),down=(4,2)),'lattice':PA},
                    {'basis':BasisE(up=(4,2),down=(4,2)),'lattice':PB}
                    ],
        terms=      [
                    Hopping('t1',t1),
                    Hubbard('U',U)
                    ],
        nambu=      False
        )
    a.addapps('GFC',GFC(nstep=200,save_data=False,vtype='RD',run=VCACCTGFC))
    a.addapps('DOS',DOS(BZ=hexagon_bz(nk=50),emin=-5,emax=5,ne=400,eta=0.05,save_data=False,run=VCADOS,plot=True,show=True))
    a.addapps('EB',EB(path=hexagon_gkm(nk=100),emax=6.0,emin=-6.0,eta=0.05,ne=400,save_data=False,plot=True,show=True,run=VCAEB))
    a.runapps()
