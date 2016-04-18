from Hamiltonian.Core.BasicClass.QuadraticPy import *
from Hamiltonian.Core.BasicClass.LatticePy import *
def test_quadratic():
    test_quadratic_body()
    test_quadratic_operators()

def test_quadratic_body():
    p1=Point(site=0,scope="WG",rcoord=[0.0,0.0],icoord=[0.0,0.0],struct=Fermi(atom=0,norbital=2,nspin=2,nnambu=2))
    p2=Point(site=1,scope="WG",rcoord=[1.0,0.0],icoord=[0.0,0.0],struct=Fermi(atom=1,norbital=2,nspin=2,nnambu=2))
    bond=Bond(neighbour=1,spoint=p1,epoint=p2)
    a=QuadraticList(Hopping('t1',1.0,neighbour=1,indexpackages=sigmaz("SP")),Hopping('t2',1,neighbour=1,indexpackages=sigmax("SL")*sigmax("SP")))
    print a.mesh(bond,False)
    b=QuadraticList(Onsite('mu1',1.0,neighbour=1,indexpackages=sigmaz("SP")),Onsite('mu2',1,neighbour=1,indexpackages=sigmaz("SP")*sigmay("OB")))
    print b.mesh(bond,False)
    c=QuadraticList(Pairing('delta1',1.0,neighbour=1,indexpackages=sigmaz("SP")),Pairing('delta2',1,neighbour=1,indexpackages=sigmaz("SP")+sigmay("OB")))
    print c.mesh(bond,False)

def test_quadratic_operators():
    p1=Point(site=0,scope="WG",rcoord=[0.0,0.0],icoord=[0.0,0.0],struct=Fermi(atom=0,norbital=2,nspin=2,nnambu=2))
    p2=Point(site=1,scope="WG",rcoord=[1.0,0.0],icoord=[0.0,0.0],struct=Fermi(atom=1,norbital=2,nspin=2,nnambu=2))
    l=Lattice(name="WG",points=[p1,p2])
    table=l.table(nambu=True)
    a=QuadraticList(Hopping('t1',1.0,neighbour=1,indexpackages=sigmaz("SP")))
    b=QuadraticList(Onsite('mu',1.0,neighbour=0,indexpackages=sigmaz("SP")))
    c=QuadraticList(Pairing('delta',1.0,neighbour=1,indexpackages=sigmaz("SP"),modulate=lambda **karg: 1))
    d=a+b+c
    opts=OperatorList()
    for bond in l.bonds:
        opts.extend(d.operators(bond,table,False))
    print opts
    print d
