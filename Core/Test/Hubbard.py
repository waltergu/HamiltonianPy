from Hamiltonian.Core.BasicClass.HubbardPy import *
from Hamiltonian.Core.BasicClass.LatticePy import *
def test_hubbard():
    p1=Point(site=0,scope="WG",rcoord=[0.0,0.0],icoord=[0,0],struct=Fermi(norbital=2,nspin=2,nnambu=1))
    l=Lattice(name="WG",points=[p1])
    l.plot(show='y')
    table=l.table(nambu=True)
    a=HubbardList(Hubbard('U,U,J',[20.0,12.0,5.0]))
    opts=OperatorList()
    for bond in l.bonds:
        opts.extend(a.operators(bond,table))
    print opts
