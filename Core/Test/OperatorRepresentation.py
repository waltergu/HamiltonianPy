from Hamiltonian.Core.BasicClass.OperatorRepresentationPy import *
from Hamiltonian.Core.BasicClass.QuadraticPy import *
from Hamiltonian.Core.BasicClass.LatticePy import *
import time
def test_opt_rep():
    m=2;n=2;nloop=500
    p=Point(site=0,scope="WG",rcoord=[0.0,0.0],icoord=[0.0,0.0],struct=Fermi(atom=0,norbital=1,nspin=2,nnambu=2))
    a1=array([1.0,0.0]);a2=array([0.0,1.0])
    l=Lattice(name="WG",points=[p],translations=((a1,m),(a2,n)))
    l.plot(show='y')
    table=l.table(nambu=True)
    a=QuadraticList(Hopping('t',1.0,neighbour=1,indexpackages=sigmaz("SP")))
    b=QuadraticList(Onsite('mu',1.0,neighbour=0,indexpackages=sigmaz("SP")))
    c=QuadraticList(Pairing('delta',1.0,neighbour=1,indexpackages=sigmaz("SP")))
    opts=OperatorList()
    for bond in l.bonds:
        opts.extend(a.operators(bond,table))
        opts.extend(b.operators(bond,table))
        opts.extend(c.operators(bond,table))
    print opts
    basis=BasisE(nstate=2*m*n)
#    basis=BasisE((2*m*n,m*n))
#    basis=BasisE(up=(m*n,m*n/2),down=(m*n,m*n/2))
#    print basis
    stime=time.time()
    for i in xrange(nloop):
        opt_rep(opts[0],basis,transpose=False)
#        print opt_rep(opts[0],basis,transpose=False)
    etime=time.time()
    print etime-stime
