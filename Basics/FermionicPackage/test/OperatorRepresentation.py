'''
Fermionic operator representation test.
'''

__all__=['test_f_opt_rep']

from numpy import *
from HamiltonianPy.Basics import *
import itertools
import time

def test_f_opt_rep():
    print 'test_f_opt_rep'
    m=2;n=2;nloop=500
    a1=array([1.0,0.0]);a2=array([0.0,1.0])
    rcoords=tiling([array([0.0,0.0])],vectors=[a1,a2],translations=itertools.product(xrange(m),xrange(n)))
    l=Lattice(name="WG",rcoords=rcoords)
    l.plot(pid_on=True)
    config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY)
    for pid in l.pids:
        config[pid]=Fermi(atom=0,norbital=1,nspin=2,nnambu=2)
    table=config.table(mask=[])
    a=+Hopping('t',1.0,neighbour=1,indexpacks=sigmaz("SP"))
    b=+Onsite('mu',1.0,neighbour=0,indexpacks=sigmaz("SP"))
    c=+Pairing('delta',1.0,neighbour=1,indexpacks=sigmaz("SP"))
    opts=OperatorCollection()
    for bond in l.bonds:
        opts+=a.operators(bond,config,table)
        opts+=b.operators(bond,config,table)
        opts+=c.operators(bond,config,table)
    print opts
    basis=BasisF(nstate=2*m*n)
#    basis=BasisF((2*m*n,m*n))
#    basis=BasisF(up=(m*n,m*n/2),down=(m*n,m*n/2))
#    print basis
    stime=time.time()
    for i in xrange(nloop):
        f_opt_rep(opts.values()[0],basis,transpose=False)
#        print f_opt_rep(opts[0],basis,transpose=False)
    etime=time.time()
    print etime-stime
    print
