'''
BasisF test.
'''
__all__=['test_basisf']

from HamiltonianPP.Basics.FermionicPackage.BasisFPy import *
from numba import jit
import time

def test_basisf():
    print 'test_basisf'
    stime=time.time()
    m=12;n=6;nloop=100
    a=BasisF(up=(m,n),down=(m,n))
    for i in xrange(nloop):
        test_while1(a.nbasis,a.basis_table)
        test_while2(a.nbasis,a.basis_table)
    etime=time.time()
    print etime-stime
    print

@jit
def test_while1(nbasis,basis_table):
    ntable=len(basis_table)
    for i in xrange(nbasis):
        if ntable==0:
            rep=i
            seq=i
        else:
            rep=basis_table[i]
            lb=0;ub=ntable
            result=(lb+ub)/2
            while basis_table[result]!=rep:
                if basis_table[result]>rep:
                    ub=result
                else:
                    lb=result
                if ub==lb: 
                    raise ValueError("Seq_basis error: the input basis_rep is not in the basis_table.")
                result=(lb+ub)/2

def test_while2(nbasis,basis_table):
    ntable=len(basis_table)
    for i in xrange(nbasis):
        rep=basis_rep(i,basis_table)
        seq=seq_basis(rep,basis_table)
