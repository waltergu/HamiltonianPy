'''
BasisF test.
'''
__all__=['test_basisf']

from HamiltonianPy.Basics.FermionicPackage.BasisF import *
from numba import jit
import time

def test_basisf():
    print 'test_basisf'
    m,n,nloop=12,6,10
    a=BasisF(up=(m,n),down=(m,n))
    stime=time.time()
    for j in xrange(nloop):
        loop(a.table,a.nbasis)
    etime=time.time()
    print 'time consumed: %ss.'%(etime-stime)
    print

@jit
def loop(table,nbasis):
    for i in xrange(nbasis):
        rep=table[i]
        seq=sequence(rep,table)
