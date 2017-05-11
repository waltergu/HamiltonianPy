'''
FBasis test.
'''
__all__=['test_fbasis']

from HamiltonianPy.Basics.FermionicPackage.Basis import *
from numba import jit
import time

def test_fbasis():
    print 'test_fbasis'
    m,n,nloop=12,6,10
    a=FBasis(up=(m,n),down=(m,n))
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
