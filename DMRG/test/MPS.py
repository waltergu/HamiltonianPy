'''
MPS test.
'''

__all__=['test_mps']

from numpy import *
from HamiltonianPy.Math.Tensor import *
from HamiltonianPy.DMRG.MPS import *
from copy import copy,deepcopy

def test_mps():
    print 'test_mps'
    N=3
    ms,labels=[],[]
    for i in xrange(N):
        labels.append(('END' if i==0 else 'B%s'%(i-1),'S%s'%i,'END' if i==N-1 else 'B%s'%i))
        if i==0:
            ms.append(array([[[1,0],[0,1]]]))
        elif i==N-1:
            ms.append(array([[[0],[1]],[[1],[0]]]))
        else:
            ms.append(array([[[1,0],[0,1]],[[1,0],[0,1]]]))
    a=MPS(ms,labels)
    #print 'a:\n%s'%a
    print 'a.state: %s'%a.state
    print 'a.norm: %s'%a.norm
    print '-------------------'

    for i in xrange(a.nsite+1):
        b=deepcopy(a)
        b.canonicalization(cut=i)
        #print 'b[%s]:\n%s'%(i,b)
        print 'b[%s].state: %s'%(i,b.state)
        print 'b[%s].norm: %s'%(i,b.norm)
        print 'b[%s].is_canonical: %s'%(i,b.is_canonical())
        print '-------------------'
    print