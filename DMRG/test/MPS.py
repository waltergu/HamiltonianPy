'''
MPS test.
'''

__all__=['test_mps']

from numpy import *
from HamiltonianPy.Basics import Label
from HamiltonianPy.Math.Tensor import *
from HamiltonianPy.DMRG.MPS import *
from copy import copy,deepcopy

def test_mps():
    print 'test_mps'
    N=3
    ms,labels=[],[]
    for i in xrange(N):
        L=Label(identifier=i)
        S=Label(identifier='S%s'%i)
        R=Label(identifier=(i+1)%N)
        labels.append((L,S,R))
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
        b._reset_(merge='L',reset=i)
        print 'b[%s].state: %s'%(i,b.state)
        print 'b[%s].norm: %s'%(i,b.norm)
        print '-------------------'

    for i in xrange(a.nsite+1):
        c=MPS.from_state(a.state,shapes=[2]*N,labels=labels,cut=i)
        #print 'c[%s]:\n%s'%(i,c)
        print 'c[%s].state: %s'%(i,c.state)
        print 'c[%s].norm: %s'%(i,c.norm)
        print 'c[%s].is_canonical: %s'%(i,c.is_canonical())
        print '-------------------'
    print
