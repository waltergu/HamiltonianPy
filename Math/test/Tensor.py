'''
Tensor test.
'''

__all__=['test_tensor']

from numpy import *
from HamiltonianPy.Math.Tensor import *

def test_tensor():
    print 'test_tensor'
    b=Tensor(random.random((2,2,3)),labels=['i','j','k'],dtype=float64)
    print "b: %s"%b
    print "b.transpose(axes=[2,1,0]): %s"%b.transpose(axes=[2,1,0])
    print "b.transpose(labels=['i','j','k']): %s"%b.transpose(labels=['i','j','k'])
    print "b.take(0,label='i'): %s"%b.take(0,label='i')
    print "b.take([0,1],axis=0): %s"%b.take([0,1],axis=0)
    print "contract(b,b): %s"%contract(b,b)
    u,s,v=b.svd(labels1=['i','j'],new='m',labels2=['k'])
    print contract(u,s,v)
    print b-contract(u,s,v)
    print
