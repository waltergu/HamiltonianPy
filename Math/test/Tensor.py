'''
Tensor test.
'''

__all__=['test_tensor']

from numpy import *
from HamiltonianPP.Math.TensorPy import *

def test_tensor():
    print 'test_tensor'
    a=Tensor([[1,1],[1,1],[1,1]],labels=['row','col'])
    print "a: %s"%a
    b=Tensor(random.random((2,2,3)),labels=['i','j','k'],dtype=float64)
    print "b: %s"%b
    print "b[0:,1,1].relabel(['i']): %s"%b[0:,1,1].relabel(['i'])
    print "b.reorder([2,1,0]): %s"%b.reorder([2,1,0])
    print "b.reorder(['i','j','k']): %s"%b.reorder(['i','j','k'])
    print "b.take(0,'i'): %s"%b.take(0,'i')
    print "b.take([0,1],0): %s"%b.take([0,1],0)
    print "concode(['m','j','k'],['j','k','l']):%s"%(concode(['m','j','k'],['j','k','l']),)
    print "contract(a,a): %s"%contract(a,a)
    u,s,v=b.svd(labels1=['i','j'],new='m',labels2=['k'])
    print contract(u,s,v)
    print b-contract(u,s,v)
    print
