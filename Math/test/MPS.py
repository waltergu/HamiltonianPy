'''
MPS test.
'''

__all__=['test_mps']

from numpy import *
from HamiltonianPP.Math.TensorPy import *
from HamiltonianPP.Math.MPSPy import *

def test_mps():
    print 'test_mps'
    N=4
    ms,labels=[],[]
    for i in xrange(N):
        labels.append(('S%s'%i,'B%s'%i))
        ms.append(array([[[1,0],[0,1]],[[1,0],[0,1]]]))
    a=GMPS(ms,labels)
    print 'a:\n%s'%a
    print 'a.state: %s'%a.state
    b=GMPS([array([[[1,0],[0,1]]]),array([[[0],[1]],[[1],[0]]])],labels=[('S0','B0'),('S1','B1')])
    print 'b:\n%s'%b
    print 'b.state: %s'%b.state
    print
    
    c=b.to_vmps()
    print "b:%s"%b
    print b.state
    print c.state
