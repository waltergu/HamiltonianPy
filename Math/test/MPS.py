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
        labels.append(('B%s'%(i-1 if i>0 else N-1),'S%s'%i,'B%s'%i))
        ms.append(array([[[1,0],[0,1]],[[1,0],[0,1]]]))
    a=GMPS(ms,labels)
    print 'a:\n%s'%a
    print 'a.state: %s'%a.state
    b=GMPS([array([[[1,0],[0,1]]]),array([[[0],[1]],[[1],[0]]])],labels=[('B1','S0','B0'),('B0','S1','B1')])
    print 'b:\n%s'%b
    print 'b.state: %s'%b.state
    print
    
    c=b.to_vmps()
    print "b:%s"%b
    print b.state
    print b.norm
    print c.state
    print
    
    for i in xrange(2):
        d=c.to_mmps(i)
        print 'd[%s]:%s'%(i,d)
        print 'state:%s'%d.state
        print
