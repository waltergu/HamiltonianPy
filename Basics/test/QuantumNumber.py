'''
QuantumNumber test.
'''

__all__=['test_quantum_number']

from HamiltonianPy.Basics.QuantumNumber import *
from collections import OrderedDict

def test_quantum_number():
    print 'test_quantum_number'
    a=U1(OrderedDict(NE=2,Sz=-2))
    b=U1(OrderedDict(NE=1,Sz=-1))
    print 'a:%s'%(a,)
    print 'b:%s'%(b,)
    print 'a+b:%s'%(a+b,)
    print 'a*2:%s'%(a*2,)
    c=U1(OrderedDict(S=11))
    print 'c:%s'%(c,)
    print 'b.direct_sum(c):%s'%(b.direct_sum(c),)
    print
