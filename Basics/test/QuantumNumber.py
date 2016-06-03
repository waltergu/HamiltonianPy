'''
QuantumNumber test.
'''

__all__=['test_quantum_number']

from HamiltonianPy.Basics.QuantumNumber import *
from collections import OrderedDict

def test_quantum_number():
    print 'test_quantum_number'
    a=QuantumNumber([('NE',2,'U1'),('Sz',-2,'U1')])
    b=QuantumNumber([('NE',1,'U1'),('Sz',-1,'U1')])
    print 'a:%s'%(a,)
    print 'b:%s'%(b,)
    print 'a+b:%s'%(a+b,)
    c=QuantumNumber([('SP',-1,'Z2')])
    print 'c:%s'%(c,)
    d=a.direct_sum(c)
    print 'd(a.direct_sum(c)):%s'%(d,)
    e=d.replace(NE=11)
    print 'e(d.replace(NE=11)):%s'%(e,)
    print 'd+e:%s'%(d+e,)
    print
