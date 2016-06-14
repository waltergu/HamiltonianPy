'''
QuantumNumber test.
'''

__all__=['test_quantum_number']

from HamiltonianPy.Basics.QuantumNumber import *
from collections import OrderedDict
import time

def test_quantum_number():
    test_quantum_number_element()
    test_quantum_number_collection()

def test_quantum_number_element():
    print 'test_quantum_number_element'
    a=QuantumNumber([('NE',2,'U1'),('Sz',-2,'U1')])
    b=QuantumNumber([('NE',1,'U1'),('Sz',-1,'U1')])
    print 'a: %s'%(a,)
    print 'b: %s'%(b,)
    print 'a+b: %s'%(a+b,)
    c=QuantumNumber([('SP',1,'Z2')])
    print 'c: %s'%(c,)
    d=a.direct_sum(c)
    print 'd(a.direct_sum(c)): %s'%(d,)
    e=d.replace(NE=11)
    print 'e(d.replace(NE=11)): %s'%(e,)
    print 'd+e: %s'%(d+e,)
    print

from copy import copy,deepcopy
def test_quantum_number_collection():
    print 'test_quantum_number_collection'
    a=QuantumNumberCollection([(QuantumNumber([('Sz',1,'U1')]),1),(QuantumNumber([('Sz',0,'U1')]),2),(QuantumNumber([('Sz',-1,'U1')]),1)])
    print 'a: %s'%a
    print deepcopy(a)
    b=QuantumNumberCollection()
    for i in xrange(2):
        b+=a
    print 'b: ',b
    QuantumNumber.set_repr_form(QuantumNumber.repr_forms[2])
    print 'b.map:'
    for key,value in b.map.items():
        print '%s: %s'%(key,value)
    print 'subset(b,%s,%s):%s'%(b.keys()[1],b.keys()[2],b.subset(*b.keys()[1:3]))
    print
