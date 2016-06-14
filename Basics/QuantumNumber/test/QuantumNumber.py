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
    test_quantum_number_time()

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
    print 'copy(a): %s'%copy(a)
    print 'deepcopy(a): %s'%deepcopy(a)

    b=QuantumNumberCollection()
    for i in xrange(2):
        b+=a
    print 'b: ',b
    print 'b.map:'
    for key,value in b.map.items():
        print '%s: %s'%(key,value)
    print 'b.slices:'
    for key,value in b.slices.items():
        print '%s: %s'%(key,value)
    print 'b.permutation:%s'%b.permutation

    c=b.subset(*b.keys()[1:3])
    print 'c(b.subset(%s,%s)):%s'%(b.keys()[1],b.keys()[2],c)
    print 'c.map:'
    for key,value in c.map.items():
        print '%s: %s'%(key,value)
    print 'c.slices:'
    for key,value in c.slices.items():
        print '%s: %s'%(key,value)
    print 'c.permutation:%s'%(c.permutation)
    print

import time
def test_quantum_number_time():
    print 'test_quantum_number_time'
    N=6
    a=QuantumNumberCollection([(QuantumNumber([('Sz',1,'U1')]),1),(QuantumNumber([('Sz',0,'U1')]),2),(QuantumNumber([('Sz',-1,'U1')]),1)])
    b=QuantumNumberCollection()
    t1=time.time()
    for i in xrange(N):
        b+=a
    t2=time.time()
    print 'Summation form 1 to %s: %ss.'%(N,t2-t1)
    t3=time.time()
    c=b+b
    t4=time.time()
    print 'Summation of %s and %s: %ss.'%(N,N,t4-t3)
    print
