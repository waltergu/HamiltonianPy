'''
DMRG test.
'''

__all__=['test_dmrg']

from numpy import *
from HamiltonianPy.Basics import *
from HamiltonianPy.DMRG import *

def test_dmrg():
    print 'test_dmrg'
    a=QuantumNumberCollection([(QuantumNumber([('Sz',1,'U1')]),1),(QuantumNumber([('Sz',0,'U1')]),1),(QuantumNumber([('Sz',-1,'U1')]),1)])
    matrix=SpinMatrix((1,'z'),dtype=float64)
    print 'a:%s'%a
    print 'a+a:%s'%(a+a)
    print '--------------------'
    print '(a+a).map:'
    for key,value in (a+a).map.items():
        print '%s:%s'%(key,value)
    print '--------------------'
    print kron(matrix,matrix,a,a,a+a,format='csr')
    print
