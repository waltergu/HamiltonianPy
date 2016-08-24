'''
DQMC test.
'''

__all__=['test_dqmc']

from numpy import *
from numpy.linalg import svd
from HamiltonianPy.QMC import *

def test_dqmc():
    print 'test_dqmc'
    a=BMatrix(*svd(random.random((3,3))))
    print 'a:\n%s'%a.M
    b=random.random((3,3))
    print 'b:\n%s'%b
    print 'b.dot(a):\n%s'%b.dot(a.M)
    c=dot(b,a)
    print 'dot(b,a):\n%s'%c.M
    print 'c=dot(b,a):\n%s'%c
    print 
