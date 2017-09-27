'''
DQMC test.
'''

__all__=['test_dqmc']

from numpy import *
from numpy.linalg import svd,inv
from HamiltonianPy.QMC import *

def test_dqmc():
    print 'test_dqmc'
    test_B()
    test_Bs()
    print

def test_B():
    print 'test_B'
    random.seed()
    a=B(*svd(random.random((3,3))))
    print 'a:\n%s'%a.M
    b=random.random((3,3))
    print 'b:\n%s'%b
    c1,c2=b.dot(a.M),a.ldot(b)
    print 'dot(b,a):\n%s'%c1
    print 'a.ldot(b):\n%s'%c2.M
    print 'the difference:\n%s'%(c1-c2.M)
    d1,d2=a.M.dot(b),a.rdot(b)
    print 'a.dot(b):\n%s'%d1
    print 'a.rdot(b):\n%s'%d2.M
    print 'the difference:\n%s'%(d1-d2.M)
    print

def test_Bs():
    print 'test_Bs'
    random.seed()
    T=random.random((3,3))
    Vs=[random.random((3,3)) for i in xrange(3)]
    a=Bs(T,Vs,pos=2)
    a>>=Vs[1].dot(T)
    print 'a.G:\n%s'%a.G
    b1=Vs[0].dot(T)
    b2=Vs[2].dot(T).dot(Vs[1]).dot(T)
    print inv(identity(3)+b1.dot(b2))
    print
