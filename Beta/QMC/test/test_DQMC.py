'''
DQMC test.
'''

__all__=['dqmc']

import numpy as np
from numpy.linalg import svd,inv
from HamiltonianPy.Beta.QMC import *
from unittest import TestCase,TestLoader,TestSuite

class TestDQMC(TestCase):
    def test_B(self):
        print()
        np.random.seed()
        a=B(*svd(np.random.random((3,3))))
        print('a:\n%s'%a.M)
        b=np.random.random((3,3))
        print('b:\n%s'%b)
        c1,c2=b.dot(a.M),a.ldot(b)
        print('dot(b,a):\n%s'%c1)
        print('a.ldot(b):\n%s'%c2.M)
        print('the difference:\n%s'%(c1-c2.M))
        d1,d2=a.M.dot(b),a.rdot(b)
        print('a.dot(b):\n%s'%d1)
        print('a.rdot(b):\n%s'%d2.M)
        print('the difference:\n%s'%(d1-d2.M))

    def test_Bs(self):
        print()
        np.random.seed()
        T=np.random.random((3,3))
        Vs=[np.random.random((3,3)) for _ in range(3)]
        a=Bs(T,Vs,pos=2)
        a>>=Vs[1].dot(T)
        print('a.G:\n%s'%a.G)
        b1=Vs[0].dot(T)
        b2=Vs[2].dot(T).dot(Vs[1]).dot(T)
        print(inv(np.identity(3)+b1.dot(b2)))

dqmc=TestSuite([
            TestLoader().loadTestsFromTestCase(TestDQMC),            
            ])

if __name__=='__main__':
    from unittest import main
    main(verbosity=2)