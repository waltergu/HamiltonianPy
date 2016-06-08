'''
Spin degree of freedom test.
'''

__all__=['test_spin_deg_fre']

from HamiltonianPy.Basics.Geometry import *
from HamiltonianPy.Basics.DegreeOfFreedom import *
from HamiltonianPy.Basics.SpinPackage import *
from numpy import *

def test_spin_deg_fre():
    test_sid()
    test_spin()
    test_spin_matrix()
    test_spin_pack()

def test_sid():
    print 'test_sid'
    print SID(S=2)
    print

def test_spin():
    print 'test_spin'
    a=Spin(S=2)
    print 'a: %s'%a
    print 'a.table:%s'%a.table(PID(scope='WG',site=0))
    print

def test_spin_matrix():
    print 'test_spin_matrix'
    N=1
    print SpinMatrix((N,'x'),dtype=float64)
    print SpinMatrix((N,'y'),dtype=complex128)
    print SpinMatrix((N,'z'),dtype=float64)
    print SpinMatrix((N,'+'),dtype=float64)
    print SpinMatrix((N,'-'),dtype=float64)
    print

def test_spin_pack():
    print 'test_spin_pack'
    a=SpinPack(1.0,pack=('x','x'))
    print 'a: %s'%a
    print 'a*2: %s'%(a*2)
    print '2*a: %s'%(2*a)
    print 'Heisenberg*2: %s'%(2*Heisenberg())
    print
