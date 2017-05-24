'''
BaseSpace test.
'''

__all__=['test_basespace']

from numpy import *
from HamiltonianPy.Basics.BaseSpace import *

def test_basespace():
    test_kspace()

def test_kspace():
    print 'test_kspace'

    a=KSpace(reciprocals=[array([2*pi,0.0]),array([0.0,2*pi])],nk=100)
    print 'volume: %s'%(a.volume('k')/(2*pi)**2)
    a.plot(name='square')

    b=KSpace(reciprocals=[array([1.0,0.0]),array([0.5,sqrt(3.0)/2])],nk=100)
    print 'volume: %s'%b.volume('k')
    b.plot(name='hexagon')

    c=square_bz(reciprocals=[array([1.0,1.0]),array([1.0,-1.0])],nk=100)
    print 'volume: %s'%c.volume('k')
    c.plot(name='diamond')

    d=rectangle_bz(nk=100)
    print 'volume: %s'%(d.volume('k')/(2*pi)**2)
    d.plot(name='rectangle')

    e=hexagon_bz(nk=100,vh='v')
    print 'volume: %s'%(e.volume('k')/(2*pi)**2)
    e.plot(name='hexagon_v')

    f=hexagon_bz(nk=100,vh='h')    
    print 'volume: %s'%(f.volume('k')/(2*pi)**2)
    f.plot(name='hexagon_h')

    hexagon_gkm(nk=100).plot(name='hexagon_gkm')
    square_gxm(nk=100).plot(name='square_gxm')

    f=BaseSpace(('k',array([1,2,3,4])),('t',array([11,12,13,14])))
    for i,paras in enumerate(f('*')):
        print i,paras
    for i,paras in enumerate(f('+')):
        print i,paras
    print
