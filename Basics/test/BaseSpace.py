'''
BaseSpace test.
'''

__all__=['test_basespace']

from numpy import *
from HamiltonianPy.Basics.BaseSpace import *

def test_basespace():
    print 'test_basespace'

    a=KSpace(reciprocals=[array([2*pi,0.0]),array([0.0,2*pi])],nk=100)
    print 'volume: %s'%(a.volume('k')/(2*pi)**2)
    a.plot(name='square')

    b=KSpace(reciprocals=[array([1.0,0.0]),array([0.5,sqrt(3.0)/2])],nk=100)
    print 'volume: %s'%b.volume('k')
    b.plot(name='hexagon')

    f=BaseSpace(('k',array([1,2,3,4])),('t',array([11,12,13,14])))
    for i,paras in enumerate(f('*')):
        print i,paras
    for i,paras in enumerate(f('+')):
        print i,paras

    c=FBZ(reciprocals=[array([2*pi,0.0]),array([0.0,2*pi])],nks=100)
    print 'volume: %s'%(c.volume('k')/(2*pi)**2)
    c.plot(name='square(fbz)')

    d=FBZ(reciprocals=[array([1.0,0.0]),array([0.5,sqrt(3.0)/2])],nks=100)
    print 'volume: %s'%d.volume('k')
    d.plot(name='hexagon(fbz)')
    print
