'''
BaseSpace test.
'''

__all__=['test_basespace']

from numpy import *
from HamiltonianPy.Basics.BaseSpace import *
import numpy.linalg as nl
import time

def test_basespace():
    print 'test_basespace'

    nk=100
    a1,a2=array([2*pi,0.0]),array([0.0,2*pi])
    square=KSpace(reciprocals=[a1,a2],nk=nk)
    print 'volume: %s'%(square.volume('k')/(2*pi)**2)
    square.plot(name='square')

    square=FBZ(reciprocals=[a1,a2],nks=nk)
    print 'volume: %s'%(square.volume('k')/(2*pi)**2)
    square.plot(name='square(fbz)')

    t1=time.time()
    path=square.path([(0,a1/2),(a1/2,(a1+a2)/2),((a1+a2)/2,-(a1+a2)/2),(-(a1+a2)/2,-a2/2),(-a2/2,0)])
    t2=time.time()
    print 'time,rank: %1.2fs,%s'%(t2-t1,path.rank('k'))
    path.plot(name='square(path)')
    path,indices=square.path([(0,a1/2),(a1/2,(a1+a2)/2),((a1+a2)/2,0)],mode='B')
    print 'diff: %s'%nl.norm(square.mesh('k')[indices,:]-path.mesh('k'))

    b1,b2=array([1.0,0.0]),array([0.5,sqrt(3.0)/2])
    hexagon=KSpace(reciprocals=[b1,b2],nk=nk)
    print 'volume: %s'%hexagon.volume('k')
    hexagon.plot(name='hexagon')

    hexagon=FBZ(reciprocals=[b1,b2],nks=nk)
    print 'volume: %s'%hexagon.volume('k')
    hexagon.plot(name='hexagon(fbz)')

    t1=time.time()
    path=hexagon.path([(0,b1/2),(b1/2,(b1+b2)/3),((b1+b2)/3,-(b1+b2)/3),(-(b1+b2)/3,-b2/2),(-b2/2,0)])
    t2=time.time()
    print 'time,rank: %1.2fs,%s'%(t2-t1,path.rank('k'))
    path.plot(name='hexagon(path)')

    f=BaseSpace(('k',array([1,2,3,4])),('t',array([11,12,13,14])))
    for i,paras in enumerate(f('*')):
        print i,paras
    for i,paras in enumerate(f('+')):
        print i,paras
    print
