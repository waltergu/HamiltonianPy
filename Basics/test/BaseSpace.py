'''
BaseSpace test.
'''

__all__=['test_basespace']

from numpy import *
from HamiltonianPP.Basics.BaseSpacePy import *

def test_basespace():
    test_kspace()

def test_kspace():
    print 'test_kspace'

    a=KSpace(reciprocals=[array([2*pi,0.0]),array([0.0,2*pi])],nk=100)
    print 'a.volume: %s'%(a.volume['k']/(2*pi)**2)
    a.plot(show=True)

    b=KSpace(reciprocals=[array([1.0,0.0]),array([0.5,sqrt(3.0)/2])],nk=100)
    print 'b.volume: %s'%b.volume
    b.plot(show=True)

    c=square_bz(reciprocals=[array([1.0,1.0]),array([1.0,-1.0])],nk=100)
    print 'c.volume: %s'%c.volume
    c.plot(show=True)

    d=rectangle_bz(nk=100)
    print 'd.volume: %s'%(d.volume['k']/(2*pi)**2)
    d.plot(show=True)

    e=hexagon_bz(nk=100,vh='v')
    print 'e.volume: %s'%(a.volume['k']/(2*pi)**2)
    e.plot(show=True)

    hexagon_gkm(nk=100).plot(show=True)
    square_gxm(nk=100).plot()

    f=BaseSpace(dict(tag='k',mesh=array([1,2,3,4])),{'tag':'t','mesh':array([11,12,13,14])})
    for i,paras in enumerate(f('*')):
        print i,paras
    for i,paras in enumerate(f('+')):
        print i,paras
    print
