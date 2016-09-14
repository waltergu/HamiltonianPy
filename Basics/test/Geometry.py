'''
Geometry test.
'''

__all__=['test_geometry']

from numpy import *
from HamiltonianPy.Basics.Geometry import *
import time,itertools

def test_geometry():
    test_functions()
    test_point()
    test_bond()
    test_tiling()
    test_bonds()
    test_lattice()
    #test_lattice_expand_attach()
    #test_super_lattice() 

def test_functions():
    print 'test_function'
    a=array([1.0,-1.0,0.0])
    print 'azimuthd of %s: %s'%(a,azimuthd(a))
    print 'azimuth of %s: %s'%(a,azimuth(a))
    b=array([1.0,1.0,0.0])
    print 'inner of %s and %s: %s'%(a,b,inner(a,b))
    print 'cross of %s and %s: %s'%(a,b,cross(a,b))
    c=array([1.0,0.0,0.0])
    print 'volume of %s, %s and %s: %s'%(a,b,c,volume(a,b,c))
    print 'polar of %s: %s'%(c,polar(c))
    d=array([1.0,0.0,-1.0])
    print 'polard of %s: %s'%(d,polard(d))
    print 'is_parallel of %s and %s: %s'%(a,b,is_parallel(a,b))
    print

def test_point():
    print 'test_point'
    a=Point(pid=PID(site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0])
    b=Point(pid=PID(site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0])
    print '%s==%s: %s'%(b,a,b==a)
    print '%s is %s: %s'%(b,a,b is a)
    print

def test_bond():
    print 'test_bond'
    a=Bond(0,Point(pid=PID(site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0]),Point(pid=PID(site=1),rcoord=[0.0,1.0],icoord=[0.0,0.0]))
    print 'a:\n%s'%a
    print 'a.rcoord: %s'%a.rcoord
    print 'a.icoord: %s'%a.icoord
    print 'a.is_intra_cell: %s'%a.is_intra_cell()
    print

def test_tiling():
    print 'test_tiling'
    p1=Point(pid=PID(scope='WG',site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0])
    m,n=3,3
    a1=array([1.0,0.0])
    a2=array([0.0,1.0])
    supercluster,map=tiling(cluster=[p1],vectors=[a1,a2],indices=itertools.product(xrange(m),xrange(n)),return_map=True)
    print '\n'.join(['%s' for i in xrange(m*n)])%tuple(supercluster)
    for key,value in map.iteritems():
        print '%s: %s'%(key,value)
    print

def test_bonds():
    print 'test_bonds'
    p1=Point(pid=PID(site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0])
    a1,a2=array([1.0,0.0]),array([0.0,1.0])
    bs,mdists=bonds(cluster=[p1],vectors=[a1,a2],options=dict(nneighbour=4,return_min_dists=True))
    for bond in bs:
        print bond
    print 'mdists:%s'%mdists
    print

def test_lattice():
    print 'test_lattice'
    m=10;n=10
    name='L'+str(m)+str(n)
    p1=Point(pid=PID(scope=name,site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0])
    a1=array([1.0,0.0])
    a2=array([0.0,1.0])
    stime=time.time()
    a=Lattice(name,points=tiling(cluster=[p1],vectors=[a1,a2],indices=itertools.product(xrange(m),xrange(n))),vectors=[a1*m,a2*n],nneighbour=2)
    etime=time.time()
    print 'Construction time for %s*%s lattice: %s'%(m,n,etime-stime)
    a.plot(show=True)
    stime=time.time()
    b=Lattice(name,points=tiling(cluster=[p1],vectors=[a1,a2],indices=itertools.product(xrange(m),xrange(n))),nneighbour=2)
    etime=time.time()
    print 'Construction time for %s*%s cluster: %s'%(m,n,etime-stime)
    b.plot(show=True)
    print

def test_lattice_expand_attach():
    print 'test_lattice_expand_attach'
    name='WG'
    p=Point(pid=PID(scope=name,site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0])
    a1,a2=array([1.0,0.0]),array([0.0,1.0])
    for m in xrange(4):
        if m==0:
            a=Lattice(name,points=[p],vectors=[a1,a2],nneighbour=2)
            a.attach([Point(pid=PID(scope='bath',site=0),rcoord=[-0.5,-0.3],icoord=[0.0,0.0])],r=1.0)
            a.attach([Point(pid=PID(scope='bath',site=1),rcoord=[-0.5,0.3],icoord=[0.0,0.0])],r=1.0)
        else:
            bp=Point(pid=PID(scope=name,site=m),rcoord=a1*m,icoord=[0.0,0.0])
            a.expand([bp],vectors=[a1*(m+1),a2])
        a.plot(pid_on=True)
    print

def test_super_lattice():
    print 'test_super_lattice'
    m=2
    points=[None for i in xrange(4)]
    points[0]=Point(pid=PID(site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0])
    points[1]=Point(pid=PID(site=1),rcoord=[0.0,1.0],icoord=[0.0,0.0])
    points[2]=Point(pid=PID(site=2),rcoord=[1.0,0.0],icoord=[0.0,0.0])
    points[3]=Point(pid=PID(site=3),rcoord=[1.0,1.0],icoord=[0.0,0.0])
    a1=array([2.0,0.0])
    a2=array([0.0,2.0])
    a=SuperLattice(
        name='Super',
        sublattices=[Lattice(name='sub'+str(i),points=translation(points,a1*i)) for i in xrange(m)],
        vectors=[a1*m,a2],
        nneighbour=2
        )
    a.plot(pid_on=True)
    print
