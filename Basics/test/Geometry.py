'''
Geometry test.
'''

__all__=['test_geometry']

from numpy import *
from HamiltonianPy.Basics.Geometry import *
import time,itertools

def test_geometry():
    test_functions()
    test_tiling()
    test_point()
    test_bond()
    test_link()
    test_lattice()
    test_super_lattice_merge()
    test_super_lattice_union()

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
    print 'isparallel of %s and %s: %s'%(a,b,isparallel(a,b))
    e=array([1.0,1.0])
    f1,f2=array([1.0,0.0]),array([0.0,1.0])
    print 'rcoord: %s'%e
    print 'vectors: %s, %s'%(f1,f2)
    print 'issubordinate(rcoord,vectors): %s'%issubordinate(e,[f1,f2])
    print

def test_tiling():
    print 'test_tiling'
    cluster=[array([0.0,0.0])]
    m,n=3,3
    a1=array([1.0,0.0])
    a2=array([0.0,1.0])
    supercluster=tiling(cluster=cluster,vectors=[a1,a2],translations=itertools.product(xrange(m),xrange(n)))
    print '\n'.join(['%s: %s'%(i,coord) for i,coord in enumerate(supercluster)])

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
    print 'a.isintracell: %s'%a.isintracell()
    print

def test_link():
    print 'test_link'
    print "intralinks with mode='nb':"
    a1,a2=array([1.0,0.0]),array([0.0,1.0])
    links,mindists=intralinks('nb',cluster=[array([0.0,0.0])],vectors=[a1,a2],nneighbour=3,return_mindists=True)
    print '\n'.join([str(link) for link in links])
    print 'mindists: %s'%', '.join(str(mindist) for mindist in mindists)
    print
    print "intralinks with mode='nb':"
    links=intralinks('dt',cluster=[array([0.0,0.0])],vectors=[a1,a2],r=2.0,mindists=mindists)
    print '\n'.join([str(link) for link in links])
    print
    print 'interlinks'
    links=interlinks([array([0.0,0.0])],[array([0.0,1.0])],maxdist=2.0,mindists=mindists)
    print '\n'.join([str(link) for link in links])
    print

def test_lattice():
    print 'test_lattice'
    m=10;n=10
    name='L'+str(m)+str(n)
    a1,a2=array([1.0,0.0]),array([0.0,1.0])
    rcoords=tiling(cluster=[array([0.0,0.0])],vectors=[a1,a2],translations=itertools.product(xrange(m),xrange(n)))
    stime=time.time()
    a=Lattice(name,rcoords,vectors=[a1*m,a2*n],nneighbour=2)
    etime=time.time()
    print 'Construction time for %s*%s lattice: %s'%(m,n,etime-stime)
    a.plot(show=True,pidon=False,suspend=False)
    stime=time.time()
    b=Lattice(name,rcoords,nneighbour=2)
    etime=time.time()
    print 'Construction time for %s*%s cluster: %s'%(m,n,etime-stime)
    b.plot(show=True,pidon=False,suspend=False)
    c=Lattice('WG',rcoords=[array([0.0,0.0])],vectors=[a1,a2],nneighbour=2)
    c.plot(show=True,pidon=True,suspend=False)
    print

def test_super_lattice_merge():
    print 'test_super_lattice_merge'
    M,m,n=2,2,2
    a1,a2=array([1.0,0.0]),array([0.0,1.0])
    rcoords=tiling(cluster=[array([0.0,0.0])],vectors=[a1,a2],translations=itertools.product(xrange(m),xrange(n)))
    a=SuperLattice.merge(
        name='Super',
        sublattices=[Lattice(name='sub'+str(i),rcoords=translation(rcoords,a1*m*i)) for i in xrange(M)],
        vectors=[a1*m*M,a2*n],
        nneighbour=2
        )
    a.plot(pidon=True)
    print

def test_super_lattice_union():
    print 'test_super_lattice_union'
    N=4
    a=Lattice(name='bath',rcoords=[array([-0.5,-0.3]),array([-0.5,+0.3])],nneighbour=0)
    name,a1,a2='WG',array([1.0,0.0]),array([0.0,1.0])
    for m in xrange(N):
        b=Lattice('%s-%s'%(name,m),rcoords=[a1*m],vectors=[a2])
        a=SuperLattice.union(
            name=           'super',
            sublattices=    [a,b],
            maxdist=        1.0,
            mindists=       [0.0,1.0]
            )
        a.plot(pidon=True,suspend=False)
    print
