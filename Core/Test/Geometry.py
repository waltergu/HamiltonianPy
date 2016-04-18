from numpy import *
from HamiltonianPP.Core.BasicClass.GeometryPy import *
import time,itertools
def test_geometry():
    test_functions()
    test_point()
    test_bond()
    test_tiling()
    #test_bonds()
    #test_lattice()
    #test_lattice_table()
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
    a=Point(id=ID(site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0])
    b=Point(id=ID(site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0])
    print '%s==%s: %s'%(b,a,b==a)
    print '%s is %s: %s'%(b,a,b is a)
    print

def test_bond():
    print 'test_bond'
    a=Bond(0,Point(id=ID(site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0]),Point(id=ID(site=1),rcoord=[0.0,1.0],icoord=[0.0,0.0]))
    print 'a:\n%s'%a
    print 'a.rcoord: %s'%a.rcoord
    print 'a.icoord: %s'%a.icoord
    print 'a.is_intra_cell: %s'%a.is_intra_cell()
    print

def test_tiling():
    print 'test_tiling'
    p1=Point(id=ID(scope='WG',site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0])
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
    p1=Point(id=ID(site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0])
    m,n=2,2
    a1,a2=array([1.0,0.0]),array([0.0,1.0])
    for bond in bonds(cluster=[p1],vectors=[a1,a2],nneighbour=2):
        print bond

def test_lattice():
    p1=Point(id=0,rcoord=[0.0,0.0],icoord=[0.0,0.0])
    a1=array([1.0,0.0])
    a2=array([0.0,1.0])
    m=2;n=2
    stime=time.time()
    a=Lattice('L'+str(m)+str(n),[p1],translations=((a1,m),(a2,n)),vectors=[a1*m,a2*n],nneighbour=2)
    etime=time.time()
    print etime-stime
    for p in a.points:
        print p
    for bond in a.bonds:
        print bond
    a.plot(show=True)
    stime=time.time()
    b=Lattice('C'+str(m)+str(n),[p1],translations=((a1,m),(a2,n)),nneighbour=2)
    etime=time.time()
    print etime-stime
    b.plot(show=True)

def test_lattice_table():
    p1=Point(site=0,rcoord=[0.0,0.0],icoord=[0.0,0.0],struct=Fermi(norbital=2,nspin=2,nnambu=2))
    p2=Point(site=1,rcoord=[1.0,0.0],icoord=[0.0,0.0],struct=Fermi(norbital=2,nspin=2,nnambu=2))
    a=Lattice('C',[p1,p2],nneighbour=2)
    print a.table(nambu=True)

def test_super_lattice():
    m=4
    points=[None for i in xrange(4)]
    points[0]=Point(site=0,rcoord=[0.0,0.0],icoord=[0.0,0.0],struct=Fermi(norbital=1,nspin=2,nnambu=1))
    points[1]=Point(site=1,rcoord=[0.0,1.0],icoord=[0.0,0.0],struct=Fermi(norbital=1,nspin=2,nnambu=1))
    points[2]=Point(site=2,rcoord=[1.0,0.0],icoord=[0.0,0.0],struct=Fermi(norbital=1,nspin=2,nnambu=1))
    points[3]=Point(site=3,rcoord=[1.0,1.0],icoord=[0.0,0.0],struct=Fermi(norbital=1,nspin=2,nnambu=1))
    a1=array([2.0,0.0])
    a2=array([0.0,2.0])
    a=SuperLattice(
        name='Super',
        sublattices=[Lattice(name='sub'+str(i),points=translation(points,a1*i)) for i in xrange(m)],
        vectors=[a1*m,a2],
        nneighbour=2
        )
    a.plot()
    print a.table()
    for lattice in a.sublattices:
        print lattice.table()
