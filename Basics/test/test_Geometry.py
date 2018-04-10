'''
Geometry test (16 tests in total).
'''

__all__=['geometry']

from HamiltonianPy.Basics.Geometry import *
from unittest import TestCase,TestLoader,TestSuite
import numpy as np
import numpy.linalg as nl
import itertools as it
import time

class TestFunctions(TestCase):
    def test_azimuth(self):
        a=np.array([1.0,-1.0,0.0])
        self.assertEqual(azimuthd(a),315.0)
        self.assertEqual(azimuth(a),np.pi*7/4)

    def test_polar(self):
        a=np.array([1.0,0.0,-1.0])
        self.assertEqual(polard(a),135.0)
        self.assertEqual(polar(a),np.pi*3/4)

    def test_volume(self):
        a,b,c=np.array([1.0,-1.0,0.0]),np.array([1.0,1.0,0.0]),np.array([0.0,0.0,1.0])
        self.assertEqual(volume(a,b,c),2.0)

    def test_isparallel(self):
        a,b=np.array([1.0,-1.0,0.0]),np.array([1.0,1.0,0.0])
        self.assertEqual(isparallel(a,b),0)
        self.assertEqual(isparallel(a+b,a+b),+1)
        self.assertEqual(isparallel(a-b,b-a),-1)

    def test_issubordinate(self):
        e,f1,f2=np.array([1.0,1.0]),np.array([1.0,0.0]),np.array([0.0,1.0])
        self.assertTrue(issubordinate(e,[f1,f2]))

    def test_tiling(self):
        m,n=3,4
        cluster=[np.array([0.0,0.0])]
        a1,a2=np.array([1.0,0.0]),np.array([0.0,1.0])
        supercluster=tiling(cluster=cluster,vectors=[a1,a2],translations=it.product(xrange(m),xrange(n)))
        for i,coord in enumerate(supercluster):
            self.assertEqual(nl.norm(a1*(i/n)+a2*(i%n)-coord),0.0)

    def test_minimumlengths(self):
        point,a1,a2=np.array([0.0,0.0]),np.array([1.0,0.0]),np.array([0.0,1.0])
        lengths=minimumlengths(cluster=[point],vectors=[a1,a2],nneighbour=3)
        result=np.array([0.0,1.0,np.sqrt(2.0),2.0])
        self.assertEqual(nl.norm(lengths-result),0.0)

class TestPoint(TestCase):
    def setUp(self):
        self.p1=Point(pid=PID(site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0])
        self.p2=Point(pid=PID(site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0])

    def test_eq(self):
        self.assertTrue(self.p1==self.p2)

class TestBond(TestCase):
    def setUp(self):
        self.bond=Bond(0,Point(pid=PID(site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0]),Point(pid=PID(site=1),rcoord=[0.0,1.0],icoord=[0.0,0.0]))

    def test_rcoord(self):
        result=np.array([0.0,1.0])
        self.assertEqual(nl.norm(self.bond.rcoord-result),0.0)

    def test_icoord(self):
        result=np.array([0.0,0.0])
        self.assertEqual(nl.norm(self.bond.icoord-result),0.0)

class TestLink(TestCase):
    def setUp(self):
        self.cluster=[np.array([0.0,0.0])]
        self.vectors=[np.array([1.0,0.0]),np.array([0.0,1.0])]
        self.neighbours={i:length for i,length in enumerate(minimumlengths(cluster=self.cluster,vectors=self.vectors,nneighbour=3))}

    def test_intralinks(self):
        print
        links=intralinks(cluster=self.cluster,vectors=self.vectors,neighbours=self.neighbours)
        print '\n'.join([str(link) for link in links])

    def test_interlinks(self):
        print
        links=interlinks([np.array([0.0,0.0])],[np.array([0.0,1.0])],neighbours=self.neighbours)
        print '\n'.join([str(link) for link in links])

class TestLattice(TestCase):
    def setUp(self):
        self.point=np.array([0.0,0.0])
        self.a1=np.array([1.0,0.0])
        self.a2=np.array([0.0,1.0])

    def name(self,m,n):
        return 'L%s%s'%(m,n)

    def rcoords(self,m,n):
        return tiling(cluster=[self.point],vectors=[self.a1,self.a2],translations=it.product(xrange(m),xrange(n)))

    def vectors(self,m,n):
        return [self.a1*m,self.a2*n]

    def test_periodic(self):
        print
        m,n=10,10
        stime=time.time()
        lattice=Lattice('%s_P'%self.name(m,n),rcoords=self.rcoords(m,n),vectors=self.vectors(m,n),neighbours=2)
        etime=time.time()
        print 'time(%s): %s'%(lattice.name,etime-stime)
        lattice.plot(show=True,pidon=False,suspend=False)

    def test_open(self):
        print
        m,n=10,10
        stime=time.time()
        lattice=Lattice('%s_O'%self.name(m,n),rcoords=self.rcoords(m,n),neighbours=2)
        etime=time.time()
        print 'time(%s): %s'%(lattice.name,etime-stime)
        lattice.plot(show=True,pidon=False,suspend=False)

    def test_merge(self):
        M,m,n=2,2,2
        lattice=Lattice.merge(
            name=           'Merge',
            sublattices=    [Lattice(name='sub%s'%i,rcoords=translation(self.rcoords(m,n),self.a1*m*i)) for i in xrange(M)],
            vectors=        [self.a1*m*M,self.a2*n],
            neighbours=     2
        )
        lattice.plot(pidon=True)

class TestSuperLattice(TestCase):
    def setUp(self):
        self.name='WG'
        self.a1=np.array([1.0,0.0])
        self.a2=np.array([0.0,1.0])
        self.lattice=Lattice(name='bath',rcoords=[np.array([-0.4,-0.3]),np.array([-0.4,+0.3])],neighbours=0)

    def test_superlattice(self):
        N=4
        for n in xrange(N):
            self.lattice=SuperLattice(
                name=           'Super',
                sublattices=    [self.lattice,Lattice('%s-%s'%(self.name,n),rcoords=[self.a1*n],vectors=[self.a2])],
                neighbours=     {0:0.0,1:1.0,-1:0.5}
            )
            self.lattice.plot(pidon=True,suspend=False)

geometry=TestSuite([
                    TestLoader().loadTestsFromTestCase(TestFunctions),
                    TestLoader().loadTestsFromTestCase(TestPoint),
                    TestLoader().loadTestsFromTestCase(TestBond),
                    TestLoader().loadTestsFromTestCase(TestLink),
                    TestLoader().loadTestsFromTestCase(TestLattice),
                    TestLoader().loadTestsFromTestCase(TestSuperLattice),
                    ])

