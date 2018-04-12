'''
Tensor test (17 tests in total).
'''

__all__=['tensor']

import numpy as np
import numpy.linalg as nl
from HamiltonianPy import *
from HamiltonianPy.TensorNetwork import *
from time import time
from unittest import TestCase,TestLoader,TestSuite

class TestDTensor(TestCase):
    def test_transpose(self):
        axes=[1,2,0]
        transpose=self.tensor.transpose(axes)
        result=Tensor(self.tensor.data.transpose(axes),labels=[self.tensor.labels[axis] for axis in axes])
        self.assertTrue((transpose==result).all())

    def test_svd(self):
        u1,s1,v1=svd(self.tensor,row=[0,1],new=Label('new',None,None),col=[2])
        u2,s2,v2=svd(self.tensor,row=[0],new=Label('new',None,None),col=[1,2])
        self.assertAlmostEqual((u1*s1*v1-self.tensor).norm,0.0)
        self.assertAlmostEqual((u2*s2*v2-self.tensor).norm,0.0)

    def test_expanded_svd(self):
        L,S,R=self.tensor.labels
        E=[Label('S%s'%i,qns=self.qns,flow=S.flow) for i in xrange(self.s)]
        for cut in xrange(self.s):
            I=[Label('B%i'%i,None,None) for i in xrange(self.s if cut in (0,self.s) else self.s-1)]
            ms=expanded_svd(self.tensor,L=[L],S=S,R=[R],E=E,I=I,cut=cut)
            ms=[ms[0],ms[1]]+ms[2] if cut==0 else ms[0]+[ms[1],ms[2]] if cut==4 else ms[0][:cut]+[ms[1]]+ms[0][cut:]
            self.assertAlmostEqual((np.product(ms).merge((E,S))-self.tensor).norm,0.0)

    def test_directsum(self):
        L,S,R=self.tensor.labels
        m1=directsum([self.tensor,self.tensor],[Label('l',None,None),Label('s',None,None),Label('r',None,None)],axes=[0])
        m2=directsum([self.tensor,self.tensor],[Label('l',None,None),Label('s',None,None),Label('r',None,None)],axes=[1])
        m3=directsum([self.tensor,self.tensor],[Label('l',None,None),Label('s',None,None),Label('r',None,None)],axes=[2])
        self.assertEqual(m1.shape,(L.dim,2*S.dim,2*R.dim))
        self.assertEqual(m2.shape,(2*L.dim,S.dim,2*R.dim))
        self.assertEqual(m3.shape,(2*L.dim,2*S.dim,R.dim))

    def test_deparallelization(self):
        m1=directsum([self.tensor,self.tensor],[Label('l',None),Label('s',None),Label('r',None)],axes=[0])
        m2=directsum([self.tensor,self.tensor],[Label('l',None),Label('s',None),Label('r',None)],axes=[2])
        T1R,M1R=deparallelization(m1,row=[0],new=Label('new',None),col=[1,2],mode='R')
        T1C,M1C=deparallelization(m1,row=[0],new=Label('new',None),col=[1,2],mode='C')
        T2R,M2R=deparallelization(m2,row=[0,1],new=Label('new',None),col=[2],mode='R')
        T2C,M2C=deparallelization(m2,row=[0,1],new=Label('new',None),col=[2],mode='C')
        self.assertAlmostEqual((T1R*M1R-m1).norm,0.0)
        self.assertAlmostEqual((T1C*M1C-m1).norm,0.0)
        self.assertAlmostEqual((T2R*M2R-m2).norm,0.0)
        self.assertAlmostEqual((T2C*M2C-m2).norm,0.0)

class TestNBDTensor(TestDTensor):
    def setUp(self):
        self.QNS,self.l,self.s,self.r=SQNS(0.5),1,4,5
        L=Label('l',qns=self.qns**self.l,flow=0)
        S=Label('s',qns=self.qns**self.s,flow=0)
        R=Label('r',qns=self.qns**self.r,flow=0)
        self.tensor=random([L,S,R])

    @property
    def qns(self):
        return len(self.QNS)

    def test_qngenerate(self):
        print
        lqns,sqns,rqns=self.QNS**self.l,self.QNS**self.s,self.QNS**self.r
        tensor=random([Label('l',lqns,+1),Label('s',sqns,+1),Label('r',rqns,-1)])
        tensor.relabel(self.tensor.labels)
        stime=time()
        tensor.qngenerate(-1,axes=[0,1],qnses=[lqns,sqns],flows=[+1,+1])
        tensor.qngenerate(+1,axes=[1,2],qnses=[sqns,rqns],flows=[+1,-1])
        tensor.qngenerate(+1,axes=[2,0],qnses=[rqns,lqns],flows=[-1,+1])
        etime=time()
        print 'qngenerate time in total: %ss'%(etime-stime)

class TestQNDTensor(TestDTensor):
    def setUp(self):
        self.qns,self.l,self.s,self.r=SQNS(0.5),1,4,5
        L=Label('l',qns=self.qns**self.l,flow=+1)
        S=Label('s',qns=self.qns**self.s,flow=+1)
        R=Label('r',qns=self.qns**self.r,flow=-1)
        self.tensor=random([L,S,R])

    def test_merge_and_split(self):
        L,S,R=self.tensor.labels
        LS,lspermutation=Label.union([L,S],'LS',flow=+1,mode=1)
        SR,srpermutation=Label.union([S,R],'SR',flow=-1,mode=1)
        merge1=self.tensor.merge(([L,S],LS,lspermutation))
        merge2=self.tensor.merge(([S,R],SR,srpermutation))
        r1=Tensor(self.tensor.data.reshape((-1,R.dim))[lspermutation,:],labels=[LS,R])
        r2=Tensor(self.tensor.data.reshape((L.dim,-1))[:,srpermutation],labels=[L,SR])
        self.assertAlmostEqual((merge1-r1).norm,0.0)
        self.assertAlmostEqual((merge2-r2).norm,0.0)
        split1=merge1.split((LS,[L,S],np.argsort(lspermutation)))
        split2=merge2.split((SR,[S,R],np.argsort(srpermutation)))
        self.assertAlmostEqual((split1-self.tensor).norm,0.0)
        self.assertAlmostEqual((split2-self.tensor).norm,0.0)

class TestSTensor(TestCase):
    def setUp(self):
        self.qns,self.l,self.s,self.r=SQNS(0.5),1,4,5
        L=Label('l',qns=(self.qns**self.l).sorted(history=False),flow=+1)
        S=Label('s',qns=(self.qns**self.s).sorted(history=False),flow=+1)
        R=Label('r',qns=(self.qns**self.r).sorted(history=False),flow=-1)
        self.stensor=random([L,S,R],mode='S')
        self.dtensor=self.stensor.todtensor()

    def test_merge_and_split(self):
        sL,sS,sR=self.stensor.labels
        sLS,slsrecord=Label.union([sL,sS],'LS',flow=+1,mode=2)
        sSR,ssrrecord=Label.union([sS,sR],'SR',flow=-1,mode=2)
        merge1=self.stensor.merge(([sL,sS],sLS,slsrecord))
        merge2=self.stensor.merge(([sS,sR],sSR,ssrrecord))
        dL,dS,dR=self.dtensor.labels
        dLS,dlspermutation=Label.union([dL,dS],'LS',flow=+1,mode=1)
        dSR,dsrpermutation=Label.union([dS,dR],'SR',flow=-1,mode=1)
        r1=self.dtensor.merge(([dL,dS],dLS,dlspermutation)).tostensor()
        r2=self.dtensor.merge(([dS,dR],dSR,dsrpermutation)).tostensor()
        self.assertAlmostEqual((merge1-r1).norm,0.0)
        self.assertAlmostEqual((merge2-r2).norm,0.0)
        split1=merge1.split((sLS,[sL,sS],slsrecord))
        split2=merge2.split((sSR,[sS,sR],ssrrecord))
        self.assertAlmostEqual((split1-self.stensor).norm,0.0)
        self.assertAlmostEqual((split2-self.stensor).norm,0.0)

    def test_transpose(self):
        axes=[1,2,0]
        stranspose=self.stensor.transpose(axes)
        dtranspose=self.dtensor.transpose(axes)
        self.assertAlmostEqual((stranspose.todtensor()-dtranspose).norm,0.0)

    def test_svd(self):
        us1,ss1,vs1=svd(self.stensor,row=[0,1],new=Label('new',None,None),col=[2])
        us2,ss2,vs2=svd(self.stensor,row=[0],new=Label('new',None,None),col=[1,2])
        self.assertAlmostEqual((us1*ss1*vs1-self.stensor).norm,0.0)
        self.assertAlmostEqual((us2*ss2*vs2-self.stensor).norm,0.0)
        sd1=svd(self.dtensor,row=[0,1],new=Label('new',None,None),col=[2])[1]
        sd2=svd(self.dtensor,row=[0],new=Label('new',None,None),col=[1,2])[1]
        ss1.qnsort()
        ss2.qnsort()
        sd1.qnsort()
        sd2.qnsort()
        self.assertAlmostEqual((ss1-sd1).norm,0.0)
        self.assertAlmostEqual((ss2-sd2).norm,0.0)

    def test_directsum(self):
        L,S,R=self.stensor.labels
        sm1=directsum([self.stensor,self.stensor],[Label('l',None,None),Label('s',None,None),Label('r',None,None)],axes=[0])
        sm2=directsum([self.stensor,self.stensor],[Label('l',None,None),Label('s',None,None),Label('r',None,None)],axes=[1])
        sm3=directsum([self.stensor,self.stensor],[Label('l',None,None),Label('s',None,None),Label('r',None,None)],axes=[2])
        self.assertEqual(sm1.shape,(L.dim,2*S.dim,2*R.dim))
        self.assertEqual(sm2.shape,(2*L.dim,S.dim,2*R.dim))
        self.assertEqual(sm3.shape,(2*L.dim,2*S.dim,R.dim))
        dm1=directsum([self.dtensor,self.dtensor],[Label('l',None,None),Label('s',None,None),Label('r',None,None)],axes=[0])
        dm2=directsum([self.dtensor,self.dtensor],[Label('l',None,None),Label('s',None,None),Label('r',None,None)],axes=[1])
        dm3=directsum([self.dtensor,self.dtensor],[Label('l',None,None),Label('s',None,None),Label('r',None,None)],axes=[2])
        dm1.qnsort()
        dm2.qnsort()
        dm3.qnsort()
        self.assertAlmostEqual((dm1.tostensor()-sm1).norm,0.0)
        self.assertAlmostEqual((dm2.tostensor()-sm2).norm,0.0)
        self.assertAlmostEqual((dm3.tostensor()-sm3).norm,0.0)

class Test_contract(TestCase):
    def test_contract(self):
        print
        N,nmax=50,400
        stime=time()
        mps=MPS.random(sites=[SPQNS(0.5)]*N,bonds=[SPQN((0.0,0.0)),SPQN((N,0.0))],cut=N/2,nmax=nmax)
        m1,m2,m3=mps[N/2-1],mps[N/2],mps[N/2+1]
        sm1,sm2,sm3=m1.tostensor(),m2.tostensor(),m3.tostensor()
        print 'prepare time: %ss'%(time()-stime)
        print 'tensor shape: %s, %s, %s'%(m1.shape,m2.shape,m3.shape)
        stime=time()
        contraction1=m1*(m2,'einsum')*(m3,'einsum')
        print 'einsum time: %ss'%(time()-stime)
        stime=time()
        contraction2=m1*(m2,'tensordot')*(m3,'tensordot')
        print 'tensordot time: %ss.'%(time()-stime)
        stime=time()
        contraction3=m1*(m2,'block')*(m3,'block')
        print 'block time: %ss.'%(time()-stime)
        stime=time()
        contraction4=sm1*sm2*sm3
        print 'sparse time: %ss.'%(time()-stime)
        contraction4=contraction4.todtensor()
        self.assertAlmostEqual((contraction1-contraction2).norm,0.0)
        self.assertAlmostEqual((contraction2-contraction3).norm,0.0)
        self.assertAlmostEqual((contraction3-contraction4).norm,0.0)
        self.assertAlmostEqual((contraction4-contraction1).norm,0.0)

tensor=TestSuite([
            TestLoader().loadTestsFromTestCase(TestNBDTensor),
            TestLoader().loadTestsFromTestCase(TestQNDTensor),
            TestLoader().loadTestsFromTestCase(TestSTensor),
            TestLoader().loadTestsFromTestCase(Test_contract),
            ])
