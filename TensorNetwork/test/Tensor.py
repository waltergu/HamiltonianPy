'''
Tensor test.
'''

__all__=['test_tensor']

import numpy as np
from HamiltonianPy import *
from HamiltonianPy.TensorNetwork import *
from time import time

def test_tensor():
    print 'test_tensor'
    test_tensor_ordinary()
    test_tensor_qngenerate()
    test_tensor_svd()
    test_tensor_directsum()
    test_tensor_expanded_svd()
    test_tensor_deparallelization()
    test_tensor_contract()

def test_tensor_ordinary():
    print 'test_tensor_ordinary'
    i,j,k=Label('i',2),Label('j',2),Label('k',3)
    b=random([i,j,k])
    print "b: %s"%b
    print "b.transpose([2,1,0]): %s"%b.transpose([2,1,0])
    print "b.transpose([j,k,i]): %s"%b.transpose([j,k,i])
    print

def test_tensor_qngenerate():
    print 'test_tensor_qngenerate'
    N=5
    lqns,sqns=QuantumNumbers.kron([SPQNS(0.5)]*(N-1)).sorted(),SPQNS(0.5)
    rqns,permutation=QuantumNumbers.kron([lqns,sqns]).sorted(history=True)
    data=np.zeros((4**N,4**N))
    for slice in rqns.to_ordereddict().itervalues():
        data[slice,slice]=np.random.random((slice.stop-slice.start,slice.stop-slice.start))
    L,S,R=Label('i',len(lqns)),Label('j',len(sqns)),Label('k',len(rqns))
    m=DTensor(data[np.argsort(permutation),:].reshape((4**(N-1),4,4**N)),labels=[L,S,R])
    t1=time()
    m.qngenerate(-1,axes=[L,S],qnses=[lqns,sqns],flows=(1,1))
    t2=time()
    print '(L+S,R) qngenerate time: %ss.'%(t2-t1)
    t1=time()
    m.qngenerate(+1,axes=[S,R],qnses=[sqns,rqns],flows=(1,-1))
    t2=time()
    print '(L,-S+R) qngenerate time: %ss.'%(t2-t1)
    print

def test_tensor_svd():
    print 'test_tensor_svd'
    N=5
    L=Label('i',QuantumNumbers.kron([SPQNS(0.5)]*(N-1)),+1)
    S=Label('j',SPQNS(0.5),+1)
    R=Label('k',QuantumNumbers.kron([SPQNS(0.5)]*N),-1)
    m=random([L,S,R])
    print 'm.shape: %s'%(m.shape,)
    print 'm.labels: %s'%m.labels
    print

    print 'Using good quantum numbers'
    print 'L+S,R'
    t1=time()
    u,s,v=svd(m,row=m.labels[0:2],new=Label('new',None),col=m.labels[2:])
    t2=time()
    print 'svd time: %ss.'%(t2-t1)
    print 'svd diff: %s.'%(m-u*s*v).norm
    print

    print 'L,-S+R'
    t1=time()
    u,s,v=svd(m,row=m.labels[0:1],new=Label('new',None),col=m.labels[1:])
    t2=time()
    print 'svd time: %ss.'%(t2-t1)
    print 'svd diff: %s.'%(m-u*s*v).norm
    print

    print 'Not using good quantum numbers'
    qns,permutation=QuantumNumbers.kron([L.qns,S.qns]).sorted(history=True)
    data=m.merge(([L,S],Label('new',qns,1),permutation)).data.reshape((4**(N-1),4,4**N))
    L,S,R=Label('i',4**(N-1)),Label('j',4),Label('k',4**N)
    m=Tensor(data,labels=[L,S,R])

    print 'L+S,R'
    t1=time()
    u,s,v=svd(m,row=m.labels[0:2],new=Label('new',None),col=m.labels[2:])
    t2=time()
    print 'svd time: %ss.'%(t2-t1)
    print 'svd diff: %s.'%(m-u*s*v).norm
    print

    print 'L,-S+R'
    t1=time()
    u,s,v=svd(m,row=m.labels[0:1],new=Label('new',None),col=m.labels[1:])
    t2=time()
    print 'svd time: %ss.'%(t2-t1)
    print 'svd diff: %s.'%(m-u*s*v).norm
    print
    print

def test_tensor_directsum():
    print 'test_tensor_directsum'
    N=1
    L1=Label('l1',QuantumNumbers.kron([SQNS(0.5)]*N).sorted(),+1)
    S1=Label('s1',SQNS(0.5),+1)
    R1=Label('r1',QuantumNumbers.kron([SQNS(0.5)]*(N+1)).sorted(),-1)
    m1=random([L1,S1,R1])

    L2=Label('l2',QuantumNumbers.kron([SQNS(0.5)]*N).sorted(),+1)
    S2=Label('s2',SQNS(0.5),+1)
    R2=Label('r2',QuantumNumbers.kron([SQNS(0.5)]*(N+1)).sorted(),-1)
    m2=random([L2,S2,R2])

    L,S,R=Label('l',None),Label('s',None),Label('r',None)
    m=directsum([m1,m2],labels=[L,S,R],axes=[1])
    print 'm1.shape,m2.shape,m.shape: %s, %s, %s.'%(m1.shape,m2.shape,m.shape)

    lqns,lpermutation=m.labels[0].qns.sorted(history=True)
    rqns,rpermutation=m.labels[2].qns.sorted(history=True)
    m=m.reorder((0,lpermutation,lqns),(2,rpermutation,rqns))
    qns,permutation=QuantumNumbers.kron([lqns,S.qns],signs=(1,1)).sorted(history=True)
    print 'm.merge(([L,S],Label("LS",qns,1),permutation))>10**-10:\n%s'%(m.merge(([L,S],Label('LS',qns,1),permutation)).data>10**-10)
    print 'row.qns: %s'%qns
    print 'col.qns: %s'%rqns
    print

def test_tensor_expanded_svd():
    print 'test_tensor_expanded_svd'
    N=5

    print 'Using good quantum numbers'
    signs=[1 if i==0 else -1 for i in np.random.randint(2,size=N-1)]
    print 'signs: %s'%signs
    L=Label('L',SQNS(0.5),+1)
    S=Label('S',QuantumNumbers.kron([SQNS(0.5)]*(N-1),signs=signs),+1)
    R=Label('R',QuantumNumbers.kron([SQNS(0.5)]*N).sorted(),-1)
    E=[Label('S%i'%i,SQNS(0.5),flow=sign) for i,sign in enumerate(signs)]
    m=random([L,S,R])
    for n in xrange(N):
        I=[Label('B%i'%i,None) for i in xrange((N-1) if n in (0,N-1) else (N-2))]
        ms=expanded_svd(m,L=[L],S=S,R=[R],E=E,I=I,cut=n)
        ms=[ms[0],ms[1]]+ms[2] if n==0 else (ms[0]+[ms[1],ms[2]] if n==N-1 else ms[0][:n]+[ms[1]]+ms[0][n:])
        print 'cut(%s) diff: %s.'%(n,(m-np.product(ms).merge((E,S))).norm)

    print 'Not using good quantum numbers'
    L,S,R=Label('L',2),Label('S',2**(N-1)),Label('R',2**N)
    E=[Label('S%s'%i,2) for i in xrange(N-1)]
    m=random([L,S,R])
    for n in xrange(N):
        I=[Label('B%i'%i,None) for i in xrange((N-1) if n in (0,N-1) else (N-2))]
        ms=expanded_svd(m,L=[L],S=S,R=[R],E=E,I=I,cut=n)
        ms=[ms[0],ms[1]]+ms[2] if n==0 else (ms[0]+[ms[1],ms[2]] if n==N-1 else ms[0][:n]+[ms[1]]+ms[0][n:])
        print 'cut(%s) diff: %s.'%(n,(m-np.product(ms).merge((E,S))).norm)
    print

def test_tensor_deparallelization():
    print 'test_tensor_deparallelization'
    data=np.zeros((4,6))
    a1=np.random.random(6)
    a2=np.random.random(6)
    data[1,:]=a1
    data[2,:]=a2
    data[3,:]=a1
    m=DTensor(data,labels=[Label('i',4),Label('j',6)])
    T,M=deparallelization(m,row=[0],new=Label('new',None),col=[1],mode='R')
    print 'R deparallelization'
    print 'T.shape,M.shape: %s,%s.'%(T.shape,M.shape)
    print 'diff: %s.'%(m-T*M).norm
    m=DTensor(data.T,labels=[Label('i',6),Label('j',4)])
    M,T=deparallelization(m,row=[0],new=Label('new',None),col=[1],mode='C')
    print 'C deparallelization'
    print 'M.shape,T.shape: %s,%s.'%(M.shape,T.shape)
    print 'diff: %s.'%(m-M*T).norm
    print

def test_tensor_contract():
    print 'test_tensor_contract'
    N,nmax=50,400
    mps=MPS.random(sites=[SPQNS(0.5)]*N,bonds=[SPQN((0.0,0.0)),SPQN((N,0.0))],cut=N/2,nmax=nmax)
    print 'tensor shape: %s, %s, %s'%(mps[N/2-1].shape,mps[N/2].shape,mps[N/2+1].shape)
    t1=time()
    temp1=mps[N/2-1]*(mps[N/2],'einsum')*(mps[N/2+1],'einsum')
    print 'einsum time: %ss'%(time()-t1)
    t2=time()
    temp2=mps[N/2-1]*(mps[N/2],'tensordot')*(mps[N/2+1],'tensordot')
    print 'tensordot time: %ss.'%(time()-t2)
    t3=time()
    temp3=mps[N/2-1]*(mps[N/2],'block')*(mps[N/2+1],'block')
    print 'block time: %ss.'%(time()-t3)
    print 'diff: %s,%s,%s'%((temp1-temp2).norm,(temp2-temp3).norm,(temp3-temp1).norm)
    print
