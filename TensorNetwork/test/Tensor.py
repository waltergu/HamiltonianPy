'''
Tensor test.
'''

__all__=['test_tensor']

from numpy import *
from numpy.linalg import norm
from HamiltonianPy import *
from HamiltonianPy.TensorNetwork.Tensor import *
from time import time

def test_tensor():
    test_tensor_ordinary()
    test_tensor_qng()
    test_tensor_svd()
    test_tensor_directsum()
    test_tensor_expanded_svd()
    test_tensor_deparallelization()

def test_tensor_ordinary():
    print 'test_tensor_ordinary'
    b=Tensor(random.random((2,2,3)),labels=[Label('i'),Label('j'),Label('k')])
    print "b: %s"%b
    print "b.transpose([2,1,0]): %s"%b.transpose([2,1,0])
    print "b.transpose([Label('i'),Label('j'),Label('k')]): %s"%b.transpose([Label('i'),Label('j'),Label('k')])
    print "b.take(0,Label('i')): %s"%b.take(0,Label('i'))
    print "b.take([0,1],0): %s"%b.take([0,1],0)
    print "contract([b,b]): %s"%contract([b,b],engine='einsum')
    print

def test_tensor_qng():
    print 'test_tensor_qng'
    N=5
    lqns=QuantumNumbers.kron([SPQNS(0.5)]*(N-1)).sort()
    sqns=SPQNS(0.5)
    rqns,permutation=QuantumNumbers.kron([lqns,sqns]).sort(history=True)
    data=zeros((4**N,4**N))
    for qn,slice in rqns.to_ordereddict().iteritems():
        data[slice,slice]=random.random((slice.stop-slice.start,slice.stop-slice.start))
    L,S,R=Label('i'),Label('j'),Label('k')
    m=Tensor(data[argsort(permutation),:].reshape((4**(N-1),4,4**N)),labels=[L,S,R])
    t1=time()
    m.qng(axes=[L,S],qnses=[lqns,sqns],signs='++')
    t2=time()
    print '(L+S,R) qng time: %ss.'%(t2-t1)
    t1=time()
    m.qng(axes=[S,R],qnses=[sqns,rqns],signs='-+')
    t2=time()
    print '(L,-S+R) qng time: %ss.'%(t2-t1)
    print

def test_tensor_svd():
    print 'test_tensor_svd'
    N=5
    lqns=QuantumNumbers.kron([SPQNS(0.5)]*(N-1))
    sqns=SPQNS(0.5)
    rqns=QuantumNumbers.kron([SPQNS(0.5)]*N)

    print 'Using good quantum numbers'
    L,S,R=Label('i',qns=lqns),Label('j',qns=sqns),Label('k',qns=rqns)
    m=Tensor.random(shape=(4**(N-1),4,4**N),labels=[L,S,R],signs='++-')
    print 'm.shape: %s'%(m.shape,)
    print 'm.labels: %s'%m.labels

    print 'L+S,R'
    t1=time()
    u,s,v=m.svd(row=m.labels[0:2],new=Label('new'),col=m.labels[2:])
    t2=time()
    print 'svd time: %ss.'%(t2-t1)
    print 'svd diff: %s.'%(norm(m-contract([u,s,v],engine='einsum')))

    print 'L,-S+R'
    t1=time()
    u,s,v=m.svd(row=m.labels[0:1],new=Label('new'),col=m.labels[1:],row_signs='+',col_signs='-+')
    t2=time()
    print 'svd time: %ss.'%(t2-t1)
    print 'svd diff: %s.'%(norm(m-contract([u,s,v],engine='einsum')))


    print 'Not using good quantum numbers'
    qns,permutation=QuantumNumbers.kron([L.qns,S.qns]).sort(history=True)
    data=asarray(m.merge(([L,S],Label('new',qns=qns),permutation))).reshape((4**(N-1),4,4**N))
    L,S,R=Label('i'),Label('j'),Label('k')
    m=Tensor(data,labels=[L,S,R])

    print 'L+S,R'
    t1=time()
    u,s,v=m.svd(row=m.labels[0:2],new=Label('new'),col=m.labels[2:])
    t2=time()
    print 'svd time: %ss.'%(t2-t1)
    print 'svd diff: %s.'%(norm(m-contract([u,s,v],engine='einsum')))

    print 'L,-S+R'
    t1=time()
    u,s,v=m.svd(row=m.labels[0:1],new=Label('new'),col=m.labels[1:])
    t2=time()
    print 'svd time: %ss.'%(t2-t1)
    print 'svd diff: %s.'%(norm(m-contract([u,s,v],engine='einsum')))
    print

def test_tensor_directsum():
    print 'test_tensor_directsum'
    N=1
    L1=Label('l1',qns=QuantumNumbers.kron([SQNS(0.5)]*N).sort())
    S1=Label('s1',qns=SQNS(0.5))
    R1=Label('r1',qns=QuantumNumbers.kron([SQNS(0.5)]*(N+1)).sort())
    m1=Tensor.random((2**N,2,2**(N+1)),labels=[L1,S1,R1],signs='++-')

    L2=Label('l2',qns=QuantumNumbers.kron([SQNS(0.5)]*N).sort())
    S2=Label('s2',qns=SQNS(0.5))
    R2=Label('r2',qns=QuantumNumbers.kron([SQNS(0.5)]*(N+1)).sort())
    m2=Tensor.random((2**N,2,2**(N+1)),labels=[L2,S2,R2],signs='++-')

    L,S,R=Label('l',qns=L1.qns),Label('s',qns=SQNS(0.5)),Label('r',qns=R1.qns)
    m=Tensor.directsum([m1,m2],labels=[L,S,R],axes=[1])
    print 'm1.shape,m2.shape,m.shape: %s, %s, %s.'%(m1.shape,m2.shape,m.shape)

    lqns,lpermutation=m.labels[0].qns.sort(history=True)
    rqns,rpermutation=m.labels[2].qns.sort(history=True)
    m=m.reorder((0,lpermutation,lqns),(2,rpermutation,rqns))
    qns,permutation=QuantumNumbers.kron([lqns,S.qns],signs='++').sort(history=True)
    print 'm.merge(([L,S],Label("LS",qns=qns),permutation))>10**-6:\n%s'%(m.merge(([L,S],Label('LS',qns=qns),permutation))>10**-6)
    print 'qns: %s'%qns
    print

def test_tensor_expanded_svd():
    print 'test_tensor_expanded_svd'
    N=5

    print 'Using good quantum numers'
    random.seed()
    signs=''.join('+' if i==0 else '-' for i in random.randint(2,size=N-1))
    print 'signs: %s'%signs
    L=Label('L',qns=SQNS(0.5))
    S=Label('S',qns=QuantumNumbers.kron([SQNS(0.5)]*(N-1),signs=signs))
    R=Label('R',qns=QuantumNumbers.kron([SQNS(0.5)]*N).sort())
    E=[Label('S%i'%i,qns=SQNS(0.5)) for i in xrange(N-1)]
    m=Tensor.random((2,2**(N-1),2**N),labels=[L,S,R],signs='++-')
    for n in xrange(N):
        I=[Label('B%i'%i) for i in xrange((N-1) if n in (0,N-1) else (N-2))]
        ms=m.expanded_svd(L=[L],S=S,R=[R],E=E,I=I,ls='+',rs='+',es=signs,cut=n)
        ms=[ms[0],ms[1]]+ms[2] if n==0 else (ms[0]+[ms[1],ms[2]] if n==N-1 else ms[0][:n]+[ms[1]]+ms[0][n:])
        print 'cut(%s) diff: %s.'%(n,norm(m-contract(ms,engine='einsum').merge((E,S))))

    print 'Not using good quantum numbers'
    L,S,R=Label('L'),Label('S'),Label('R')
    E=[Label('S%s'%i,qns=2) for i in xrange(N-1)]
    m=Tensor.random((2,2**(N-1),2**N),labels=[L,S,R])
    for n in xrange(N):
        I=[Label('B%i'%i) for i in xrange((N-1) if n in (0,N-1) else (N-2))]
        ms=m.expanded_svd(L=[L],S=S,R=[R],E=E,I=I,cut=n)
        ms=[ms[0],ms[1]]+ms[2] if n==0 else (ms[0]+[ms[1],ms[2]] if n==N-1 else ms[0][:n]+[ms[1]]+ms[0][n:])
        print 'cut(%s) diff: %s.'%(n,norm(m-contract(ms,engine='einsum').merge((E,S))))
    print

def test_tensor_deparallelization():
    print 'test_tensor_deparallelization'
    m=zeros((4,6))
    a1=random.random(6)
    a2=random.random(6)
    m[1,:]=a1
    m[2,:]=a2
    m[3,:]=a1
    m=Tensor(m,labels=[Label('i'),Label('j')])
    T,M=m.deparallelization(row=[0],new=Label('new'),col=[1],mode='R')
    print 'R deparallelization'
    print 'T.shape,M.shape: %s,%s.'%(T.shape,M.shape)
    print 'diff: %s.'%norm(m-contract([T,M]))
    m=Tensor(m.T,labels=[Label('i'),Label('j')])
    M,T=m.deparallelization(row=[0],new=Label('new'),col=[1],mode='C')
    print 'C deparallelization'
    print 'M.shape,T.shape: %s,%s.'%(M.shape,T.shape)
    print 'diff: %s.'%norm(m-contract([M,T]))
    print
