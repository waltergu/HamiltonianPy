'''
MPO test.
'''

__all__=['test_mpo']

from numpy import *
from numpy.linalg import norm
from HamiltonianPy import *
from HamiltonianPy.Math.Tensor import *
from HamiltonianPy.DMRG.MPO import *
from HamiltonianPy.DMRG.MPS import *

def test_mpo():
    print 'test_mpo'
    N=8
    z=0j
    m=random.random((2,2))+random.random((2,2))*z
    m=(m+m.T.conjugate())/2
    l=Lattice(name='WG',points=tiling(cluster=[Point(pid=PID(scope='WG',site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0])],vectors=[array([1.0,0.0])],indices=xrange(N)),nneighbour=1)
    config=IDFConfig(priority=DEFAULT_SPIN_PRIORITY)
    for pid in l:
        config[pid]=Spin(S=0.5)
    layer=('scope','site','S')
    degfres=DegFreTree(config,layers=(layer,))
    table=degfres.table(layer=layer)
    labels=degfres.labels(layer=layer).values()
    terms=[SpinTerm('J',3.0,neighbour=1,indexpacks=IndexPackList(SpinPack(1.0,pack=(('WG',m),('WG',m)))))]
    opts=Generator(l.bonds,config,terms=terms,dtype=complex128).operators.values()
    for opt in opts:
        stat1,stat2=random.random(2**N)+random.random(2**N)*z,random.random(2**N)+random.random(2**N)*z
        stat1,stat2=stat1/norm(stat1),stat2/norm(stat2)
        dense=asarray(s_opt_rep(opt,table).todense())
        overlap1_12=vdot(stat1,dense.dot(stat2))
        overlap1_11=vdot(stat1,dense.dot(stat1))
        overlap1_22=vdot(stat2,dense.dot(stat2))
        cut1,cut2=random.randint(0,N,size=2)
        print 'cut1,cut2:',cut1,cut2
        mps1,mps2=MPS.from_state(stat1,shapes=[2]*N,labels=labels,cut=cut1),MPS.from_state(stat2,shapes=[2]*N,labels=labels,cut=cut2)
        optstr=OptStr.from_operator(opt,degfres,layer)
        overlap2_12=optstr.overlap(mps1,mps2)
        overlap2_11=optstr.overlap(mps1,mps1)
        overlap2_22=optstr.overlap(mps2,mps2)
        print 'overlap1: %s,%s,%s'%(overlap1_12,overlap1_11,overlap1_22)
        print 'overlap2: %s,%s,%s'%(overlap2_12,overlap2_11,overlap2_22)
        print 'difference: %s,%s,%s'%(overlap1_12-overlap2_12,overlap1_11-overlap2_11,overlap1_22-overlap2_22)
        print '-'*80
    print
