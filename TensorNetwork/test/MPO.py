'''
MPO test.
'''

__all__=['test_mpo']

import numpy as np
import HamiltonianPy.Misc as hm
from numpy.linalg import norm
from HamiltonianPy import *
from HamiltonianPy.TensorNetwork import *

def test_mpo():
    print 'test_mpo'
    test_mpo_spin()

def test_mpo_spin():
    print 'test_mpo_spin'
    # set the lattice
    Nsite,Nscope=2,2
    a1,a2,points=np.array([1.0,0.0]),np.array([0.0,1.0]),[]
    for scope in xrange(Nscope):
        for site in xrange(Nsite):
            points.append(Point(PID(scope=scope,site=site),rcoord=a1*site+a2*scope,icoord=[0.0,0.0]))
    lattice=SuperLattice.compose(name='WG',points=points,nneighbour=1)
    #lattice.plot(pid_on=True)

    # set the degfres
    S,priority,layers=1.0,['scope','site','S'],[('scope',),('site','S')]
    config=IDFConfig(priority=priority)
    for pid in lattice:
        config[pid]=Spin(S=S)
    table=config.table(mask=[])
    degfres=DegFreTree(mode='QN',layers=layers,priority=priority,leaves=table.keys(),map=lambda index:SQNS(S))

    # set the states
    sites=degfres.labels(layer=layers[-1],mode='S')
    bonds=degfres.labels(layer=layers[-1],mode='B')
    bonds[+0]=bonds[+0].replace(qns=QuantumNumbers.mono(SQN(0.0)))
    bonds[-1]=bonds[-1].replace(qns=QuantumNumbers.mono(SQN(0.0)))
    cut=np.random.randint(0,Nsite*Nscope+1)
    mps1=MPS.random(sites,bonds,cut=cut)
    mps2=MPS.random(sites,bonds,cut=cut)
    stat1,stat2=mps1.state,mps2.state

    # set the terms
    terms=[SpinTerm('J',1.0,neighbour=1,indexpacks=Heisenberg())]
    #terms=[SpinTerm('J',1.0,neighbour=1,indexpacks=Ising('z'))]

    # set the operators
    opts=Generator(lattice.bonds,config,terms=terms,dtype=np.complex128).operators.values()
    mopts=[s_opt_rep(opt,table) for opt in opts]
    optstrs=[OptStr.from_operator(opt,degfres,layers[-1]) for opt in opts]
    mpos=[optstr.to_mpo(degfres) for optstr in optstrs]

    # calculate the overlap
    print 'overlap'
    overlaps=[]
    for i,(mopt,optstr,mpo) in enumerate(zip(mopts,optstrs,mpos)):
        overlap1=hm.overlap(stat1,mopt,stat2)
        overlap2=optstr.overlap(mps1,mps2)
        overlap3=mpo.overlap(mps1,mps2)
        overlaps.append(overlap1)
        print 'operator %s diff between 1,2: %s.'%(i,norm(overlap1-overlap2))
        print 'operator %s diff between 1,3: %s.'%(i,norm(overlap1-overlap3))
    print

    # test algebra of optstr
    print 'Algebra of OptStr'
    pos=np.random.randint(len(optstrs))
    optstr=optstrs[pos]
    print 'pos: %s'%pos
    print 'OptStr left multiplication diff: %s'%norm(2.0*hm.overlap(stat1,mopts[pos],stat2)-(optstr*2.0).overlap(mps1,mps2))
    print 'OptStr right multiplication diff: %s'%norm(2.0*hm.overlap(stat1,mopts[pos],stat2)-(2.0*optstr).overlap(mps1,mps2))
    print 'OptStr division diff: %s'%norm(hm.overlap(stat1,mopts[pos],stat2)/2.0-(optstr/2.0).overlap(mps1,mps2))
    print

    # test compression and algebra of mpo
    for i,(sm,m) in enumerate(zip(mopts,mpos)):
        if i==0:
            mopt=sm
            mpo=m
        else:
            mopt+=sm
            mpo+=m
    print 'Compression of MPO'
    print 'Before compression'
    print 'shapes: %s'%(','.join(str(m.shape) for m in mpo))
    print 'bonds.qnses: %s,%s'%(','.join(repr(m.labels[0].qns) for m in mpo),repr(mpo[-1].labels[-1].qns))
    mpo.compress(nsweep=4,tol=10**-6)
    print 'After compression'
    print 'shapes: %s'%(','.join(str(m.shape) for m in mpo))
    print 'bonds.qnses: %s,%s'%(','.join(repr(m.labels[0].qns) for m in mpo),repr(mpo[-1].labels[-1].qns))
    print
    print 'Algebra of MPO'
    pos=np.random.randint(len(mpos))
    print 'MPO addition diff: %s'%norm(hm.overlap(stat1,mopt,stat2)-mpo.overlap(mps1,mps2))
    print 'MPO subtraction diff: %s'%norm(hm.overlap(stat1,mopt,stat2)-overlaps[pos]-(mpo-mpos[pos]).overlap(mps1,mps2))
    print 'MPO left multiplication diff: %s'%norm(hm.overlap(stat1,mopt,stat2)*2.0-(mpo*2.0).overlap(mps1,mps2))
    print 'MPO right multiplication diff: %s'%norm(hm.overlap(stat1,mopt,stat2)*2.0-(2.0*mpo).overlap(mps1,mps2))
    print 'MPO division diff: %s'%norm(hm.overlap(stat1,mopt,stat2)/2.0-(mpo/2.0).overlap(mps1,mps2))
    print

    # test MPO*MPS, MPO*MPO
    print 'mps.cut: %s'%mps1.cut
    print 'MPO*MPS diff: %s'%norm(mopt.dot(stat1)-(mpo*mps1).state)
    print 'MPO*MPO diff: %s'%norm(hm.overlap(stat1,mopt.dot(mopt),stat2)-(mpo*mpo).overlap(mps1,mps2))
    print
