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
    test_mpo_fermi()

def test_mpo_spin():
    print 'test_mpo_spin'

    # set the lattice
    Nscope,Nsite=2,2
    a1,a2,points=np.array([1.0,0.0]),np.array([0.0,1.0]),[]
    for scope in xrange(Nscope):
        for site in xrange(Nsite):
            points.append(Point(PID(scope=scope,site=site),rcoord=a1*site+a2*scope,icoord=[0.0,0.0]))
    lattice=Lattice.compose(name='WG',points=points,nneighbour=1)
    lattice.plot(pid_on=True)

    # set the terms
    terms=[SpinTerm('J',1.0,neighbour=1,indexpacks=Heisenberg())]

    # set the degfres
    S,priority,layers=1.0,['scope','site','orbital','S'],[('scope',),('site','orbital','S')]
    config=IDFConfig(priority=priority)
    for pid in lattice.pids:
        config[pid]=Spin(S=S)
    table=config.table(mask=[])
    degfres=DegFreTree(mode='QN',layers=layers,priority=priority,leaves=table.keys(),map=lambda index:SQNS(S))

    # set the operators
    opts=Generator(lattice.bonds,config,terms=terms,dtype=np.complex128).operators.values()
    optstrs=[OptStr.from_operator(opt,degfres,layers[-1]) for opt in opts]
    mpos=[optstr.to_mpo(degfres) for optstr in optstrs]

    # set the states
    sites,bonds=degfres.labels(mode='S',layer=layers[-1]),degfres.labels(mode='B',layer=layers[-1])
    bonds[+0]=bonds[+0].replace(qns=QuantumNumbers.mono(SQN(0.0)))
    bonds[-1]=bonds[-1].replace(qns=QuantumNumbers.mono(SQN(0.0)))
    cut=np.random.randint(0,Nsite*Nscope+1)
    mps1,mps2=MPS.random(sites,bonds,cut=cut,nmax=20),MPS.random(sites,bonds,cut=cut,nmax=20)

    # set the reference of test
    mopts=[soptrep(opt,table) for opt in opts]
    overlaps=[hm.overlap(mps1.state,mopt,mps2.state) for mopt in mopts]

    # test optstr
    test_optstr_overlap(optstrs,mps1,mps2,overlaps)
    test_optstr_algebra(optstrs,mps1,mps2,overlaps)
    test_optstr_relayer(optstrs,degfres,mps1,mps2,overlaps)

    # test mpo
    test_mpo_overlap(mpos,mps1,mps2,overlaps)
    test_mpo_algebra(mpos,mps1,mps2,mopts,overlaps)
    test_mpo_relayer(mpos,degfres,mps1,mps2,overlaps)
    print

def test_mpo_fermi():
    print 'test_mpo_fermi'

    # set the lattice
    Nscope,Nsite=2,2
    a1,a2,points=np.array([1.0,0.0]),np.array([0.0,1.0]),[]
    for scope in xrange(Nscope):
        for site in xrange(Nsite):
            points.append(Point(PID(scope=scope,site=site),rcoord=a1*site+a2*scope,icoord=[0.0,0.0]))
    lattice=Lattice.compose(name='WG',points=points,nneighbour=1)
    lattice.plot(pid_on=True)

    # set the terms
    terms=[Hopping('t',1.0,neighbour=1)]

    # set the degfres
    priority,layers=['scope','site','orbital','spin','nambu'],[('scope',),('site','orbital','spin')]
    config=IDFConfig(priority=priority)
    for pid in lattice.pids:
        config[pid]=Fermi(nspin=1)
    table=config.table(mask=['nambu'])
    degfres=DegFreTree(mode='QN',layers=layers,priority=priority,leaves=table.keys(),map=lambda index:PQNS(1))

    # set the operators
    opts=Generator(lattice.bonds,config,table=table,terms=terms,dtype=np.complex128).operators.values()
    optstrs=[OptStr.from_operator(opt,degfres,layers[-1]) for opt in opts]
    for i,(opt,optstr) in enumerate(zip(opts,optstrs)):
        print 'operator: %s'%i
        print opt
        print optstr
        print
    mpos=[optstr.to_mpo(degfres) for optstr in optstrs]

    # set the states
    sites,bonds=degfres.labels(mode='S',layer=layers[-1]),degfres.labels(mode='B',layer=layers[-1])
    bonds[+0]=bonds[+0].replace(qns=QuantumNumbers.mono(PQN(0)))
    bonds[-1]=bonds[-1].replace(qns=QuantumNumbers.mono(PQN(Nscope*Nsite/2)))
    cut=np.random.randint(0,Nsite*Nscope+1)
    mps1,mps2=MPS.random(sites,bonds,cut=cut,nmax=20),MPS.random(sites,bonds,cut=cut,nmax=20)

    # set the reference of test
    mopts=[foptrep(opt,basis=FBasis(nstate=Nscope*Nsite),transpose=True,dtype=np.complex128) for opt in opts]
    overlaps=[hm.overlap(mps1.state,mopt,mps2.state) for mopt in mopts]

    # test optstr
    test_optstr_overlap(optstrs,mps1,mps2,overlaps)
    test_optstr_algebra(optstrs,mps1,mps2,overlaps)
    test_optstr_relayer(optstrs,degfres,mps1,mps2,overlaps)

    # test mpo
    test_mpo_overlap(mpos,mps1,mps2,overlaps)
    test_mpo_algebra(mpos,mps1,mps2,mopts,overlaps)
    test_mpo_relayer(mpos,degfres,mps1,mps2,overlaps)
    print

def test_optstr_overlap(optstrs,mps1,mps2,overlaps):
    print 'test_optstr_overlap'
    for i,(optstr,overlap) in enumerate(zip(optstrs,overlaps)):
        print 'optstr %s diff: %s.'%(i,norm(overlap-optstr.overlap(mps1,mps2)))
    print

def test_optstr_algebra(optstrs,mps1,mps2,overlaps):
    print 'test_optstr_algebra'
    for i,(optstr,overlap) in enumerate(zip(optstrs,overlaps)):
        print 'Position: %s'%i
        print 'Left multiplication diff: %s'%norm(2.0*overlap-(optstr*2.0).overlap(mps1,mps2))
        print 'Right multiplication diff: %s'%norm(2.0*overlap-(2.0*optstr).overlap(mps1,mps2))
        print 'Division diff: %s'%norm(overlap/2.0-(optstr/2.0).overlap(mps1,mps2))
        print

def test_optstr_relayer(optstrs,degfres,mps1,mps2,overlaps):
    print 'test_optstr_relayer'
    nmps1=mps1.relayer(degfres,degfres.layers[0])
    nmps2=mps2.relayer(degfres,degfres.layers[0])
    for i,(optstr,overlap) in enumerate(zip(optstrs,overlaps)):
        print 'optstr %s diff between layer 0 and layer 1: %s.'%(i,norm(overlap-optstr.relayer(degfres,degfres.layers[0]).overlap(nmps1,nmps2)))
    print

def test_mpo_overlap(mpos,mps1,mps2,overlaps):
    print 'test_mpo_overlap'
    for i,(mpo,overlap) in enumerate(zip(mpos,overlaps)):
        print 'mpo %s diff: %s.'%(i,norm(overlap-mpo.overlap(mps1,mps2)))
    print

def test_mpo_algebra(mpos,mps1,mps2,mopts,overlaps):
    print 'test_mpo_algebra'
    mopt,summation=sum(mopts),sum(overlaps)
    for i,mpo in enumerate(mpos):
        M=mpo if i==0 else M+mpo

    print 'Compression of MPO'
    print 'Before compression'
    print 'shapes: %s'%(','.join(str(m.shape) for m in M))
    print 'bonds.qnses: %s,%s'%(','.join(repr(m.labels[0].qns) for m in M),repr(M[-1].labels[-1].qns))
    M.compress(nsweep=4,method='dpl')
    print 'After compression'
    print 'shapes: %s'%(','.join(str(m.shape) for m in M))
    print 'bonds.qnses: %s,%s'%(','.join(repr(m.labels[0].qns) for m in M),repr(M[-1].labels[-1].qns))
    print

    print '+,-*,/ of MPO'
    pos=np.random.randint(len(mpos))
    print 'Addition diff: %s'%norm(summation-M.overlap(mps1,mps2))
    print 'Subtraction diff: %s'%norm(summation-overlaps[pos]-(M-mpos[pos]).overlap(mps1,mps2))
    print 'Left multiplication diff: %s'%norm(summation*2.0-(M*2.0).overlap(mps1,mps2))
    print 'Right multiplication diff: %s'%norm(summation*2.0-(2.0*M).overlap(mps1,mps2))
    print 'Division diff: %s'%norm(summation/2.0-(M/2.0).overlap(mps1,mps2))
    print

    print 'MPO*MPS, MPO*MPO'
    print 'mps.cut: %s'%mps2.cut
    print 'MPO*MPS diff: %s'%norm(summation-hm.overlap(mps1.state,(M*mps2).state))
    print 'MPO*MPO diff: %s'%norm(hm.overlap(mps1.state,mopt.dot(mopt),mps2.state)-(M*M).overlap(mps1,mps2))
    print

def test_mpo_relayer(mpos,degfres,mps1,mps2,overlaps):
    print 'test_mpo_relayer'
    nmps1=mps1.relayer(degfres,degfres.layers[0])
    nmps2=mps2.relayer(degfres,degfres.layers[0])
    nmpos0=[mpo.relayer(degfres,degfres.layers[0]) for mpo in mpos]
    nmpos1=[mpo.relayer(degfres,degfres.layers[1]) for mpo in nmpos0]
    for i,(nmpo0,nmpo1,overlap) in enumerate(zip(nmpos0,nmpos1,overlaps)):
        print 'mpo %s diff between layer 0 and layer 1: %s.'%(i,norm(overlap-nmpo0.overlap(nmps1,nmps2)))
        print 'mpo %s diff between layer 1 and layer 0: %s.'%(i,norm(overlap-nmpo1.overlap(mps1,mps2)))
    print
