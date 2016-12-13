'''
DMRG test.
'''

__all__=['test_dmrg']

import numpy as np
import HamiltonianPy as HP
from HamiltonianPy.Basics import *
from HamiltonianPy.DMRG.MPS import *
from HamiltonianPy.DMRG.DMRG import *

def test_dmrg():
    print 'test_dmrg'
    test_dmrg_spin()
    #test_dmrg_spinless_fermion()
    #test_dmrg_spinful_fermion()

def test_dmrg_spin():
    print 'test_dmrg_spin'
    # parameters
    N,J,spin,qn_on=20,1.0,0.5,True

    # geometry
    block=Lattice(name='WG',points=[Point(PID(scope=0,site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0])],nneighbour=2)
    vector=np.array([1.0,0.0])

    # terms
    terms=[SpinTerm('J',J,neighbour=1,indexpacks=Heisenberg())]
    #terms=[SpinTerm('J',J,neighbour=1,indexpacks=IndexPackList(SpinPack(0.5,('+','-')),SpinPack(0.5,('-','+'))))]

    # config & degfres & chain & target
    priority,layers=DEFAULT_SPIN_PRIORITY,DEFAULT_SPIN_LAYERS
    config=IDFConfig(priority=priority,map=lambda pid: Spin(S=spin))
    if qn_on:
        degfres=DegFreTree(mode='QN',layers=layers,priority=priority,map=lambda index: SpinQNC(S=spin))
        mps=MPS(mode='QN')
        targets=[SpinQN(Sz=0.0)]*(N/2)
    else:
        degfres=DegFreTree(mode='NB',layers=layers,priority=priority,map=lambda index: int(spin*2+1))
        mps=MPS(mode='NB')
        targets=[None]*(N/2)

    # dmrg
    dmrg=DMRG(name='spin-%s'%(spin),mps=mps,lattice=Lattice(),terms=terms,config=config,degfres=degfres,mask=[],dtype=np.float64)
    dmrg.register(TSG(name='GTOWTH',block=block,vector=vector,scopes=range(N),targets=targets,nmax=20,run=DMRGTSG))
    dmrg.register(TSS(name='SWEEP',BS=[{}]*4,nmaxs=[50,100,200,200],run=DMRGTSS))
    #check_block(dmrg.status,dmrg.mps)
    print

def test_dmrg_spinless_fermion():
    print 'test_dmrg_spinless_fermion'
    # parameters
    N,t,qn_on=20,-0.5,True

    # geometry
    block=Lattice(name='WG',points=[Point(PID(scope=0,site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0])],nneighbour=2)
    vector=np.array([1.0,0.0])

    # terms
    terms=[Hopping('t',t,neighbour=1)]

    # idfconfig
    priority,layers=('scope','site','orbital','spin','nambu'),[('scope','site','orbital','spin')]
    config=IDFConfig(priority=priority,map=lambda pid: Fermi(atom=0,norbital=1,nspin=1,nnambu=1))
    if qn_on:
        degfres=DegFreTree(mode='QN',layers=layers,priority=priority,map=lambda index: QuantumNumberCollection([(QuantumNumber([('N',1,'U1')]),1),(QuantumNumber([('N',0,'U1')]),1)]))
        mps=MPS(mode='QN')
        targets=[QuantumNumber([('N',num,'U1')]) for num in xrange(1,N/2+1)]
    else:
        degfres=DegFreTree(mode='NB',layers=layers,priority=priority,map=lambda index: 2)
        mps=MPS(mode='NB')
        targets=[None]*(N/2)

    # dmrg
    dmrg=DMRG(name='fermion-no-spin',mps=mps,lattice=Lattice(),terms=terms,config=config,degfres=degfres,mask=['nambu'],dtype=np.float64)
    dmrg.register(TSG(name='GTOWTH',block=block,vector=vector,scopes=range(N),targets=targets,nmax=20,run=DMRGTSG))
    dmrg.register(TSS(name='SWEEP',BS=[{}]*4,nmaxs=[50,100,200,200],run=DMRGTSS))
    #check_block(dmrg.status,dmrg.mps)
    print

def test_dmrg_spinful_fermion():
    print 'test_dmrg_spinful_fermion'
    # parameters
    N,t,U,qn_on=20,-1.0,1.0,True

    # geometry
    block=Lattice(name='WG',points=[Point(PID(scope=0,site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0])],nneighbour=2)
    vector=np.array([1.0,0.0])

    # terms
    terms=[Hopping('t',t,neighbour=1),Hubbard('U',U)]

    # idfconfig
    priority,layers=('scope','site','orbital','spin','nambu'),[('scope','site','orbital'),('spin',)]
    config=IDFConfig(priority=priority,map=lambda pid: Fermi(atom=0,norbital=1,nspin=2,nnambu=1))
    if qn_on:
        degfres=DegFreTree(mode='QN',layers=layers,priority=priority,map=lambda index: QuantumNumberCollection([(FermiQN(N=1,Sz=0.5) if index.spin==1 else FermiQN(N=1,Sz=-0.5),1),(FermiQN(N=0,Sz=0.0),1)]))
        mps=MPS(mode='QN')
        targets=[FermiQN(N=num*2,Sz=0.0) for num in xrange(1,N/2+1)]
    else:
        degfres=DegFreTree(mode='NB',layers=layers,priority=priority,map=lambda index: 2)
        mps=MPS(mode='NB')
        targets=[None]*(N/2)

    # dmrg
    dmrg=DMRG(name='fermion-spin-0.5',mps=mps,lattice=Lattice(),terms=terms,config=config,degfres=degfres,mask=['nambu'],dtype=np.float64)
    dmrg.register(TSG(name='GTOWTH',block=block,vector=vector,scopes=range(N),targets=targets,nmax=20,run=DMRGTSG))
    dmrg.register(TSS(name='SWEEP',BS=[{}]*2,nmaxs=[20,30],run=DMRGTSS))
    dmrg.level_up(n=1)
    dmrg.register(TSS(name='SWEEP',BS=[{}]*4,nmaxs=[50,100,200,200],run=DMRGTSS))
    #check_block(dmrg.status,dmrg.mps)
    print

def check_block(name,mps):
    print '%s check block'%(name)
    direction,count='L',0
    while count<2:
        if direction=='L':
            try:
                mps<<=1
            except:
                direction='R'
                count+=1
        else:
            try:
                mps>>=1
            except:
                direction='L'
        print 'cut:',mps.cut
        for m in mps:
            L,S,R=m.labels
            bond_qnc_generation(np.asarray(m),bond=L.qnc,site=S.qnc,mode='R')
            bond_qnc_generation(np.asarray(m),site=S.qnc,bond=R.qnc,mode='L')
    print
