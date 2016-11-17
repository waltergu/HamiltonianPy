'''
Finite DMRG test.
'''

__all__=['test_fdmrg']

import numpy as np
import HamiltonianPy as HP
from HamiltonianPy.Basics import *
from HamiltonianPy.DMRG.Chain import*
from HamiltonianPy.DMRG.iDMRG import *
from HamiltonianPy.DMRG.fDMRG import *

def test_fdmrg():
    print 'test_fdmrg'
    test_fdmrg_spin()
    #test_fdmrg_spinless_fermion()
    #test_fdmrg_spinful_fermion()

def test_fdmrg_spin():
    print 'test_fdmrg_spin'
    # parameters
    N,J,spin,qn_on=20,1.0,1.0,True

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
        chain=EmptyChain(mode='QN')
        targets=[SpinQN(Sz=1.0)]*(N/2)
    else:
        degfres=DegFreTree(mode='NB',layers=layers,priority=priority,map=lambda index: int(spin*2+1))
        chain=EmptyChain(mode='NB')
        targets=[None]*(N/2)

    # dmrg
    idmrg=iDMRG(name='iDMRG',block=block,vector=vector,terms=terms,config=config,degfres=degfres,chain=chain,dtype=np.float64)
    idmrg.grow(scopes=range(N),targets=targets,nmax=20)
    fdmrg=fDMRG(name='fDMRG',lattice=idmrg.lattice,terms=idmrg.terms,config=idmrg.config,degfres=idmrg.degfres,chain=idmrg.chain)
    #fdmrg.chain.update(optstrs=fdmrg.chain.optstrs)
    fdmrg.sweep([20,30,60,100,200,200])
    print

def test_fdmrg_spinless_fermion():
    print 'test_fdmrg_spinless_fermion'
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
        chain=EmptyChain(mode='QN')
        targets=[QuantumNumber([('N',num,'U1')]) for num in xrange(1,N/2+1)]
    else:
        degfres=DegFreTree(mode='NB',layers=layers,priority=priority,map=lambda index: 2)
        chain=EmptyChain(mode='NB')
        targets=[None]*(N/2)

    # dmrg
    idmrg=iDMRG(name='iDMRG',block=block,vector=vector,terms=terms,config=config,degfres=degfres,mask=['nambu'],chain=chain,dtype=np.float64)
    idmrg.grow(scopes=range(N),targets=targets,nmax=20)
    fdmrg=fDMRG(name='fDMRG',lattice=idmrg.lattice,terms=idmrg.terms,config=idmrg.config,degfres=idmrg.degfres,chain=idmrg.chain)
    fdmrg.sweep([20,30,60,100,200,200])
    print

def test_fdmrg_spinful_fermion():
    print 'test_fdmrg_spinful_fermion'
    # parameters
    N,t,qn_on=20,-0.25,True

    # geometry
    block=Lattice(name='WG',points=[Point(PID(scope=0,site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0])],nneighbour=2)
    vector=np.array([1.0,0.0])

    # terms
    terms=[Hopping('t',t,neighbour=1)]

    # idfconfig
    priority,layers=('scope','site','orbital','spin','nambu'),[('scope','site','orbital'),('spin',)]
    config=IDFConfig(priority=priority,map=lambda pid: Fermi(atom=0,norbital=1,nspin=2,nnambu=1))
    if qn_on:
        degfres=DegFreTree(mode='QN',layers=layers,priority=priority,map=lambda index: QuantumNumberCollection([(FermiQN(N=1,Sz=0.5) if index.spin==1 else FermiQN(N=1,Sz=-0.5),1),(FermiQN(N=0,Sz=0.0),1)]))
        chain=EmptyChain(mode='QN')
        targets=[FermiQN(N=num*2,Sz=0.0) for num in xrange(1,N/2+1)]
    else:
        degfres=DegFreTree(mode='NB',layers=layers,priority=priority,map=lambda index: 2)
        chain=EmptyChain(mode='NB')
        targets=[None]*(N/2)

    # dmrg
    idmrg=iDMRG(name='iDMRG',block=block,vector=vector,terms=terms,config=config,degfres=degfres,mask=['nambu'],chain=chain,dtype=np.float64)
    idmrg.grow(scopes=range(N),targets=targets,nmax=20)
    fdmrg=fDMRG(name='fDMRG',lattice=idmrg.lattice,terms=idmrg.terms,config=idmrg.config,degfres=idmrg.degfres,chain=idmrg.chain)
    fdmrg.sweep([20,30,50,50,50])
    print
