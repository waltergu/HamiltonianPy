'''
DMRG test.
'''

__all__=['test_dmrg']

import mkl
import numpy as np
import HamiltonianPy as HP
from HamiltonianPy import *
from HamiltonianPy.TensorNetwork import *
from HamiltonianPy.DMRG import *

def test_dmrg():
    print 'test_dmrg'
    mkl.set_num_threads(1)
    Engine.DEBUG=True
    #test_dmrg_spin()
    #test_dmrg_spinless_fermion()
    test_dmrg_spinful_fermion()

def test_dmrg_spin():
    print 'test_dmrg_spin'
    N,J,spin,qn_on=20,1.0,1.0,True
    priority,layers=DEFAULT_SPIN_PRIORITY,DEFAULT_SPIN_LAYERS
    dmrg=DMRG(
        log=        Log('spin-%s.log'%(spin),mode='a+'),
        name=       'spin-%s'%(spin),
        mps=        MPS(mode='QN') if qn_on else MPS(mode='NB'),
        lattice=    Lattice(),
        terms=      [SpinTerm('J',J,neighbour=1,indexpacks=Heisenberg())],
        #terms=     [SpinTerm('J',J,neighbour=1,indexpacks=IndexPackList(SpinPack(0.5,('+','-')),SpinPack(0.5,('-','+'))))]
        config=     IDFConfig(priority=priority,map=lambda pid: Spin(S=spin)),
        degfres=    DegFreTree(mode='QN',layers=layers,priority=priority,map=lambda index: SQNS(S=spin)) if qn_on else
                    DegFreTree(mode='NB',layers=layers,priority=priority,map=lambda index: int(spin*2+1)),
        mask=       [],
        dtype=      np.float64
        )
    tsg=TSG(
            name=       'GTOWTH',
            block=      Lattice.compose(name='WG',points=[Point(PID(scope=0,site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0])],nneighbour=2),
            vector=     np.array([1.0,0.0]),
            scopes=     range(N),
            targets=    [SQN(1.0)]*(N/2) if qn_on else [None]*(N/2),
            nmax=       20,
            save_data=  False,
            plot=       True,
            run=DMRGTSG
            )
    dmrg.register(TSS(name='SWEEP',target=SQN(1.0),layer=0,nsite=N,nmaxs=[50,100,200,200],dependences=[tsg],save_data=False,plot=True,save_fig=True,run=DMRGTSS))
    dmrg.summary()
    print

def test_dmrg_spinless_fermion():
    print 'test_dmrg_spinless_fermion'
    N,t,qn_on=20,-0.5,True
    priority,layers=('scope','site','orbital','spin','nambu'),[('scope','site','orbital','spin')]
    degfres_map=lambda index: QuantumNumbers('C',([PQN(0.0),PQN(1.0)],[1,1]),protocal=QuantumNumbers.COUNTS)
    dmrg=DMRG(
        log=        Log('fermin-spin-o.log',mode='a+'),
        name=       'fermion-spin-o',
        mps=        MPS(mode='QN') if qn_on else MPS(mode='NB'),
        lattice=    Lattice(),
        terms=      [Hopping('t',t,neighbour=1)],
        config=     IDFConfig(priority=priority,map=lambda pid: Fermi(atom=0,norbital=1,nspin=1,nnambu=1)),
        degfres=    DegFreTree(mode='QN',layers=layers,priority=priority,map=degfres_map) if qn_on else
                    DegFreTree(mode='NB',layers=layers,priority=priority,map=lambda index: 2),
        mask=       ['nambu'],
        dtype=      np.float64
        )
    tsg=TSG(
            name=       'GTOWTH',
            block=      Lattice.compose(name='WG',points=[Point(PID(scope=0,site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0])],nneighbour=2),
            vector=     np.array([1.0,0.0]),
            scopes=     range(N),
            targets=    [PQN(num) for num in xrange(1,N/2+1)] if qn_on else [None]*(N/2),
            nmax=       20,
            save_data=  False,
            plot=       True,
            run=        DMRGTSG
            )
    dmrg.register(TSS(name='SWEEP',target=PQN(N/2),layer=0,nsite=N,nmaxs=[50,100,200,200],dependences=[tsg],save_data=False,plot=True,run=DMRGTSS))
    dmrg.summary()
    print

def test_dmrg_spinful_fermion():
    print 'test_dmrg_spinful_fermion'
    N,t,U,qn_on=20,-1.0,1.0,True
    priority,layers=('scope','site','orbital','spin','nambu'),[('scope','site','orbital'),('spin',)]
    degfres_map=lambda index: QuantumNumbers('C',([SPQN((0.0,0.0)),SPQN((1.0,0.5 if index.spin==1 else -0.5))],[1,1]),protocal=QuantumNumbers.COUNTS)
    dmrg=DMRG(
        log=        Log('fermin-spin-0.5.log',mode='a+'),
        name=       'fermion-spin-0.5',
        mps=        MPS(mode='QN') if qn_on else MPS(mode='NB'),
        lattice=    Lattice(),
        terms=      [Hopping('t',t,neighbour=1),Hubbard('U',U)],
        config=     IDFConfig(priority=priority,map=lambda pid: Fermi(atom=0,norbital=1,nspin=2,nnambu=1)),
        degfres=    DegFreTree(mode='QN',layers=layers,priority=priority,map=degfres_map) if qn_on else
                    DegFreTree(mode='NB',layers=layers,priority=priority,map=lambda index: 2),
        layer=      0,
        mask=       ['nambu'],
        dtype=      np.float64
        )
    tsg=TSG(
            name=       'GTOWTH',
            block=      Lattice.compose(name='WG',points=[Point(PID(scope=0,site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0])],nneighbour=2),
            vector=     np.array([1.0,0.0]),
            scopes=     range(N),
            targets=    [SPQN((num*2,0.0)) for num in xrange(1,N/2+1)] if qn_on else [None]*(N/2),
            nmax=       30,
            save_data=  False,
            plot=       True,
            run=        DMRGTSG
            )
    #tss=TSS(name='PRESWEEP',target=SPQN((N,0.0)),layer=0,nsite=N,protocal=1,nmaxs=[30,50,100],save_data=False,plot=True,run=DMRGTSS)
    #dmrg.register(TSS(name='SWEEP',target=SPQN((N,0.0)),layer=1,nsite=2*N,nmaxs=[100,150,200,200],dependences=[tsg,tss],save_data=False,plot=True,save_fig=True,run=DMRGTSS))
    dmrg.register(TSS(name='SWEEP',target=SPQN((N,0.0)),layer=0,nsite=N,nmaxs=[30,50,100,100,200,200],dependences=[tsg],save_data=False,plot=True,save_fig=True,run=DMRGTSS))
    dmrg.summary()
    print
