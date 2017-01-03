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
    Engine.DEBUG=True
    test_dmrg_spin()
    #test_dmrg_spinless_fermion()
    #test_dmrg_spinful_fermion()

def test_dmrg_spin():
    print 'test_dmrg_spin'
    # parameters
    qn_on=True
    N,J,spin=20,1.0,1.0

    # config related
    priority,layers=DEFAULT_SPIN_PRIORITY,DEFAULT_SPIN_LAYERS

    # dmrg
    dmrg=DMRG(
        log=        Log('/home/waltergu/Desktop/spin-%s.log'%(spin),mode='a+'),
        dout=       '/home/waltergu/Desktop',
        din=        '/home/waltergu/Desktop',
        name=       'spin-%s'%(spin),
        mps=        MPS(mode='QN') if qn_on else MPS(mode='NB'),
        lattice=    Lattice(),
        terms=      [SpinTerm('J',J,neighbour=1,indexpacks=Heisenberg())],
        #terms=     [SpinTerm('J',J,neighbour=1,indexpacks=IndexPackList(SpinPack(0.5,('+','-')),SpinPack(0.5,('-','+'))))]
        config=     IDFConfig(priority=priority,map=lambda pid: Spin(S=spin)),
        degfres=    DegFreTree(mode='QN',layers=layers,priority=priority,map=lambda index: SpinQNC(S=spin)) if qn_on else
                    DegFreTree(mode='NB',layers=layers,priority=priority,map=lambda index: int(spin*2+1)),
        mask=       [],
        dtype=      np.float64
        )
    # two site grow
    tsg=TSG(
            name=       'GTOWTH',
            block=      Lattice(name='WG',points=[Point(PID(scope=0,site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0])],nneighbour=2),
            vector=     np.array([1.0,0.0]),
            scopes=     range(N),
            targets=    [SpinQN(Sz=1.0)]*(N/2) if qn_on else [None]*(N/2),
            nmax=       20,
            run=DMRGTSG
            )
    # two site sweep
    dmrg.register(TSS(name='SWEEP',target=SpinQN(Sz=1.0),layer=0,nsite=N,nmaxs=[50,100,200,200],dependences=[tsg],run=DMRGTSS))
    dmrg.summary()
    print

def test_dmrg_spinless_fermion():
    print 'test_dmrg_spinless_fermion'
    # parameters
    qn_on=True
    N,t=20,-0.5

    # config related
    priority,layers=('scope','site','orbital','spin','nambu'),[('scope','site','orbital','spin')]
    degfres_map=lambda index: QuantumNumberCollection([(QuantumNumber([('N',1,'U1')]),1),(QuantumNumber([('N',0,'U1')]),1)])

    # dmrg
    dmrg=DMRG(
        log=        Log('/home/waltergu/Desktop/fermin-spin-o.log',mode='a+'),
        dout=       '/home/waltergu/Desktop',
        din=        '/home/waltergu/Desktop',
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
    # two site grow
    tsg=TSG(
            name=       'GTOWTH',
            block=      Lattice(name='WG',points=[Point(PID(scope=0,site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0])],nneighbour=2),
            vector=     np.array([1.0,0.0]),
            scopes=     range(N),
            targets=    [QuantumNumber([('N',num,'U1')]) for num in xrange(1,N/2+1)] if qn_on else [None]*(N/2),
            nmax=       20,
            run=        DMRGTSG
            )
    # two site sweep
    dmrg.register(TSS(name='SWEEP',target=QuantumNumber([('N',N/2,'U1')]),layer=0,nsite=N,nmaxs=[50,100,200,200],dependences=[tsg],run=DMRGTSS))
    dmrg.summary()
    print

def test_dmrg_spinful_fermion():
    print 'test_dmrg_spinful_fermion'
    # parameters
    qn_on=True
    N,t,U=20,-1.0,0.0

    # config related
    priority,layers=('scope','site','orbital','spin','nambu'),[('scope','site','orbital'),('spin',)]
    degfres_map=lambda index: QuantumNumberCollection([(FermiQN(N=1,Sz=0.5) if index.spin==1 else FermiQN(N=1,Sz=-0.5),1),(FermiQN(N=0,Sz=0.0),1)])

    # dmrg
    dmrg=DMRG(
        log=        Log('/home/waltergu/Desktop/fermin-spin-0.5.log',mode='a+'),
        dout=       '/home/waltergu/Desktop',
        din=        '/home/waltergu/Desktop',
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
    # two site grow
    tsg=TSG(
            name=       'GTOWTH',
            block=      Lattice(name='WG',points=[Point(PID(scope=0,site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0])],nneighbour=2),
            vector=     np.array([1.0,0.0]),
            scopes=     range(N),
            targets=    [FermiQN(N=num*2,Sz=0.0) for num in xrange(1,N/2+1)] if qn_on else [None]*(N/2),
            nmax=       30,
            run=        DMRGTSG
            )
    # two site sweep
    tss=TSS(name='PRESWEEP',target=FermiQN(N=N,Sz=0.0),layer=0,nsite=N,protocal=1,nmaxs=[30,30,30,30],run=DMRGTSS)
    dmrg.register(TSS(name='SWEEP',target=FermiQN(N=N,Sz=0.0),layer=1,nsite=2*N,nmaxs=[30,30,30,50],dependences=[tsg,tss],run=DMRGTSS))
    dmrg.summary()
    print
