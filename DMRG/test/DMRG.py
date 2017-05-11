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
    test_idmrg()
    test_fdmrg()

def test_idmrg():
    print 'test_idmrg'
    dmrg_spin('idmrg',spin=1.0,N=200,J=1.0,qnon=True)
    dmrg_spinless_fermion('idmrg',N=200,t=-0.5,qnon=True)
    dmrg_spinful_fermion('idmrg',N=200,t=-1.0,U=1.0,qnon=True)
    print

def test_fdmrg():
    print 'test_fdmrg'
    dmrg_spin('fdmrg',spin=1.0,N=20,J=1.0,qnon=True)
    dmrg_spinless_fermion('fdmrg',N=20,t=-0.5,qnon=True)
    dmrg_spinful_fermion('fdmrg',N=20,t=-1.0,U=1.0,qnon=True)
    print

def dmrg_spin(mode,spin,N,J,qnon=True):
    print '%s_spin'%mode
    priority,layers=DEGFRE_SPIN_PRIORITY,DEGFRE_SPIN_LAYERS
    dmrg=DMRG(
        log=        Log('spin-%s.log'%(spin),mode='a+'),
        name=       'spin-%s'%(spin),
        mps=        MPS(mode='QN') if qnon else MPS(mode='NB'),
        lattice=    Cylinder(name='WG',block=[np.array([0.0,0.0])],translation=np.array([1.0,0.0]),nneighbour=1),
        terms=      [SpinTerm('J',J,neighbour=1,indexpacks=Heisenberg())],
        config=     IDFConfig(priority=priority,map=lambda pid: Spin(S=spin)),
        degfres=    DegFreTree(mode='QN',layers=layers,priority=priority,map=lambda index: SQNS(S=spin)) if qnon else
                    DegFreTree(mode='NB',layers=layers,priority=priority,map=lambda index: int(spin*2+1)),
        mask=       [],
        dtype=      np.float64
        )
    targets=[SQN(0.0)]*(N/2) if qnon else [None]*(N/2)
    if mode=='idmrg':
        tsg=TSG(name='GTOWTH',scopes=range(N),targets=targets,nmax=200,terminate=True,save_data=False,plot=True,run=DMRGTSG)
        dmrg.register(tsg)
    else:
        target=SQN(0.0)
        tsg=TSG(name='GTOWTH',scopes=range(N),targets=targets,nmax=20,terminate=False,save_data=False,plot=True,run=DMRGTSG)
        tss=TSS(name='SWEEP',target=target,layer=0,nsite=N,nmaxs=[50,100,200,200],dependences=[tsg],save_data=False,plot=True,save_fig=True,run=DMRGTSS)
        dmrg.register(tss)
    dmrg.summary()
    print

def dmrg_spinless_fermion(mode,N,t,qnon=True):
    print '%s_spinless_fermion'%mode
    priority,layers=('scope','site','orbital','spin','nambu'),[('scope','site','orbital','spin')]
    dmrg=DMRG(
        log=        Log('fermin-spin-o.log',mode='a+'),
        name=       'fermion-spin-o',
        mps=        MPS(mode='QN') if qnon else MPS(mode='NB'),
        lattice=    Cylinder(name='WG',block=[np.array([0.0,0.0])],translation=np.array([1.0,0.0])),
        terms=      [Hopping('t',t,neighbour=1)],
        config=     IDFConfig(priority=priority,map=lambda pid: Fermi(atom=0,norbital=1,nspin=1,nnambu=1)),
        degfres=    DegFreTree(mode='QN',layers=layers,priority=priority,map=lambda index: PQNS(1)) if qnon else
                    DegFreTree(mode='NB',layers=layers,priority=priority,map=lambda index: 2),
        mask=       ['nambu'],
        dtype=      np.float64
        )
    targets=[PQN(num) for num in xrange(1,N/2+1)] if qnon else [None]*(N/2)
    if mode=='idmrg':
        tsg=TSG(name='GTOWTH',scopes=range(N),targets=targets,nmax=200,terminate=True,save_data=False,plot=True,run=DMRGTSG)
        dmrg.register(tsg)
    else:
        target=PQN(N/2)
        tsg=TSG(name='GTOWTH',scopes=range(N),targets=targets,nmax=20,terminate=False,save_data=False,plot=True,run=DMRGTSG)
        tss=TSS(name='SWEEP',target=target,layer=0,nsite=N,nmaxs=[50,100,200,200],dependences=[tsg],save_data=False,plot=True,save_fig=True,run=DMRGTSS)
        dmrg.register(tss)
    dmrg.summary()
    print

def dmrg_spinful_fermion(mode,N,t,U,qnon=True):
    print '%s_spinful_fermion'%mode
    priority,layers=DEGFRE_FERMIONIC_PRIORITY,DEGFRE_FERMIONIC_LAYERS
    dmrg=DMRG(
        log=        Log('fermin-spin-0.5.log',mode='a+'),
        name=       'fermion-spin-0.5',
        mps=        MPS(mode='QN') if qnon else MPS(mode='NB'),
        lattice=    Cylinder(name='WG',block=[np.array([0.0,0.0])],translation=np.array([1.0,0.0])),
        terms=      [Hopping('t',t,neighbour=1),Hubbard('U',U)],
        config=     IDFConfig(priority=priority,map=lambda pid: Fermi(atom=0,norbital=1,nspin=2,nnambu=1)),
        degfres=    DegFreTree(mode='QN',layers=layers,priority=priority,map=lambda index: SzPQNS(index.spin-0.5)) if qnon else
                    DegFreTree(mode='NB',layers=layers,priority=priority,map=lambda index: 2),
        layer=      0,
        mask=       ['nambu'],
        dtype=      np.float64
        )
    targets=[SPQN((num*2,0.0)) for num in xrange(1,N/2+1)] if qnon else [None]*(N/2)
    if mode=='idmrg':
        tsg=TSG(name='GTOWTH',scopes=range(N),targets=targets,nmax=200,terminate=True,save_data=False,plot=True,run=DMRGTSG)
        dmrg.register(tsg)
    else:
        target=SPQN((N,0.0))
        tsg=TSG(name='GTOWTH',scopes=range(N),targets=targets,nmax=50,terminate=False,save_data=False,plot=True,run=DMRGTSG)
        tss=TSS(name='SWEEP',target=target,layer=0,nsite=N,nmaxs=[100,200,200],dependences=[tsg],save_data=False,plot=True,save_fig=True,run=DMRGTSS)
        dmrg.register(tss)
    dmrg.summary()
    print
