'''
DMRG test.
'''

__all__=['test_dmrg']

import mkl
import numpy as np
from HamiltonianPy import *
from HamiltonianPy.TensorNetwork import *
from HamiltonianPy.DMRG import *

savedata=False

def test_dmrg():
    print 'test_dmrg'
    mkl.set_num_threads(1)
    Engine.DEBUG=True
    test_idmrg()
    test_fdmrg()

def test_idmrg():
    print 'test_idmrg'
    dmrg_spin('idmrg',spin=0.5,N=200,J=1.0,qnon=True,matvec='lo')
    dmrg_spinless_fermion('idmrg',N=200,t=-0.5,qnon=True,matvec='lo')
    dmrg_spinful_fermion('idmrg',N=200,t=-1.0,U=1.0,qnon=True,matvec='lo')
    dmrg_honeycomb_heisenberg('idmrg',N=20,J=1.0,qnon=True,matvec='lo')
    print

def test_fdmrg():
    print 'test_fdmrg'
    dmrg_spin('fdmrg',spin=0.5,N=20,J=1.0,qnon=True,matvec='lo')
    dmrg_spinless_fermion('fdmrg',N=20,t=-0.5,qnon=True,matvec='lo')
    dmrg_spinful_fermion('fdmrg',N=20,t=-1.0,U=1.0,qnon=True,matvec='lo')
    dmrg_honeycomb_heisenberg('fdmrg',N=10,J=1.0,qnon=True,matvec='lo')
    print

def dmrg_spin(mode,spin,N,J,qnon=True,matvec='lo'):
    print '%s_spin'%mode
    priority,layers=DEGFRE_SPIN_PRIORITY,DEGFRE_SPIN_LAYERS
    dmrg=DMRG(
        name=       '%s-spin-%s'%(mode,spin),
        mps=        MPS(mode='QN') if qnon else MPS(mode='NB'),
        lattice=    Cylinder(name='WG',block=[np.array([0.0,0.0])],translation=np.array([1.0,0.0]),nneighbour=1),
        terms=      [SpinTerm('J',J,neighbour=1,indexpacks=Heisenberg())],
        config=     IDFConfig(priority=priority,map=lambda pid: Spin(S=spin)),
        degfres=    DegFreTree(mode='QN',layers=layers,priority=priority,map=lambda index: SQNS(S=spin)) if qnon else
                    DegFreTree(mode='NB',layers=layers,priority=priority,map=lambda index: int(spin*2+1)),
        mask=       [],
        matvec=     matvec,
        dtype=      np.float64
        )
    targets=[SQN(0.0)]*(N/2) if qnon else [None]*(N/2)
    if mode=='idmrg':
        tsg=TSG(name='GROWTH',targets=targets,nmax=200,savedata=savedata,run=DMRGTSG)
        dmrg.register(tsg)
    else:
        tsg=TSG(name='GROWTH',targets=targets,nmax=200,savedata=savedata,plot=False,run=DMRGTSG)
        tss=TSS(name='SWEEP',target=targets[-1],nsite=N*dmrg.nspb,nmaxs=[200,200],dependences=[tsg],savedata=savedata,run=DMRGTSS)
        dmrg.register(tss)
    dmrg.summary()
    print

def dmrg_spinless_fermion(mode,N,t,qnon=True,matvec='lo'):
    print '%s_spinless_fermion'%mode
    priority,layers=DEGFRE_FERMIONIC_PRIORITY,DEGFRE_FERMIONIC_LAYERS
    dmrg=DMRG(
        name=       '%s-fermion-spin-0'%mode,
        mps=        MPS(mode='QN') if qnon else MPS(mode='NB'),
        lattice=    Cylinder(name='WG',block=[np.array([0.0,0.0])],translation=np.array([1.0,0.0])),
        terms=      [Hopping('t',t,neighbour=1)],
        config=     IDFConfig(priority=priority,map=lambda pid: Fermi(atom=0,norbital=1,nspin=1,nnambu=1)),
        degfres=    DegFreTree(mode='QN',layers=layers,priority=priority,map=lambda index: PQNS(1)) if qnon else
                    DegFreTree(mode='NB',layers=layers,priority=priority,map=lambda index: 2),
        mask=       ['nambu'],
        matvec=     matvec,
        dtype=      np.float64
        )
    targets=[PQN(num) for num in xrange(1,N/2+1)] if qnon else [None]*(N/2)
    if mode=='idmrg':
        tsg=TSG(name='GROWTH',targets=targets,nmax=200,savedata=savedata,run=DMRGTSG)
        dmrg.register(tsg)
    else:
        tsg=TSG(name='GROWTH',targets=targets,nmax=200,savedata=savedata,plot=False,run=DMRGTSG)
        tss=TSS(name='SWEEP',target=targets[-1],nsite=N*dmrg.nspb,nmaxs=[200,200],dependences=[tsg],savedata=savedata,run=DMRGTSS)
        dmrg.register(tss)
    dmrg.summary()
    print

def dmrg_spinful_fermion(mode,N,t,U,qnon=True,matvec='lo'):
    print '%s_spinful_fermion'%mode
    priority,layers=DEGFRE_FERMIONIC_PRIORITY,DEGFRE_FERMIONIC_LAYERS
    dmrg=DMRG(
        name=       '%s-fermion-spin-0.5'%mode,
        mps=        MPS(mode='QN') if qnon else MPS(mode='NB'),
        lattice=    Cylinder(name='WG',block=[np.array([0.0,0.0])],translation=np.array([1.0,0.0])),
        terms=      [Hopping('t',t,neighbour=1),Hubbard('U',U)],
        config=     IDFConfig(priority=priority,map=lambda pid: Fermi(atom=0,norbital=1,nspin=2,nnambu=1)),
        degfres=    DegFreTree(mode='QN',layers=layers,priority=priority,map=lambda index: SzPQNS(index.spin-0.5)) if qnon else
                    DegFreTree(mode='NB',layers=layers,priority=priority,map=lambda index: 2),
        mask=       ['nambu'],
        matvec=     matvec,
        dtype=      np.float64
        )
    targets=[SPQN((num*2,0.0)) for num in xrange(1,N/2+1)] if qnon else [None]*(N/2)
    if mode=='idmrg':
        tsg=TSG(name='GROWTH',targets=targets,nmax=200,savedata=savedata,run=DMRGTSG)
        dmrg.register(tsg)
    else:
        tsg=TSG(name='GROWTH',targets=targets,nmax=200,savedata=savedata,plot=False,run=DMRGTSG)
        tss=TSS(name='SWEEP',target=targets[-1],nsite=N*dmrg.nspb,nmaxs=[200,200],dependences=[tsg],savedata=savedata,run=DMRGTSS)
        dmrg.register(tss)
    dmrg.summary()
    print

def dmrg_honeycomb_heisenberg(mode,N,J,qnon=True,matvec='lo'):
    print '%s_honeycomb_heisenberg'%mode
    h4,priority,layers=Hexagon(name='H4'),DEGFRE_SPIN_PRIORITY,DEGFRE_SPIN_LAYERS
    dmrg=DMRG(
        name=       '%s-honeycomb-heisenberg'%mode,
        mps=        MPS(mode='QN') if qnon else MPS(mode='NB'),
        lattice=    Cylinder(name='WG',block=h4.rcoords,translation=h4.vectors[0],vectors=[h4.vectors[1]],nneighbour=1),
        terms=      [SpinTerm('J',J,neighbour=1,indexpacks=Heisenberg())],
        config=     IDFConfig(priority=priority,map=lambda pid: Spin(S=0.5)),
        degfres=    DegFreTree(mode='QN',layers=layers,priority=priority,map=lambda index: SQNS(S=0.5)) if qnon else
                    DegFreTree(mode='NB',layers=layers,priority=priority,map=lambda index: 2),
        mask=       [],
        matvec=     matvec,
        dtype=      np.float64
        )
    targets=[SQN(0.0)]*(N/2) if qnon else [None]*(N/2)
    if mode=='idmrg':
        tsg=TSG(name='GROWTH',targets=targets,nmax=100,savedata=savedata,run=DMRGTSG)
        dmrg.register(tsg)
    else:
        tsg=TSG(name='GROWTH',targets=targets,nmax=100,savedata=savedata,plot=False,run=DMRGTSG)
        tss=TSS(name='SWEEP',target=targets[-1],nsite=N*dmrg.nspb,nmaxs=[100,100],dependences=[tsg],savedata=savedata,run=DMRGTSS)
        dmrg.register(tss)
    dmrg.summary()
    print
