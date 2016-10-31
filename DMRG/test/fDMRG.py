'''
Finite DMRG test.
'''

__all__=['test_fdmrg']

import numpy as np
from HamiltonianPy.Basics import *
from HamiltonianPy.DMRG.Chain import*
from HamiltonianPy.DMRG.iDMRG import *
from HamiltonianPy.DMRG.fDMRG import *
import time

def test_fdmrg():
    print 'test_fdmrg'
    # parameters
    N,J1,J2,h=20,1.0,0.0,0.0
    spin=1.0

    # geometry
    p1=Point(PID(scope=0,site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0])
    points,a1=[p1],np.array([0.0,1.0])
    #p2=Point(PID(scope=0,site=1),rcoord=[0.0,1.0],icoord=[0.0,0.0])
    #points,a1=[p1,p2],np.array([0.0,2.0])
    block=Lattice(name='WG',points=points,nneighbour=2)
    #block=Lattice(name='WG',points=points,nneighbour=2,vectors=[a1])
    vector=np.array([1.0,0.0])

    # idfconfig
    idfconfig=IDFConfig(priority=DEFAULT_SPIN_PRIORITY)
    for i in xrange(N):
        idfconfig[p1.pid._replace(scope=i)]=Spin(S=spin)
        #idfconfig[p2.pid._replace(scope=i)]=Spin(S=spin)

    # qncconfig
    qncconfig=QNCConfig(priority=DEFAULT_SPIN_PRIORITY)
    for index in idfconfig.table():
        qncconfig[index]=SpinQNC(S=spin)

    # degfres
    t1=time.time()
    #degfres=DegFreTree(idfconfig,layers=DEFAULT_SPIN_LAYERS,priority=DEFAULT_SPIN_PRIORITY)
    degfres=DegFreTree(qncconfig,layers=DEFAULT_SPIN_LAYERS,priority=DEFAULT_SPIN_PRIORITY)
    t2=time.time()
    print 'degfres construction: %ss.'%(t2-t1)

    # terms
    terms=[
            SpinTerm('h',h,neighbour=0,indexpacks=S('z')),
            SpinTerm('J1',J1,neighbour=1,indexpacks=Heisenberg()),
            #SpinTerm('J2',J2,neighbour=2,indexpacks=Heisenberg()),
            #SpinTerm('J1',J1,neighbour=1,indexpacks=Ising('z')),
    ]

    idmrg=iDMRG(
        name=       'iDMRG',
        block=      block,
        vector=     vector,
        terms=      terms,
        config=     idfconfig,
        degfres=    degfres,
        #chain=      EmptyChain(mode='NB',nmax=20,target=None)
        chain=      EmptyChain(mode='QN',nmax=20,target=SpinQN(Sz=0.0))
    )
    print '\n'*2
    print 'Here goes the fDMRG'
    fdmrg=fDMRG(
        name=       'fDMRG',
        lattice=    idmrg.lattice,
        terms=      terms,
        config=     idfconfig,
        degfres=    degfres,
        chain=      idmrg.chain
    )
    fdmrg.sweep([30,40,60,100])
    print
