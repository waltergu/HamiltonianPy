'''
DegFreTree test.
'''

__all__=['test_degfretree']

from HamiltonianPy import *
from HamiltonianPy.TensorNetwork.DegFreTree import *

def test_degfretree():
    print 'test_degfretree'
    config=IDFConfig(priority=DEGFRE_FERMIONIC_PRIORITY)
    for scope in xrange(2):
        config[PID(scope=scope,site=0)]=Fermi(norbital=1,nspin=2,nnambu=1)
        config[PID(scope=scope,site=1)]=Fermi(norbital=1,nspin=2,nnambu=1)

    layers=DEGFRE_FERMIONIC_LAYERS
    priority=DEGFRE_FERMIONIC_PRIORITY
    leaves=config.table(mask=['nambu']).keys()

    map_nb=lambda index: 2
    tree=DegFreTree(mode='NB',layers=layers,priority=priority,leaves=leaves,map=map_nb)
    for layer in layers:
        print 'layer:',layer
        for i,index in enumerate(tree.indices(layer)):
            print i,index,tree[index]
        print

    a,b,c=SPQN((0,0.0)),SPQN((1,-0.5)),SPQN((1,0.5))
    map_qn=lambda index: QuantumNumbers('C',((a,b),(1,1)),QuantumNumbers.COUNTS) if index.spin==0 else QuantumNumbers('C',((a,c),(1,1)),QuantumNumbers.COUNTS)
    tree=DegFreTree(mode='QN',layers=layers,priority=priority,leaves=leaves,map=map_qn)
    for layer in layers:
        print 'layer',layer
        for i,index in enumerate(tree.indices(layer)):
            print 'i,index,len(tree[index]): %s, %s, %s'%(i,index,len(tree[index]))
            print 'tree[index]: %s'%tree[index]
        print

    config=IDFConfig(priority=DEGFRE_SPIN_PRIORITY)
    for site in xrange(4):
        config[PID(scope=1,site=site)]=Spin(S=0.5)
        config[PID(scope=2,site=site)]=Spin(S=0.5)

    layers=DEGFRE_SPIN_LAYERS
    priority=DEGFRE_SPIN_PRIORITY
    leaves=config.table(mask=[]).keys()

    map_nb=lambda index: 2
    tree=DegFreTree(mode='NB',layers=layers,priority=priority,leaves=leaves,map=map_nb)
    for layer in layers:
        print 'layer:',layer
        for i,index in enumerate(tree.indices(layer)):
            print i,index,tree[index]
        print

    map_qn=lambda index: SQNS(0.5)
    tree=DegFreTree(mode='QN',layers=layers,priority=priority,leaves=leaves,map=map_qn)
    for layer in layers:
        print 'layer:',layer
        for i,index in enumerate(tree.indices(layer)):
            print 'i,index,len(tree[index]): %s, %s, %s'%(i,index,len(tree[index]))
            print 'tree[index]: %s'%tree[index]
        print
    print
