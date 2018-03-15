'''
Structure test.
'''

__all__=['test_structure']

from HamiltonianPy import *
from HamiltonianPy.TensorNetwork.Structure import *

def test_structure():
    print 'test_structure'
    test_degfretree()

def test_degfretree():
    print 'test_degfretree'
    priority,layers=DEGFRE_FERMIONIC_PRIORITY,DEGFRE_FERMIONIC_LAYERS
    config=IDFConfig(priority,pids=[PID(scope,site) for site in (0,1) for scope in (0,1)],map=lambda pid: Fermi(norbital=1,nspin=2,nnambu=1))

    print 'fermi with mode=="NB"'
    tree=DegFreTree(mode='NB',layers=layers,priority=priority,leaves=config.table(mask=['nambu']).keys(),map=lambda index: 2)
    for layer in layers:
        print 'layer: %s'%(layer,)
        for i,index in enumerate(tree.indices(layer)):
            print 'i,index,tree[index]: %s, %s, %s'%(i,repr(index),tree[index])
        print

    print 'fermi with mode=="QN"'
    tree=DegFreTree(mode='QN',layers=layers,priority=priority,leaves=config.table(mask=['nambu']).keys(),map=lambda index: SzPQNS(-0.5 if index.spin==0 else 0.5))
    for layer in layers:
        print 'layer: %s'%(layer,)
        for i,index in enumerate(tree.indices(layer)):
            print 'i,index,len(tree[index]),tree[index]: %s, %s, %s, %s'%(i,repr(index),len(tree[index]),tree[index])
        print

    priority,layers=DEGFRE_SPIN_PRIORITY,DEGFRE_SPIN_LAYERS
    config=IDFConfig(priority=DEGFRE_SPIN_PRIORITY,pids=[PID(scope,site) for site in xrange(4) for scope in (0,1)],map=lambda pid: Spin(S=0.5))

    print 'spin with mode=="NB"'
    tree=DegFreTree(mode='NB',layers=layers,priority=priority,leaves=config.table(mask=[]).keys(),map=lambda index: 2)
    for layer in layers:
        print 'layer: %s'%(layer,)
        for i,index in enumerate(tree.indices(layer)):
            print 'i,index,tree[index]: %s, %s, %s'%(i,repr(index),tree[index])
        print

    print 'spin with mode=="QN"'
    tree=DegFreTree(mode='QN',layers=layers,priority=priority,leaves=config.table(mask=[]).keys(),map=lambda index: SQNS(0.5))
    for layer in layers:
        print 'layer: %s'%(layer,)
        for i,index in enumerate(tree.indices(layer)):
            print 'i,index,len(tree[index]),tree[index]: %s, %s, %s, %s'%(i,repr(index),len(tree[index]),tree[index])
        print
    print
