'''
Degree of freedom test.
'''

__all__=['test_deg_fre']

from HamiltonianPy.Basics.Geometry import *
from HamiltonianPy.Basics.DegreeOfFreedom import *
from HamiltonianPy.Basics.QuantumNumber import *
from HamiltonianPy.Basics.FermionicPackage.DegreeOfFreedom import *
from HamiltonianPy.Basics.SpinPackage.DegreeOfFreedom import *

def test_deg_fre():
    test_table()
    test_deg_fre_tree()

def test_table():
    print 'test_table'
    a=Table(['i1','i2'])
    b=Table(['i3','i4'])
    c=Table.union([a,b],key=lambda key: key[1])
    print 'a: %s'%a
    print 'b: %s'%b
    print 'c=union(a,b): %s'%c
    print 'c.reverse_table: %s'%c.reversed_table
    print 'c["i4"]: %s'%c['i4']
    print 'c.subset: %s'%c.subset(select=lambda key: True if key!='i1' else False)
    print

def test_deg_fre_tree():
    print 'test_deg_fre_tree'
    config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY)
    for scope in xrange(100):
        config[PID(scope=scope,site=0)]=Fermi(norbital=1,nspin=2,nnambu=1)
        config[PID(scope=scope,site=1)]=Fermi(norbital=1,nspin=2,nnambu=1)
    layers=DEFAULT_FERMI_LAYERS
    tree=DegFreTree(config,layers=layers)
    for layer in layers:
        print 'layer:',layer
        for i,index in enumerate(tree.indices(layer)):
            print i,index,tree[index]
        print

    table=config.table()
    a=QuantumNumber([('NE',0,'U1'),('Sz',0,'U1')])
    b=QuantumNumber([('NE',1,'U1'),('Sz',-1,'U1')])
    c=QuantumNumber([('NE',1,'U1'),('Sz',1,'U1')])
    config=QNCConfig(priority=DEFAULT_FERMIONIC_PRIORITY)
    for index in table:
        if index.spin==0:
            config[index.replace(nambu=None)]=QuantumNumberCollection([(a,1),(b,1)])
        else:
            config[index.replace(nambu=None)]=QuantumNumberCollection([(a,1),(c,1)])
    tree=DegFreTree(config,layers=layers)
    for layer in layers:
        print 'layer',layer
        for i,index in enumerate(tree.indices(layer)):
            print i,index,tree[index].n,tree[index]
        print

    layers=DEFAULT_SPIN_LAYERS
    config=IDFConfig(priority=DEFAULT_SPIN_PRIORITY)
    for site in xrange(4):
        config[PID(scope=1,site=site)]=Spin(S=0.5)
        config[PID(scope=2,site=site)]=Spin(S=0.5)
    tree=DegFreTree(config,layers=layers)
    for layer in layers:
        print 'layer:',layer
        for i,index in enumerate(tree.indices(layer)):
            print i,index,tree[index]
        print

    table=config.table()
    a=QuantumNumber([('Sz',-1,'U1')])
    b=QuantumNumber([('Sz',1,'U1')])
    qnc=QuantumNumberCollection([(a,1),(b,1)])
    config=QNCConfig(priority=DEFAULT_SPIN_PRIORITY)
    for index in table:
        config[index]=qnc
    tree=DegFreTree(config,layers=layers)
    for layer in layers:
        print 'layer:',layer
        for i,index in enumerate(tree.indices(layer)):
            print i,index,tree[index]
        print
    print
