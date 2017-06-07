'''
Fermionic degree of freedom test.
'''

__all__=['test_fermionic_deg_fre']

from HamiltonianPy.Basics.Geometry import *
from HamiltonianPy.Basics.DegreeOfFreedom import *
from HamiltonianPy.Basics.FermionicPackage import *
from HamiltonianPy.Basics.QuantumNumber import *

def test_fermionic_deg_fre():
    test_index()
    test_fid()
    test_fermi()
    test_idfconfig()
    test_degfretree()
    test_indexpack()

def test_index():
    print 'test_index'
    a=Index(pid=PID(site=0),iid=FID(spin=1))
    b=Index(pid=PID(site=1),iid=FID(nambu=CREATION))
    c=b.replace(nambu=1-b.nambu)
    print 'a:',a
    print 'b:',b
    print 'c:',c
    print 'c.scope,c.site,c.orbital,c.spin,c.nambu: %s,%s,%s,%s,%s'%(c.scope,c.site,c.orbital,c.spin,c.nambu)
    print 'c.to_tuple(["site","orbital","spin","nambu","scope"]): ',c.to_tuple(["site","orbital","spin","nambu","scope"])
    print

def test_fid():
    print 'test_fid'
    fid=FID(orbital=1,spin=1,nambu=CREATION)
    print 'fid:',fid
    print 'fid.dagger:',fid.dagger
    print

def test_fermi():
    print 'test_fermi'
    a=Fermi(atom=0,norbital=2,nspin=2,nnambu=2)
    b=Fermi(atom=1,norbital=1,nspin=2,nnambu=2)
    print 'a: %s'%a
    print 'b: %s'%b
    print 'a!=b: %s'%(a!=b)
    print 'a.indices(mask=["nambu"]): %s'%a.indices(pid=PID(site=0),mask=['nambu'])
    print 'a.indices(mask=[]): %s'%a.indices(pid=PID(site=0),mask=[])
    for i in xrange(a.norbital*a.nspin*a.nnambu):
        fid=a.state_index(i)
        print 'a.state_index(%s): '%i,fid
        print 'a.seq_state(%s): %s'%(fid,a.seq_state(fid))
    print

def test_idfconfig():
    print 'test_idfconfig'
    a=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY)
    a[PID(site=0)]=Fermi(atom=0,norbital=1,nspin=2,nnambu=2)
    a[PID(site=1)]=Fermi(atom=1,norbital=1,nspin=2,nnambu=2)
    print 'a: %s'%a
    print 'a.table:%s'%a.table(mask=[])
    print

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

def test_indexpack():
    print 'test_indexpack'
    a=FermiPack(1.0,orbitals=[0,0])
    b=FermiPack(2.0,atoms=[0,0])
    c=FermiPack(3.0,spins=[0,0])
    print 'a: %s'%a
    print 'b: %s'%b
    print 'c: %s'%c
    print 'c+a+b: %s'%(c+a+b)
    print 'c+(a+b): %s'%(c+(a+b))
    print "a*b: %s"%(a*b)
    print "a*b*c: %s"%(a*b*c)
    print "a*(b*c): %s"%(a*(b*c))
    print "a*2.0j: %s"%(a*2.0j)
    print "sigmax('sp'): %s"%sigmax('sp')
    print "sigmax('ob'): %s"%sigmax('ob')
    print "sigmax('sl'): %s"%sigmax('sl')
    print "sigmay('sp'): %s"%sigmay('sp')
    print "sigmay('ob'): %s"%sigmay('ob')
    print "sigmay('sl'): %s"%sigmay('sl')
    print "sigmaz('sp'): %s"%sigmaz('sp')
    print "sigmaz('ob'): %s"%sigmaz('ob')
    print "sigmaz('sl')*(3+2.0j): %s"%(sigmaz('sl')*(3+2.0j))
    print "sigmax('sp')*sigmay('sp'): %s"%(sigmax('sp')*sigmay('sp'))
    print
