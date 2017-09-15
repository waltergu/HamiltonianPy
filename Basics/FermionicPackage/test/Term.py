'''
Fermionic term test.
'''

__all__=['test_fermionic_term']

from HamiltonianPy.Basics.Geometry import *
from HamiltonianPy.Basics.DegreeOfFreedom import *
from HamiltonianPy.Basics.Operator import *
from HamiltonianPy.Basics.FermionicPackage import *

def test_fermionic_term():
    test_quadratic()
    test_hubbard()
    test_coulomb()

def test_quadratic():
    print 'test_quadratic'
    p1=Point(pid=PID(site=0,scope='WG'),rcoord=[0.0,0.0],icoord=[0.0,0.0])
    p2=Point(pid=PID(site=1,scope='WG'),rcoord=[1.0,0.0],icoord=[0.0,0.0])
    bond=Bond(neighbour=1,spoint=p1,epoint=p2)

    config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY)
    config[p1.pid]=Fermi(atom=0,norbital=2,nspin=2,nnambu=2)
    config[p2.pid]=Fermi(atom=1,norbital=2,nspin=2,nnambu=2)

    a=Hopping('t',1,neighbour=1,indexpacks=sigmax("SL")*sigmax("SP"))
    b=Onsite('mu',1,indexpacks=sigmaz("SP")*sigmay("OB"))
    c=Pairing('delta',1,neighbour=1,indexpacks=sigmaz("SP")+sigmay("OB"))
    print 'a: %s\nb: %s\nc: %s\n'%(a,b,c)

    l=Lattice.compose(name="WG",points=[p1,p2])
    table=config.table(mask=[])
    a=Hopping('t1',1.0,neighbour=1,indexpacks=sigmaz("SP"))
    b=Onsite('mu',1.0,indexpacks=sigmaz("SP"))
    c=Pairing('delta',1.0,neighbour=1,indexpacks=sigmaz("SP"))
    opts=Operators()
    for bond in l.bonds:
        opts+=a.operators(bond,config,table)
        opts+=b.operators(bond,config,table)
        opts+=c.operators(bond,config,table)
    print 'opts:\n%s'%repr(opts)
    print

def test_hubbard():
    print 'test_hubbard'
    p1=Point(PID(site=0,scope="WG"),rcoord=[0.0,0.0],icoord=[0.0,0.0])
    config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY)
    config[p1.pid]=Fermi(norbital=2,nspin=2,nnambu=1)
    l=Lattice.compose(name="WG",points=[p1])
    table=config.table(mask=[])
    a=Hubbard('UUJJ',[20.0,12.0,5.0,5.0])
    print 'a: %s'%a
    opts=Operators()
    for bond in l.bonds:
        opts+=a.operators(bond,config,table)
    print 'opts:\n%s'%repr(opts)
    print

def test_coulomb():
    print 'test_coulomb'
    p1=Point(PID(site=0,scope='WG'),rcoord=[0.0,0.0],icoord=[0.0,0.0])
    p2=Point(PID(site=1,scope='WG'),rcoord=[1.0,0.0],icoord=[0.0,0.0])
    l=Lattice.compose(name='WG',points=[p1,p2],nneighbour=1)
    config=IDFConfig(priority=DEFAULT_FERMIONIC_PRIORITY,pids=l.pids,map=lambda pid: Fermi(atom=pid.site%2,norbital=1,nspin=2,nnambu=1))
    table=config.table(mask=['nambu'])
    a=Coulomb('U',1.0,neighbour=0,indexpacks=(sigmap('sp'),sigmam('sp')))
    b=Coulomb('V',8.0,neighbour=1,indexpacks=(sigmaz('sp'),sigmaz('sp')))
    print 'a: %s\nb: %s\n'%(a,b)
    opts=Operators()
    for bond in l.bonds:
        opts+=a.operators(bond,config,table)
        opts+=b.operators(bond,config,table)
    print 'opts:\n%s'%repr(opts)
    print
