from HamiltonianPP.Core.BasicClass.GeometryPy import *
from HamiltonianPP.Core.BasicClass.DegreeOfFreedomPy import *
from HamiltonianPP.Core.BasicClass.FermionicPackagePy import *
def test_deg_fre():
    test_table()
    test_index()
    test_fid()
    test_fermi()
    test_configuration()

def test_table():
    print 'test_table'
    a=Table({'i1':0,'i2':1})
    b=Table({'i3':0,'i4':1})
    c=union([a,b],key=lambda key: key[1])
    print 'a: %s'%a
    print 'b: %s'%b
    print 'union(a,b)(c): %s'%c
    print 'reverse_table(c): %s'%reversed_table(c)
    print 'c["i4"]: %s'%c['i4']
    print 'subset: %s'%subset(c,mask=lambda key: True if key!='i1' else False)
    print

def test_index():
    print 'test_index'
    a=Index(pid=PID(site=(0,0,0)),iid=FID(spin=1))
    b=Index(pid=PID(site=(0,0,1)),iid=FID(nambu=CREATION))
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
    print 'a.table(nambu=False): %s'%a.table(pid=PID(site=(0,0,0)),nambu=False)
    print 'a.table(nambu=True): %s'%a.table(pid=PID(site=(0,0,0)),nambu=True)
    for i in xrange(a.norbital*a.nspin*a.nnambu):
        fid=a.state_index(i)
        print 'a.state_index(%s): '%i,fid
        print 'a.seq_state(%s): %s'%(fid,a.seq_state(fid))
    print

def test_configuration():
    print 'test_configuration'
    a=Configuration(priority=DEFAULT_FERMIONIC_PRIORITY)
    a[PID(site=(0,0,0))]=Fermi(atom=0,norbital=1,nspin=2,nnambu=2)
    a[PID(site=(0,0,1))]=Fermi(atom=1,norbital=1,nspin=2,nnambu=2)
    print 'a: %s'%a
    print 'a.table:%s'%a.table(nambu=True)
    print
