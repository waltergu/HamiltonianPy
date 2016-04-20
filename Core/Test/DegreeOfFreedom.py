from HamiltonianPP.Core.BasicClass.GeometryPy import *
from HamiltonianPP.Core.BasicClass.DegreeOfFreedomPy import *
from HamiltonianPP.Core.BasicClass.FermionicPackage import *
def test_deg_fre():
    test_table()
    test_index()
    test_fid()
    test_fermi()
    test_configuration()
    test_indexpackage()

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

def test_indexpackage():
    print 'test_indexpackage'
    a=IndexPackage(1.0,orbitals=[0,0])
    b=IndexPackage(2.0,atoms=[0,0])
    c=IndexPackage(3.0,spins=[0,0])
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
    print
