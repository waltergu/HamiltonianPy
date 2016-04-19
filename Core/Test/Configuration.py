from HamiltonianPP.Core.BasicClass.GeometryPy import *
from HamiltonianPP.Core.BasicClass.ConfigurationPy import *
def test_configuration():
    test_fid()
    test_index()
    test_fermi()
    #test_index_functions()

def test_fid():
    print 'test_fid'
    fid=FID(orbital=1,spin=1,nambu=CREATION)
    print 'fid:',fid
    print 'fid.dagger:',fid.dagger
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
    print

def test_fermi():
    print 'test_fermi'
    


def test_index_functions():
    a=Index(4,3,2,1)
    b='4 3 2 1'
    print a.to_str('NSCOP'),a.to_str('NCSO'),a.to_str('CSNO'),a.to_str('SCON')
    print to_index(b,'COSN')
    print to_index('H6 '+b,'PSOCN')
