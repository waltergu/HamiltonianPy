from Hamiltonian.Core.BasicClass.IndexPy import *
def test_index():
    test_index_body()
    test_index_functions()

def test_index_body():
    a=Index(0,0,0,CREATION)
    b=Index(0,0,1,CREATION)
    print 'a:\n',a
    print 'b:\n',b
    print 'a==a.dagger: ',a==a.dagger

def test_index_functions():
    a=Index(4,3,2,1)
    b='4 3 2 1'
    print a.to_str('NSCOP'),a.to_str('NCSO'),a.to_str('CSNO'),a.to_str('SCON')
    print to_index(b,'COSN')
    print to_index('H6 '+b,'PSOCN')
