from Hamiltonian.Core.BasicClass.IndexPackagePy import *
def test_indexpackage():
    test_indexpackage_body()
    test_indexpackage_functions()

def test_indexpackage_body():
    a=IndexPackage(1.0,orbitals=[0,0])
    b=IndexPackage(2.0,atoms=[0,0])
    c=IndexPackage(3.0,spins=[0,0])
    print a
    print b
    print c
    print c+a+b
    print c+(a+b)
    print "a*b:\n",a*b
    print "a*b*c:\n",a*b*c
    print "a*(b*c):\n",a*(b*c)
    print "a*2.0j:\n",a*2.0j

def test_indexpackage_functions():
    print sigmax('sp')
    print sigmax('ob')
    print sigmax('sl')
    print sigmay('sp')
    print sigmay('ob')
    print sigmay('sl')
    print sigmaz('sp')
    print sigmaz('ob')
    print sigmaz('sl')*(3+2.0j)
