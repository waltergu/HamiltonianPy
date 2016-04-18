from Hamiltonian.Core.BasicClass.NamePy import *
def test_name():
    a=Name('Hexagon','CPT')
    a.update({0:1.0})
    a.update({1:2.0+2.0j})
    print a
