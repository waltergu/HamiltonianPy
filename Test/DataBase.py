'''
DataBase test.
'''

__all__=['test_database']

def test_database(arg):
    if arg in ('hexagon','database','all'):
        from HamiltonianPP.DataBase.test.Hexagon import *
        test_hexagon()
    if arg in ('triangle','database','all'):
        from HamiltonianPP.DataBase.test.Triangle import *
        test_triangle()
