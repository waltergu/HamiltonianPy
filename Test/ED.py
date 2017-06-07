'''
ED test.
'''

__all__=['test_ed']

def test_ed(arg):
    if arg in ('fed','ed','all'):
        from HamiltonianPy.ED import test
        test.test_fed()
    if arg in ('sed','ed','all'):
        from HamiltonianPy.ED import test
        test.test_sed()
