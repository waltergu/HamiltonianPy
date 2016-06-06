'''
DMRG test.
'''

__all__=['test_dmrg']

def test_ed(arg):
    if arg in ('dmrg','all'):
        from HamiltonianPy.DMRG import test
        test.test_ed()
