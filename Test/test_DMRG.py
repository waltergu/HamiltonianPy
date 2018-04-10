'''
DMRG test.
'''

__all__=['test_dmrg']

def test_dmrg(arg):
    from HamiltonianPy.DMRG import test
    if arg in ('dmrg','all'):
        test.test_dmrg()
