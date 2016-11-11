'''
DMRG test.
'''

__all__=['test_dmrg']

def test_dmrg(arg):
    from HamiltonianPy.DMRG import test
    if arg in ('mps','dmrg','all'):
        test.test_mps()
    if arg in ('mpo','dmrg','all'):
        test.test_mpo()
    if arg in ('fdmrg','dmrg','all'):
        test.test_fdmrg()
