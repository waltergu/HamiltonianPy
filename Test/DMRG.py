'''
DMRG test.
'''

__all__=['test_dmrg']

def test_dmrg(arg):
    if arg in ('block','dmrg','all'):
        from HamiltonianPy.DMRG import test
        test.test_block()
    if arg in ('sdmrg','dmrg','all'):
        from HamiltonianPy.DMRG import test
        test.test_sdmrg()
