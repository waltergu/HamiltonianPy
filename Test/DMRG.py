'''
DMRG test.
'''

__all__=['test_dmrg']

def test_dmrg(arg):
    if arg in ('mps','dmrg','all'):
        from HamiltonianPy.DMRG import test
        test.test_mps()
    if arg in ('mpo','dmrg','all'):
        from HamiltonianPy.DMRG import test
        test.test_mpo()
    if arg in ('block','dmrg','all'):
        from HamiltonianPy.DMRG import test
        test.test_block()
    if arg in ('idmrg','dmrg','all'):
        from HamiltonianPy.DMRG import test
        test.test_idmrg()
    if arg in ('fdmrg','dmrg','all'):
        from HamiltonianPy.DMRG import test
        test.test_fdmrg()
