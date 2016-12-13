'''
DMRG test.
'''

__all__=['test_dmrg']

def test_dmrg(arg):
    from HamiltonianPy.DMRG import test
    if arg in ('mps','all'):
        test.test_mps()
    if arg in ('mpo','all'):
        test.test_mpo()
    if arg in ('dmrg','all'):
        test.test_dmrg()
