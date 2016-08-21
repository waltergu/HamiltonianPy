'''
QMC test.
'''

__all__=['test_qmc']

def test_qmc(arg):
    if arg in ('qmc','dqmc','all'):
        from HamiltonianPy.QMC import test
        test.test_dqmc()
