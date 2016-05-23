'''
MERA test.
'''

__all__=['test_mera']

def test_mera(arg):
    if arg in ('mera','all'):
        from HamiltonianPy.MERA import test
        test.test_mera()
