'''
VCA test.
'''

__all__=['test_vca']

def test_vca(arg):
    if arg in ('vca','all'):
        from HamiltonianPy.VCA import test
        test.test_vca()
    if arg in ('vcacct','all'):
        from HamiltonianPy.VCA import test
        test.test_vcacct()
