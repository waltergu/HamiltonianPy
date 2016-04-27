'''
ED test.
'''

__all__=['test_ed']

def test_ed(arg):
    if arg in ('ed','all'):
        from HamiltonianPP.ED import test
        test.test_ed()
