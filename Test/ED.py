'''
ED test.
'''

__all__=['test_ed']

def test_ed(arg):
    if arg in ('ed_square','ed','all'):
        from HamiltonianPP.ED.test.ED import *
        test_ed_square()
