'''
Math test.
'''

__all__=['test_math']

def test_math(arg):
    if arg in ('lanczos','math','all'):
        from HamiltonianPP.Math.test.Lanczos import *
        test_lanczos()
