'''
Math test.
'''

__all__=['test_math']

def test_math(arg):
    if arg in ('lanczos','math','all'):
        from HamiltonianPP.Math.test.Lanczos import *
        test_lanczos()
    if arg in ('tensor','math','all'):
        from HamiltonianPP.Math.test.Tensor import *
        test_tensor()
    if arg in ('tree','math','all'):
        from HamiltonianPP.Math.test.Tree import *
        test_tree()
    if arg in ('mps','math','all'):
        from HamiltonianPP.Math.test.MPS import *
        test_mps()
