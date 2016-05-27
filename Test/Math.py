'''
Math test.
'''

__all__=['test_math']

def test_math(arg):
    if arg in ('lanczos','math','all'):
        from HamiltonianPy.Math.test.Lanczos import *
        test_lanczos()
    if arg in ('tensor','math','all'):
        from HamiltonianPy.Math.test.Tensor import *
        test_tensor()
    if arg in ('tree','math','all'):
        from HamiltonianPy.Math.test.Tree import *
        test_tree()
    if arg in ('mps','math','all'):
        from HamiltonianPy.Math.test.MPS import *
        test_mps()