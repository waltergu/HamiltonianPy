'''
Math test.
'''

__all__=['test_math']

def test_math(arg):
    if arg in ('linalg','math','all'):
        from HamiltonianPy.Math.test.linalg import *
        test_linalg()
    if arg in ('tensor','math','all'):
        from HamiltonianPy.Math.test.Tensor import *
        test_tensor()
    if arg in ('tree','math','all'):
        from HamiltonianPy.Math.test.Tree import *
        test_tree()
