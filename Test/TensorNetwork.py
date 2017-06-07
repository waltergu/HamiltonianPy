'''
TensorNetwork test.
'''

__all__=['test_tensornetwork']

def test_tensornetwork(arg):
    if arg in ('tensor','tensornetwork','all'):
        from HamiltonianPy.TensorNetwork.test import test_tensor
        test_tensor()
    if arg in ('mps','tensornetwork','all'):
        from HamiltonianPy.TensorNetwork.test import test_mps
        test_mps()
    if arg in ('mpo','tensornetwork','all'):
        from HamiltonianPy.TensorNetwork.test import test_mpo
        test_mpo()
