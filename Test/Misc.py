'''
Misc test.
'''

__all__=['test_misc']

def test_misc(arg):
    if arg in ('mpi','misc','all'):
        from HamiltonianPy.Misc.test import test_mpi
        test_mpi()
    if arg in ('tree','misc','all'):
        from HamiltonianPy.Misc.test import test_tree
        test_tree()
    if arg in ('lanczos','linalg','misc','all'):
        from HamiltonianPy.Misc.Linalg.test import test_lanczos
        test_lanczos()
    if arg in ('kron','linalg','misc','all'):
        from HamiltonianPy.Misc.Linalg.test import test_kron
        test_kron()
