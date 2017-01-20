'''
Lanczos test.
'''

__all__=['test_lanczos']

from numpy import *
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from HamiltonianPy.Misc.Linalg import Lanczos

def test_lanczos():
    print 'test_lanczos'
    a=Lanczos(csr_matrix(array([[0.,-1.,-1.,0.],[-1.,0.,0.,-1.],[-1.,0.,0.,-1.],[0.,-1.,-1.,0.]])))
    print a.matrix.todense()
    print a.eig(job='v')
    print eigsh(a.matrix,which='SA',k=1)
    print
