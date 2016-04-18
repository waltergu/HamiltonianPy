from Hamiltonian.Core.BasicAlgorithm.LanczosPy import *
from scipy.sparse.linalg import eigsh
def test_lanczos():
    a=Lanczos(csr_matrix(array([[0.,-1.,-1.,0.],[-1.,0.,0.,-1.],[-1.,0.,0.,-1.],[0.,-1.,-1.,0.]])))
    print a.matrix.todense()
    print a.eig(job='v')
    print eigsh(a.matrix,which='SA',k=1)
