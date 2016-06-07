from numpy import *
from HamiltonianPP.Basics import *
from collections import namedtuple
from scipy.sparse import kron,csr_matrix

__all__=['DEFAULT_SPIN_PRIORITY','SID','Spin','SpinMatrix','OperatorS','SOptRep']

def test():
    test_spin_matrix()

def test_spin_matrix():
    N=2
    print SpinMatrix((N,'x'),dtype=float64)
    print SpinMatrix((N,'y'),dtype=complex128)
    print SpinMatrix((N,'z'),dtype=float64)
    print SpinMatrix((N,'+'),dtype=float64)
    print SpinMatrix((N,'-'),dtype=float64)

if __name__=='__main__':
    test()
