'''
MPI test.
'''

__all__=['test_mpi']

from HamiltonianPy.Basics.MPI import *
import numpy as np
from mpi4py import MPI

def test_mpi():
    if MPI.COMM_WORLD.Get_rank()==0:
        print 'test_mpi'
    def test(n):
        with open('%s.dat'%n,'w+') as fout:
            fout.write(str(np.array(xrange(10))+n))
    mpirun(test,[(i,) for i in range(10)])
