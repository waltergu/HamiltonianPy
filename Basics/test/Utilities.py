'''
Utilities test.
'''

__all__=['test_utilities']

from HamiltonianPy.Basics.Utilities import *
from mpi4py import MPI
import numpy as np
from time import sleep

def test_utilities():
    print 'test_utilities'
    test_timers()
    test_sheet()
    test_mpi()
    print

def test_timers():
    print 'test_timers'
    np.random.seed()
    keys=['Preparation','Diagonalization','Truncation']
    timers=Timers(*keys)
    for i in xrange(4):
        for key in keys:
            with timers.get(key):
                sleep(np.random.random())
        timers.record()
        print timers,'\n'
        timers.graph()
    timers.close()
    print

def test_sheet():
    print 'test_info'
    info=Sheet(rows=('nnz','gse','overlap','nbasis'),cols=('value',))
    info['nnz']=10
    info['gse']=-0.12345667
    info['overlap']=0.99999899
    info['nbasis']=200
    print info
    print

def test_mpi():
    print 'test_mpi'
    if MPI.COMM_WORLD.Get_rank()==0:
        print 'test_mpi'
    def test(n):
        with open('%s.dat'%n,'w+') as fout:
            fout.write(str(np.array(xrange(4))+n))
    mpirun(test,[(i,) for i in range(4)])
    print
