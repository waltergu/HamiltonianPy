'''
Log test.
'''

__all__=['test_log']

from HamiltonianPy.Basics.Log import *
from time import sleep
import numpy as np

def test_log():
    print 'test_log'
    test_timers()
    test_info()

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
    print

def test_info():
    print 'test_info'
    info=Info('nnz','gse','overlap','nbasis')
    info['nnz']=10
    info['gse']=-0.12345667
    info['overlap']=0.99999899
    info['nbasis']=200
    print info
    print
