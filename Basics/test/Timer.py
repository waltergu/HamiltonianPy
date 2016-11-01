'''
Timer test.
'''

__all__=['test_timer']

from HamiltonianPy.Basics.Timer import *
from time import sleep
import numpy as np

def test_timer():
    print 'test_timer'
    np.random.seed()
    logger=TimerLogger('Preparation','Diagonalization','Truncation','Total')
    logger.proceed('Total')
    for i in xrange(4):
        for key in logger.keys:
            if key!='Total':
                logger.proceed(key)
                sleep(np.random.random())
                logger.suspend(key)
                logger.record(key)
        logger.record('Total')
        print logger,'\n'
    print
