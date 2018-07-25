'''
DMRG test.
'''

from HamiltonianPy.DMRG.test import *
from unittest import TestSuite

dmrg=TestSuite()
dmrg.addTest(idmrg)
dmrg.addTest(fdmrg)