'''
FreeSystem test.
'''

from HamiltonianPy.FreeSystem.test import *
from unittest import TestSuite

fresys=TestSuite()
fresys.addTest(tba)
fresys.addTest(flqt)
fresys.addTest(scmf)
