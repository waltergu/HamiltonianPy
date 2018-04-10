'''
VCA test.
'''

from HamiltonianPy.VCA.test import *
from unittest import TestSuite

vcaall=TestSuite()
vcaall.addTest(vca)
vcaall.addTest(vcacct)
