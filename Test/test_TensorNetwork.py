'''
TensorNetwork test.
'''

from HamiltonianPy.TensorNetwork.test import *
from unittest import TestSuite

tensornetwork=TestSuite()
tensornetwork.addTest(tensor)
tensornetwork.addTest(structure)
tensornetwork.addTest(mps)
tensornetwork.addTest(mpo)
