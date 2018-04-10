from test_LatticePack import *
from test_KSpacePack import *
from unittest import TestSuite

__all__=['extensions','lattice','kspace']

extensions=TestSuite()
extensions.addTest(lattice)
extensions.addTest(kspace)
