'''
Basics test.
'''

from HamiltonianPy.Basics.test import *
from HamiltonianPy.Basics.QuantumNumber.test import *
from HamiltonianPy.Basics.FockPackage.test import *
from HamiltonianPy.Basics.SpinPackage.test import *
from HamiltonianPy.Basics.Extensions.test import *
from unittest import TestSuite

basics=TestSuite()
basics.addTest(utilities)
basics.addTest(basespace)
basics.addTest(geometry)
basics.addTest(degreeoffreedom)
basics.addTest(engineapp)
basics.addTest(quantumnumber)
basics.addTest(fock)
basics.addTest(spin)
basics.addTest(extensions)
