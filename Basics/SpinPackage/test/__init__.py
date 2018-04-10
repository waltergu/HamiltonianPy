from test_DegreeOfFreedom import *
from test_Operator import *
from test_OperatorRepresentation import *
from test_Term import *
from unittest import TestSuite

__all__=['spin','sdegreeoffreedom','soperator','soptrep','sterm']

spin=TestSuite()
spin.addTest(sdegreeoffreedom)
spin.addTest(soperator)
spin.addTest(soptrep)
spin.addTest(sterm)
