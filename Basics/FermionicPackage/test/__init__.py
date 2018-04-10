from test_DegreeOfFreedom import *
from test_Operator import *
from test_Term import *
from test_Basis import *
from test_OperatorRepresentation import *
from unittest import TestSuite

__all__=['fermionic','fdegreeoffreedom','foperator','fterm','fbasis','foptrep']

fermionic=TestSuite()
fermionic.addTest(fdegreeoffreedom)
fermionic.addTest(foperator)
fermionic.addTest(fterm)
fermionic.addTest(fbasis)
fermionic.addTest(foptrep)
