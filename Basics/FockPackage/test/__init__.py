from .test_DegreeOfFreedom import *
from .test_Operator import *
from .test_Term import *
from .test_Basis import *
from .test_OperatorRepresentation import *
from unittest import TestSuite

__all__=['fock','fockdegreeoffreedom','fockoperator','fockterm','fockbasis','fockoptrep']

fock=TestSuite()
fock.addTest(fockdegreeoffreedom)
fock.addTest(fockoperator)
fock.addTest(fockterm)
fock.addTest(fockbasis)
fock.addTest(fockoptrep)
