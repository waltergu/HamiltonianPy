'''
------------
Introduction
------------

This subpackage is an extension to support the spin systems.

========================    =========================================================================================================
MODULES                     DESCRIPTION
========================    =========================================================================================================
`DegreeOfFreedom`           defines the spin degrees of freedom
`Operator`                  defines the spin operators
`OperatorRepresentation`    provides the method to get the sparse matrix representations of spin operators on occupation number basis
`Term`                      defines the spin terms
========================    =========================================================================================================
'''

from .DegreeOfFreedom import *
from .Operator import *
from .OperatorRepresentation import *
from .Term import *
