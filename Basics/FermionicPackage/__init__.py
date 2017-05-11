'''
------------
Introduction
------------

This subpackage is an extension to support the fermionic systems.

========================    ==============================================================================================================
MODULES                     DESCRIPTION
========================    ==============================================================================================================
`Basis`                     defines the occupation number basis
`DegreeOfFreedom`           defines the fermionic degrees of freedom
`Operator`                  defines the fermionic operators
`OperatorRepresentation`    provides the method to get the sparse matrix representations of fermionic operators on occupation number basis
`Term`                      defines the fermionic terms
========================    ==============================================================================================================
'''

from DegreeOfFreedom import *
from Operator import *
from Term import *
from Basis import *
from OperatorRepresentation import *
