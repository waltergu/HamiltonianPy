'''
------------
Introduction
------------

This subpackage is an extension to support the fermionic/bosonic systems whose descriptions are based on Fock space.

========================    ======================================================================================================================
MODULES                     DESCRIPTION
========================    ======================================================================================================================
`Basis`                     defines the occupation number basis
`DegreeOfFreedom`           defines the fermionic/bosonic degrees of freedom
`Operator`                  defines the fermionic/bosonic operators
`OperatorRepresentation`    provides the method to get the sparse matrix representations of fermionic/bosonic operators on occupation number basis
`Term`                      defines the fermionic/bosonic terms
========================    ======================================================================================================================
'''

from .DegreeOfFreedom import *
from .Operator import *
from .Term import *
from .Basis import *
from .OperatorRepresentation import *
