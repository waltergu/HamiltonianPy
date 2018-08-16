'''
============
Introduction
============

This subpackage implements the exact diagonalization for fermionic/hard-core-bosonic systems and spin systems.

=======     =============================================================
MODULES     DESCRIPTION
=======     =============================================================
`ED`        Base class for exact diagonalization
`FED`       Exact diagonalization for fermionic/hard-core-bosonic systems
`SED`       Exact diagonalization for spin systems
=======     =============================================================
'''

from .ED import *
from .FED import *
from .SED import *
