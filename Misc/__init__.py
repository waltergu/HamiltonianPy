'''
============
Introduction
============

This subpackage contains miscellaneous classes and functions as aids to the algorithms.

================    ====================================================
MODULES             DESCRIPTION
================    ====================================================
`Tree`              Tree data structure
`BerryCurvature`    Berry curvature for filled bands
`Parity`            Parity of a permutation
`MPI`               A simple wrapper for mpi4py
`Lanczos`           Lanczos algorithm for large spare Hermitian matrices
`Linalg`            Linear algebras
`Calculus`          Calculus related
================    ====================================================
'''

from MPI import *
from Tree import *
from Parity import *
from BerryCurvature import *
from Lanczos import *
from Linalg import *
from Calculus import *
