'''
============
Introduction
============

This subpackage is the basic for all Hamiltonian-based algorithms. Generally, to apply such a algorithm, the procedure goes as follows:

#. Construct the lattice of the system;
#. Describe the internal degrees of freedom on the lattice;
#. Provide the terms of the Hamiltonian;
#. Select an algorithm as the engine;
#. Assign the tasks and run them by registering apps on the engine.
#. Analyze the results.

The modules in this subpackage deals with Proc.1~3 and help the algorithms implemented in other packages with Proc.4~5.
Concretely, the roles they play are

+-----------------------+-----------------------------------------------------------------------------------------------------------+
|   MODULES             |   DESCRIPTION                                                                                             |
+=======================+===========================================================================================================+
|   `Utilities`         |   The utilities of the subpackage.                                                                        |
+-----------------------+-----------------------------------------------------------------------------------------------------------+
|   `Geometry`          |   Provide enormous functions and classes to describe and construct any lattice in 1,2 and 3 dimension.    |
+-----------------------+-----------------------------------------------------------------------------------------------------------+
|   `DegreeOfFreedom`   |   Define the way to describe the internal degrees of freedom.                                             |
+-----------------------+-----------------------------------------------------------------------------------------------------------+
|   `Term`              |   Define the way to describe the terms of the Hamiltonian.                                                |
+-----------------------+-----------------------------------------------------------------------------------------------------------+
|   `Operator`          |   Define the way to describe the operators of the Hamiltonian.                                            |
+-----------------------+-----------------------------------------------------------------------------------------------------------+
|                       |   Provide the methods to generate and update operators based on the bonds of a lattice, the configuration |
|   `Generator`         |   of the internal degrees of freedom and the terms of a Hamiltonian, which synthesizes all the above      |
|                       |   components and is the core of the subpackage.                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------+
|   `EngineApp`         |   Define the framework of the interactions between algorithms and tasks.                                  |
+-----------------------+-----------------------------------------------------------------------------------------------------------+
|   `BaseSpace`         |   Define a unified way to describe parameter spaces (e.g. k-space, etc).                                  |
+-----------------------+-----------------------------------------------------------------------------------------------------------+
|   `FockPackage`       |   An extension to deal with fermionic and bosonic systems.                                                |
+-----------------------+-----------------------------------------------------------------------------------------------------------+
|   `SpinPackage`       |   An extension to deal with spin systems.                                                                 |
+-----------------------+-----------------------------------------------------------------------------------------------------------+
|   `QuantumNumber`     |   An extension to deal with good quantum numbers.                                                         |
+-----------------------+-----------------------------------------------------------------------------------------------------------+
|   `Extensions`        |   The extensions of the subpackage.                                                                       |
+-----------------------+-----------------------------------------------------------------------------------------------------------+
'''

from Utilities import *
from BaseSpace import *
from Geometry import *
from DegreeOfFreedom import *
from Operator import *
from Term import *
from EngineApp import *
from Generator import *
from FockPackage import *
from SpinPackage import *
from QuantumNumber import *
from Extensions import *
