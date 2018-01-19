'''
############
Introduction
############

In this package, we provide a unified framework to make a generic quantum lattice system programmable. Two central concepts are proposed to acheive this:

* unit-cell description ansatz (UCDA), and
* engine-app framework (EAF).

UCDA incoporates the geometry of the latttice, the internal degrees of freedom of the Hilbert space and the terms of the Hamiltonian as a whole, and generates all the needed operators of the system. EAF handles the interactions between the core algotithms (engine) and the user-defined tasks (app) as well as the logging of the program and the recording of the results, etc. Fundamental classes and functions are offered in the subpackage `Basics` as the APIs. Based on this, several algorithms are implemented, including:

* subpackage `FreeSystem`: tight binding approximation (TBA) and Bogoliubov de Gennes (BdG) equations for fermionic systems,
* subpackage `ED`: exact diagonalization (ED) for electron and spin systems,
* subpackage `VCA`: cluster pertubation theory (CPT) and variational cluster approach (VCA) for fermionic systems,
* subpackage `DMRG`: density matrix renormalization group (DMRG) for electron and spin systems,
* subpackage `FBFM`: spin wave theory for flat band ferromagnets (FBFM).

It works with python 2.7, and requires several packages:

* numpy latest version
* scipy latest version
* matplotlib latest version
* mpi4py

For source code, please visit https://github.com/waltergu/HamiltonianPy.

###########
Subpackages
###########

===============   ==========================================================================
SUBPACKAGE        DESCRIPTION
===============   ==========================================================================
`Basics`          The general framework of the Hamiltonian-based algorithms
`Misc`            Miscellaneous functions and classes as supports to the algorithms
`FreeSystem`      Free fermionic system algorithm, including TBA, BdG, Floquet and SCMF
`FBFM`            Spin excitations for flat band ferromagnets
`ED`              Exact diagonalization for fermionic systems and spin systems
`VCA`             CPT and VCA for fermionic systems
`TensorNetwork`   Tensor, MPS and MPO
`DMRG`            Density matrix renormalization group for electron systems and spin systems
`Management`      Project construction and maintenance
===============   ==========================================================================

########
Contents
########

.. toctree::
   :numbered:
   :maxdepth: 4

   Basics/index
   Misc/index
   FreeSystem/index
   FBFM/index
   ED/index
   VCA/index
   TensorNetwork/index
   DMRG/index
   Management/index
'''

from Basics import *
