'''
############
Introduction
############

This package provides a unified framework to generate the operators of electron and spin systems.
Based on this, it implements several algorithms, including:

* tight binding approximation (TBA) and Bogoliubov de Gennes (BdG) equations for fermionic systems,
* exact diagonalization (ED) for electron systems and spin systems,
* cluster perturbation theory (CPT) and variational cluster approach (VCA) for fermionic systems,
* density matrix renormalization group (DMRG) for electron systems and spin systems,
* spin wave theory for flat band ferromagnets.

It works with python 2.7, and requires several packages:

* numpy latest version
* scipy latest version
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

#################
Contents
#################

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
