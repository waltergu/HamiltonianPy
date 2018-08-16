# HamiltonianPy

In this package, we provide a unified framework to make a generic quantum lattice system programmable. Two central concepts are proposed to acheive this:

* unit-cell description ansatz (UCDA), and
* engine-app framework (EAF).

UCDA incoporates the geometry of the latttice, the internal degrees of freedom of the Hilbert space and the terms of the Hamiltonian as a whole, and generates all the needed operators of the system. EAF handles the interactions between the core algotithms (engine) and the user-defined tasks (app) as well as the logging of the program and the recording of the results, etc. Fundamental classes and functions are offered in the subpackage `Basics` as the APIs. Based on this, several algorithms are implemented, including:

* subpackage `FreeSystem`: tight binding approximation (TBA) and Bogoliubov de Gennes (BdG) equations for fermionic systems,
* subpackage `ED`: exact diagonalization (ED) for electron and spin systems,
* subpackage `VCA`: cluster pertubation theory (CPT) and variational cluster approach (VCA) for fermionic systems,
* subpackage `TensorNetwork`: tensor network represetation of Hamiltonians (MPO) and states (MPS),
* subpackage `DMRG`: density matrix renormalization group (DMRG) for electron and spin systems,
* subpackage `FBFM`: spin wave theory for flat band ferromagnets (FBFM).

Dependency
----------
It works with python 3.6, and requires several packages:
* numpy latest version
* scipy latest version
* numba latest version
* matplotlib latest version
* mpi4py latest version

Notes
-----
python 2.x is no longer supported.

Source code
-----------
https://github.com/waltergu/HamiltonianPy


Authors
-------
* Zhao-Long Gu (NJU)


Contact
-------
waltergu1989@gmail.com
