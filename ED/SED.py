'''
==========================
Spin exact diagonalization
==========================

Exact diagonalization for spin systems, including:
    * classes: SED
'''

__all__=['SED']

from .ED import *
from collections import OrderedDict
import numpy as np
import HamiltonianPy as HP

class SED(ED):
    '''
    Exact diagonalization for a spin system.

    Attributes
    ----------
    qnses : QNSConfig
        The configuration of the quantum numbers.
    '''

    def __init__(self,lattice,config,qnses=None,sectors=None,terms=(),dtype=np.complex128,**karg):
        '''
        Constructor.

        Parameters
        ----------
        lattice : Lattice
            The lattice of the system.
        config : IDFConfig
            The configuration of the internal degrees of freedom on the lattice.
        qnses : QNSConfig, optional
            The configuration of the quantum numbers.
        sectors : iterable of QuantumNumber, optional
            The target spaces of the system.
        terms : list of Term, optional
            The terms of the system.
        dtype : np.float32, np.float64, np.complex64, np.complex128
            The data type of the matrix representation of the Hamiltonian.
        '''
        if sectors is not None:
            assert isinstance(qnses,HP.QNSConfig)
            assert config.priority==qnses.priority
        self.lattice=lattice
        self.config=config
        self.qnses=qnses
        self.sectors=set(sectors) if sectors is not None else {None}
        self.terms=terms
        self.dtype=dtype
        self.sector=None
        self.generator=HP.Generator(bonds=lattice.bonds,config=config,table=config.table(),terms=terms,dtype=dtype)
        if self.map is None: self.parameters.update(OrderedDict((term.id,term.value) for term in terms))
        self.operators=self.generator.operators
        self.logging()

    def matrix(self,sector=None,reset=True):
        '''
        The matrix representation of the Hamiltonian.

        Parameters
        ----------
        sector : QuantumNumber, optional
            The sector of the matrix representation of the Hamiltonian.
        reset : logical, optional
            True for resetting the matrix cache and False for not.

        Returns
        -------
        csr_matrix
            The matrix representation of the Hamiltonian.
        '''
        if reset:
            table=self.generator.table
            if self.sectors=={None} or len(table)<=1:
                self.generator.setmatrix(sector,HP.soptrep,table)
            else:
                assert sector is not None
                cut,qnses=len(table)//2,[self.qnses[index] for index in sorted(table,key=table.get)]
                lqns,lpermutation=HP.QuantumNumbers.kron(qnses[:cut]).sorted(history=True)
                rqns,rpermutation=HP.QuantumNumbers.kron(qnses[cut:]).sorted(history=True)
                subslice=HP.QuantumNumbers.kron([lqns,rqns]).subslice(targets=(sector,))
                rcs=(subslice//len(rqns),subslice%len(rqns),np.zeros(len(lqns)*len(rqns),dtype=np.int64))
                rcs[2][subslice]=range(len(subslice))
                self.generator.setmatrix(sector,HP.soptrep,table,cut=cut,permutations=(lpermutation,rpermutation),rcs=rcs)
        return self.generator.matrix(sector)
