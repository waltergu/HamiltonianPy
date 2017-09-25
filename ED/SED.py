'''
==========================
Spin exact diagonalization
==========================

Exact diagonalization for spin systems, including:
    * classes: SED
'''

__all__=['SED']

from ED import *
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
    target : QuantumNumber
        The target space of the SED.
    '''

    def __init__(self,lattice,config,qnses,terms=(),target=None,dtype=np.complex128,**karg):
        '''
        Constructor.

        Parameters
        ----------
        lattice : Lattice
            The lattice of the system.
        config : IDFConfig
            The configuration of the internal degrees of freedom on the lattice.
        qnses : QNSConfig
            The configuration of the quantum numbers.
        terms : list of Term, optional
            The terms of the system.
        target : QuantumNumber, optional
            The target space of the SED.
        dtype : np.float32, np.float64, np.complex64, np.complex128
            The data type of the matrix representation of the Hamiltonian.
        '''
        assert config.priority==qnses.priority
        self.lattice=lattice
        self.config=config
        self.qnses=qnses
        self.terms=terms
        self.target=target
        self.dtype=dtype
        self.generator=HP.Generator(bonds=lattice.bonds,config=config,table=config.table(),terms=terms,dtype=dtype)
        if self.status.map is None: self.status.update(OrderedDict((term.id,term.value) for term in terms))
        self.operators=self.generator.operators

    def set_matrix(self,refresh=True):
        '''
        Set the csr_matrix representation of the Hamiltonian.
        '''
        if refresh:
            table=self.generator.table
            if self.target is None or len(table)<=1:
                self.generator.set_matrix(HP.soptrep,table)
            else:
                cut,qnses=len(table)/2,[self.qnses[index] for index in sorted(table,key=table.get)]
                lqns,lpermutation=HP.QuantumNumbers.kron(qnses[:cut]).sort(history=True)
                rqns,rpermutation=HP.QuantumNumbers.kron(qnses[cut:]).sort(history=True)
                subslice=HP.QuantumNumbers.kron([lqns,rqns]).subslice(targets=(self.target,))
                rcs=(np.divide(subslice,len(rqns)),np.mod(subslice,len(rqns)),np.zeros(len(lqns)*len(rqns),dtype=np.int64))
                rcs[2][subslice]=xrange(len(subslice))
                self.generator.set_matrix(HP.soptrep,table,cut=cut,permutations=(lpermutation,rpermutation),rcs=rcs)
        self.matrix=self.generator.matrix
