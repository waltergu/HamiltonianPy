'''
==========================
Spin exact diagonalization
==========================

Exact diagonalization for spin systems, including:
    * classes: SED
'''

__all__=['SED']

from ED import *
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

    def __init__(self,lattice,config,qnses,terms=[],target=None,dtype=np.complex128,**karg):
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
        self.status.update(const=self.generator.parameters['const'])
        self.status.update(alter=self.generator.parameters['alter'])
        self.operators=self.generator.operators

    def set_matrix(self):
        '''
        Set the csr_matrix representation of the Hamiltonian.
        '''
        self.matrix,table=0,self.generator.table
        if self.target is None or len(table)<=1:
            for operator in self.operators.itervalues():
                self.matrix+=HP.soptrep(operator,table)
        else:
            cut,qnses=len(table)/2,[self.qnses[index] for index in sorted(table,key=table.get)]
            lqns,lpermutation=HP.QuantumNumbers.kron(qnses[:cut]).sort(history=True)
            rqns,rpermutation=HP.QuantumNumbers.kron(qnses[cut:]).sort(history=True)
            subslice=HP.QuantumNumbers.kron([lqns,rqns]).subslice(targets=(self.target,))
            rcs=(np.divide(subslice,len(rqns)),np.mod(subslice,len(rqns)),np.zeros(len(lqns)*len(rqns),dtype=np.int64))
            rcs[2][subslice]=xrange(len(subslice))
            for operator in self.operators.itervalues():
                self.matrix+=HP.soptrep(operator,table,cut=cut,permutations=(lpermutation,rpermutation),rcs=rcs)
