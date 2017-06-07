'''
==============================
Fermionic exat diagonalization
==============================

Exact diagonalization for fermionic systems, including:
    * classes: FED
    * functions: FGF
'''

__all__=['FED','FGF']

from ED import *
from scipy.sparse import csr_matrix
from copy import deepcopy
import HamiltonianPy as HP
import numpy as np

class FED(ED):
    '''
    Exact diagonalization for an electron system.

    Attributes
    ----------
    basis : FBasis
        The occupation number basis of the system.
    '''

    def __init__(self,basis,lattice,config,terms=[],dtype=np.complex128,**karg):
        '''
        Constructor.

        Parameters
        ----------
        basis : FBasis
            The occupation number basis of the system.
        lattice : Lattice
            The lattice of the system.
        config : IDFConfig
            The configuration of the internal degrees of freedom on the lattice.
        terms : list of Term, optional
            The terms of the system.
        dtype : np.float32, np.float64, np.complex64, np.complex128
            The data type of the matrix representation of the Hamiltonian.
        '''
        self.basis=basis
        self.lattice=lattice
        self.config=config
        self.terms=terms
        self.dtype=dtype
        self.generator=HP.Generator(bonds=lattice.bonds,config=config,table=config.table(mask=['nambu']),terms=terms,dtype=dtype)
        self.status.update(const=self.generator.parameters['const'])
        self.status.update(alter=self.generator.parameters['alter'])
        self.operators=self.generator.operators

    def set_matrix(self):
        '''
        Set the csr_matrix representation of the Hamiltonian.
        '''
        self.matrix=0
        for operator in self.operators.itervalues():
            self.matrix+=HP.foptrep(operator,self.basis,transpose=False)
        self.matrix+=self.matrix.T.conjugate()
        self.matrix=self.matrix.T

    def __replace_basis__(self,nambu,spin):
        '''
        Return a new ED instance with the basis replaced.

        Parameters
        ----------
        nambu : CREATION or ANNIHILATION
            CREATION for adding one electron and ANNIHILATION for subtracting one electron.
        spin : 0 or 1
            0 for spin down and 1 for spin up.

        Returns
        -------
        ED
            The new ED instance with the wanted new basis.
        '''
        if self.basis.mode=='FG':
            return self
        elif self.basis.mode=='FP':
            result=deepcopy(self)
            if nambu==HP.CREATION:
                result.basis=HP.FBasis((self.basis.nstate,self.basis.nparticle+1))
            else:
                result.basis=HP.FBasis((self.basis.nstate,self.basis.nparticle-1))
            result.matrix=csr_matrix((result.basis.nbasis,result.basis.nbasis),dtype=self.dtype)
            result.set_matrix()
            return result
        else:
            result=deepcopy(self)
            if nambu==HP.CREATION and spin==0:
                result.basis=HP.FBasis(up=(self.basis.nstate[0],self.basis.nparticle[0]),down=(self.basis.nstate[1],self.basis.nparticle[1]+1))
            elif nambu==HP.ANNIHILATION and spin==0:
                result.basis=HP.FBasis(up=(self.basis.nstate[0],self.basis.nparticle[0]),down=(self.basis.nstate[1],self.basis.nparticle[1]-1))
            elif nambu==HP.CREATION and spin==1:
                result.basis=HP.FBasis(up=(self.basis.nstate[0],self.basis.nparticle[0]+1),down=(self.basis.nstate[1],self.basis.nparticle[1]))
            else:
                result.basis=HP.FBasis(up=(self.basis.nstate[0],self.basis.nparticle[0]-1),down=(self.basis.nstate[1],self.basis.nparticle[1]))
            result.matrix=csr_matrix((result.basis.nbasis,result.basis.nbasis),dtype=self.dtype)
            result.set_matrix()
            return result

    def Hmats_Omats(self,operators):
        '''
        The matrix representations of the system's Hamiltonian and the input operators.

        Parameters
        ----------
        operators : list of Operator
            The input Operators.

        Returns
        -------
        Hmats : list of csr_matrix
            The matrix representations of the system's Hamiltonian.
        Omats : list of csr_matrix
            The matrix representations of the input operators.
        '''
        Hmats,Omats=[],[]
        for i,operator in enumerate(operators):
            assert operator.rank==1
            if i==0 or (self.basis.mode=='FS' and i==len(operators)/2): fed=self.__replace_basis__(nambu=operator.indices[0].nambu,spin=operator.indices[0].spin)
            Hmats.append(fed.matrix)
            Omats.append(HP.foptrep(operator,basis=[self.basis,fed.basis],transpose=True))
        return Hmats,Omats

def FGF(**karg):
    '''
    The zero-temperature single-particle Green's functions.
    '''
    return GF(filter=lambda engine,app,i,j: True if engine.basis.mode in ('FP','FG') or (i<app.nopt/2 and j<app.nopt/2) or (i>=app.nopt/2 and j>=app.nopt/2) else False,**karg)
