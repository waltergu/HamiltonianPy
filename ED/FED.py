'''
=================================================
Fermionic/Hard-core-bosonic exact diagonalization
=================================================

Exact diagonalization for fermionic/hard-core-bosonic systems, including:
    * classes: FED
    * functions: fedspgen, fedspcom, FGF
'''

__all__=['FED','fedspgen','fedspcom','FGF']

from .ED import *
from copy import deepcopy
from collections import OrderedDict
import HamiltonianPy as HP
import HamiltonianPy.Misc as HM
import HamiltonianPy.FreeSystem as TBA
import numpy as np

class FED(ED):
    '''
    Exact diagonalization for a fermionic/hard-core-bosonic system.
    '''

    def __init__(self,sectors,lattice,config,terms=(),dtype=np.complex128,**karg):
        '''
        Constructor.

        Parameters
        ----------
        sectors : set of FBasis
            The occupation number bases of the system.
        lattice : Lattice
            The lattice of the system.
        config : IDFConfig
            The configuration of the internal degrees of freedom on the lattice.
        terms : list of Term, optional
            The terms of the system.
        dtype : np.float32, np.float64, np.complex64, np.complex128
            The data type of the matrix representation of the Hamiltonian.
        '''
        if len(terms)>0: assert len({term.statistics for term in terms})==1
        self.sectors=set(sectors)
        self.lattice=lattice
        self.config=config
        self.terms=terms
        self.dtype=dtype
        self.sector=None
        self.generator=HP.Generator(bonds=lattice.bonds,config=config,table=config.table(mask=['nambu']),terms=terms,boundary=self.boundary,dtype=dtype,half=True)
        if self.map is None: self.parameters.update(OrderedDict((term.id,term.value) for term in terms))
        self.operators=self.generator.operators
        self.logging()

    @property
    def statistics(self):
        '''
        The statistics of the system, 'f' for fermionic and 'b' for bosonic.
        '''
        return next(iter(self.terms)).statistics

    def matrix(self,sector,reset=True):
        '''
        The matrix representation of the Hamiltonian.

        Parameters
        ----------
        sector : FBasis
            The sector of the matrix representation of the Hamiltonian.
        reset : logical, optional
            True for resetting the matrix cache and False for not.

        Returns
        -------
        csr_matrix
            The matrix representation of the Hamiltonian.
        '''
        self.sector=sector
        if reset: self.generator.setmatrix(sector,HP.foptrep,sector,transpose=False,dtype=self.dtype)
        matrix=self.generator.matrix(sector)
        return matrix.T+matrix.conjugate()

    def newbasis(self,nambu,spin):
        '''
        Return a new basis with one more or one less particle.

        Parameters
        ----------
        nambu : CREATION or ANNIHILATION
            CREATION for adding one particle and ANNIHILATION for subtracting one particle.
        spin : 0 or 1
            0 for spin down and 1 for spin up.

        Returns
        -------
        FBasis
            The new basis.
        '''
        basis=self.sector
        if basis.mode=='FG':
            result=basis
        elif basis.mode=='FP':
            result=basis.replace(nparticle=basis.nparticle+1) if nambu==HP.CREATION else basis.replace(nparticle=basis.nparticle-1)
        else:
            if nambu==HP.CREATION and spin==0:
                result=basis.replace(nparticle=basis.nparticle+1,spinz=basis.spinz-0.5)
            elif nambu==HP.ANNIHILATION and spin==0:
                result=basis.replace(nparticle=basis.nparticle-1,spinz=basis.spinz+0.5)
            elif nambu==HP.CREATION and spin==1:
                result=basis.replace(nparticle=basis.nparticle+1,spinz=basis.spinz+0.5)
            else:
                result=basis.replace(nparticle=basis.nparticle-1,spinz=basis.spinz-0.5)
        return result

    def totba(self):
        '''
        Convert the free part of the system to tba.
        '''
        return TBA.TBA(
            dlog=       self.log.dir,
            din=        self.din,
            dout=       self.dout,
            name=       self.name,
            parameters= self.parameters,
            map=        self.map,
            lattice=    self.lattice,
            config=     self.config,
            terms=      [term for term in self.terms if isinstance(term,HP.Quadratic)],
            dtype=      self.dtype
            )

def fedspgen(fed,operators,method='S'):
    '''
    This function generates the blocks of the zero-temperature single-particle Green's function of a fermionic/hard-core-bosonic system.

    Parameters
    ----------
    fed : FED
        The fermionic/hard-core-bosonic system.
    operators : list of Operator
        The input Operators.
    method : 'S', 'B', or 'NB'
        * 'S': simple Lanczos method;
        * 'B': block Lanczos method;
        * 'NB': number of blocks.

    Yields
    ------
    When `method` is 'S' or 'B': BGF
        The blocks of the zero-temperature single-particle Green's function.
    When `method` is 'NB': int
        The number of blocks.
    '''
    oldbasis=fed.sector
    if method=='NB':
        yield 4 if oldbasis.mode=='FS' else 2
    else:
        blocks=[{'inds':[],'opts':[]} for i in range(4 if oldbasis.mode=='FS' else 2)]
        for i,operator in enumerate(operators):
            eindex=2 if oldbasis.mode=='FS' and operator.indices[0].spin==1 else 0
            hindex=3 if oldbasis.mode=='FS' and operator.indices[0].spin==1 else 1
            blocks[eindex]['inds'].append(i)
            blocks[hindex]['inds'].append(i)
            blocks[eindex]['opts'].append(operator.dagger)
            blocks[hindex]['opts'].append(operator)
        for i,block in enumerate(blocks):
            newbasis=fed.newbasis(nambu=HP.CREATION if i%2==0 else HP.ANNIHILATION,spin=0 if i<=1 else 1)
            fed.addsector(newbasis,setcurrent=True)
            matrix=fed.matrix(fed.sector,reset=True)
            fed.removesector(fed.sector,newcurrent=oldbasis)
            yield BGF(
                    method=     method,
                    indices=    block['inds'],
                    sign=       (-1)**i,
                    matrix=     matrix,
                    operators=  [HP.foptrep(operator,basis=[oldbasis,newbasis],transpose=True,dtype=fed.dtype) for operator in block['opts']],
                    )

def fedspcom(blocks,omega):
    '''
    This function composes the zero-temperature single-particle Green's function of a fermionic/hard-core-bosonic system from its blocks.

    Parameters
    ----------
    blocks : list of BGF
        The blocks of the Green's function.
    omega : number
        The frequency.

    Returns
    -------
    2d ndarray
        The composed Green's function.
    '''
    assert len(blocks) in (2,4)
    if len(blocks)==2:
        return blocks[0].gf(omega).T+blocks[1].gf(omega)
    else:
        gfdw,indsdw=blocks[0].gf(omega).T+blocks[1].gf(omega),blocks[0].indices
        gfup,indsup=blocks[2].gf(omega).T+blocks[3].gf(omega),blocks[2].indices
        return HM.reorder(HM.blockdiag(gfdw,gfup),axes=[0,1],permutation=np.argsort(np.concatenate((indsdw,indsup))))

def FGF(**karg):
    '''
    The zero-temperature single-particle Green's functions.
    '''
    return GF(generate=fedspgen,compose=fedspcom,**karg)
