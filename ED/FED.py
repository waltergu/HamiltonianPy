'''
=================================================
Fermionic/Hard-core-bosonic exact diagonalization
=================================================

Exact diagonalization for fermionic/hard-core-bosonic systems, including:
    * classes: FED
    * functions: fedspgen, fedspcom, FGF
'''

__all__=['FED','fedspgen','fedspcom','FGF']

from ED import *
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
        sectors : iterable of FBasis
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
        self.sectors={sector.rep:sector for sector in sectors}
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
        sector : str
            The sector of the matrix representation of the Hamiltonian.
        reset : logical, optional
            True for resetting the matrix cache and False for not.

        Returns
        -------
        csr_matrix
            The matrix representation of the Hamiltonian.
        '''
        if reset: self.generator.setmatrix(sector,HP.foptrep,self.sectors[sector],transpose=False,dtype=self.dtype)
        self.sector=sector
        matrix=self.generator.matrix(sector)
        return matrix.T+matrix.conjugate()

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
        basis=self.sectors[self.sector]
        if basis.mode=='FG':
            return self
        elif basis.mode=='FP':
            result=deepcopy(self)
            new=basis.replace(nparticle=basis.nparticle+1) if nambu==HP.CREATION else basis.replace(nparticle=basis.nparticle-1)
            result.sectors={new.rep:new}
            result.sector=new.rep
            return result
        else:
            result=deepcopy(self)
            if nambu==HP.CREATION and spin==0:
                new=basis.replace(nparticle=basis.nparticle+1,spinz=basis.spinz-0.5)
            elif nambu==HP.ANNIHILATION and spin==0:
                new=basis.replace(nparticle=basis.nparticle-1,spinz=basis.spinz+0.5)
            elif nambu==HP.CREATION and spin==1:
                new=basis.replace(nparticle=basis.nparticle+1,spinz=basis.spinz+0.5)
            else:
                new=basis.replace(nparticle=basis.nparticle-1,spinz=basis.spinz-0.5)
            result.sectors={new.rep:new}
            result.sector=new.rep
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
    basis=fed.sectors[fed.sector]
    if method=='NB':
        yield 4 if basis.mode=='FS' else 2
    else:
        blocks=[{'inds':[],'opts':[]} for i in xrange(4 if basis.mode=='FS' else 2)]
        for i,operator in enumerate(operators):
            eindex=2 if basis.mode=='FS' and operator.indices[0].spin==1 else 0
            hindex=3 if basis.mode=='FS' and operator.indices[0].spin==1 else 1
            blocks[eindex]['inds'].append(i)
            blocks[hindex]['inds'].append(i)
            blocks[eindex]['opts'].append(operator.dagger)
            blocks[hindex]['opts'].append(operator)
        for i,block in enumerate(blocks):
            nfed=fed.__replace_basis__(nambu=HP.CREATION if i%2==0 else HP.ANNIHILATION,spin=0 if i<=1 else 1)
            yield BGF(
                    method=     method,
                    indices=    block['inds'],
                    sign=       (-1)**i,
                    matrix=     nfed.matrix(nfed.sector,reset=True),
                    operators=  [HP.foptrep(operator,basis=[basis,nfed.sectors[nfed.sector]],transpose=True,dtype=fed.dtype) for operator in block['opts']],
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
