'''
1D fermionic exact diagonalization with translation symmetry, including:
    * classes: TrFBasis, TrFED, EB
    * functions: trfoptrep, TrFEDEB
'''

__all__=['TrFBasis','trfoptrep','TrFED','EB','TrFEDEB']

from fbasis import *
from numba import jit
from math import sqrt
from scipy.sparse import csr_matrix
from collections import OrderedDict
import numpy as np
import HamiltonianPy as HP
import HamiltonianPy.ED as ED
import HamiltonianPy.Misc as HM
import matplotlib.pyplot as plt

class TrFBasis(HP.FBasis):
    '''
    1D translation invariant fermionic binary basis.

    Attributes
    ----------
    dk : int
    nk : int
    seqs : 1d ndarray of int
    maps : 1d ndarray of int
    translations : 1d ndarray of int
    signs : 1d ndarray of int
    masks : 1d ndarray of int
    '''

    def __init__(self,basis,dk,nk):
        '''
        Constructor.

        Parameters
        ----------
        basis : FBasis
        dk : int
        nk : int
        '''
        assert isinstance(basis,HP.FBasis) and basis.mode in ('FP','FS')
        self.dk=dk
        self.nk=nk
        self.mode='%sTR'%basis.mode
        self.nstate=basis.nstate
        self.nparticle=basis.nparticle
        self.spinz=basis.spinz
        self.table=basis.table
        seqs,maps,translations,signs,nbasis=trbasis(self.table,dk,nk,self.nstate)
        self.seqs=seqs[:nbasis]
        self.maps=maps
        self.translations=translations
        self.signs=signs
        self.masks=trmasks(self.seqs,self.translations,self.signs,nk)

    def tostr(self,protocol=0):
        '''
        Convert instance to string.
        '''
        assert protocol in (0,1)
        if protocol==0:
            return '\n'.join('{}({}): {:b}'.format(i,self.translations[index],self.table[index]) for i,index in enumerate(self.seqs))
        else:
            return '\n'.join('{:b}({},{}): {:b}'.format(basis,translation,sign,self.table[self.seqs[index]]) for basis,index,translation,sign in zip(self.table,self.maps,self.translations,self.signs))

    @property
    def nbasis(self):
        '''
        The number of basis.
        '''
        return len(self.seqs)

    def indices(self,k):
        '''
        The indices of the basis that is compatiable with the input k.

        Parameters
        ----------
        k : int

        Returns
        -------
        1d ndarray
        '''
        return np.concatenate(np.argwhere(self.masks[k,:]>=0))

def trfoptrep(operator,k,basis,dtype=np.complex128):
    '''
    The matrix representation of an operator on a translation invariant basis.

    Parameters
    ----------
    operator : FOperator
    k : int
    basis : TrFBasis
    dtype : np.float32, np.float64, np.complex64, or np.complex128, optional
    '''
    assert operator.rank%2==0 and isinstance(basis,TrFBasis) and 0<=k<basis.nk
    value,nambus,sequences=operator.value,(np.array([index.nambu for index in operator.indices])>0)[::-1],np.array(operator.seqs)[::-1]
    table,seqs,maps,translations,signs,masks,nk=basis.table,basis.seqs,basis.maps,basis.translations,basis.signs,basis.masks,basis.nk
    data,indices,indptr,dim=_trfoptrep_(k,value,nambus,sequences,table,seqs,maps,translations,signs,masks,nk,dtype)
    return csr_matrix((data,indices,indptr),shape=(dim,dim)).T

@jit
def _trfoptrep_(k,value,nambus,sequences,table,seqs,maps,translations,signs,masks,nk,dtype):
    ndata,data,indices,indptr=0,np.zeros(len(seqs),dtype=dtype),np.zeros(len(seqs),dtype=np.int32),np.zeros(len(seqs)+1,dtype=np.int32)
    dim,eye,temp=0,long(1),np.zeros(len(sequences)+1,dtype=np.int64)
    factor=np.exp(1.0j*2*np.pi*k/nk)
    for i in xrange(len(seqs)):
        if masks[k,i]>=0:
            indptr[dim]=ndata
            temp[0]=table[seqs[i]]
            for m in xrange(len(sequences)):
                if bool(temp[m]&eye<<sequences[m])==nambus[m]: break
                temp[m+1]=temp[m]|eye<<sequences[m] if nambus[m] else temp[m]&~(eye<<sequences[m])
            else:
                seq=HP.sequence(temp[-1],table)
                j=maps[seq]
                if masks[k,j]>=0:
                    nsign=0
                    for m in xrange(len(sequences)):
                        for n in xrange(sequences[m]):
                            if temp[m]&eye<<n: nsign+=1
                    indices[ndata]=masks[k,j]
                    sign=(1 if j==i else signs[seq])*(-1)**nsign
                    phase=1 if j==i else factor**translations[seq]
                    data[ndata]=sign*value*phase*sqrt(1.0*translations[seqs[i]]/translations[seqs[j]])
                    ndata+=1
            dim+=1
    indptr[dim]=ndata
    return data[:ndata],indices[:ndata],indptr[:dim+1],dim

class TrFED(ED.FED):
    '''
    Translation invariant exact diagonalization of 1d fermionic systems.
    '''

    def __init__(self,basis,lattice,config,terms=(),dtype=np.complex128,**karg):
        '''
        Constructor.

        Parameters
        ----------
        basis : TrFBasis
        lattice : Lattice
        config : IDFConfig
        terms : list of Term, optional
        dtype : np.float32, np.float64, np.complex64, np.complex128
        '''
        self.basis=basis
        self.lattice=lattice
        self.config=config
        self.terms=terms
        self.dtype=dtype
        self.generator=HP.Generator(bonds=lattice.bonds,config=config,table=config.table(mask=['nambu']),terms=terms,dtype=dtype,half=False)
        if self.map is None: self.parameters.update(OrderedDict((term.id,term.value) for term in terms))
        self.operators=self.generator.operators
        self.logging()

    def matrix(self,sector,reset=True):
        '''
        The matrix representation of the Hamiltonian.

        Parameters
        ----------
        sector : int
        reset : logical, optional

        Returns
        -------
        csr_matrix
        '''
        if reset: self.generator.setmatrix(sector,trfoptrep,k=sector,basis=self.basis,dtype=self.dtype)
        return self.generator.matrix(sector)

class EB(HP.EB):
    '''
    Energy bands.

    Attributes
    ----------
    ns : int
    kend : logical
    '''

    def __init__(self,ns=6,kend=True,**karg):
        '''
        Constructor.

        Parameters
        ----------
        ns : int, optional
        kend : logical, optional
        '''
        super(EB,self).__init__(**karg)
        self.ns=ns
        self.kend=kend

def TrFEDEB(engine,app):
    '''
    This function calculates the energy bands.
    '''
    result=np.zeros((engine.basis.nk+(1 if app.kend else 0),app.ns+1))
    for i in xrange(engine.basis.nk):
        result[i,0]=i
        result[i,1:]=engine.eigs(sector=i,k=app.ns,evon=False,resettimers=True if i==0 else False,showes=False)[1]
        engine.log<<'\n'
    if app.kend:
        result[-1,0]=engine.basis.nk
        result[-1,1:]=result[0,1:]
    name='%s_%s'%(engine,app.name)
    if app.savedata: np.savetxt('%s/%s.dat'%(engine.dout,name),result)
    if app.plot: app.figure('L',result,'%s/%s'%(engine.dout,name))
    if app.returndata: return result
