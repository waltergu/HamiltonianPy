'''
Spin wave theory for spin reserved flat band ferromagnets.
    * constants: FBFM_PRIORITY
    * classes: 
    * functions: 
'''

__all__=[]

import numpy as np
import HamiltonianPy as HP
import scipy.linalg as sl

FBFM_PRIORITY=('spin','scope','nambu','site','orbital')

class FBFMBasis(object):
    '''
    Attributes
    ----------
    E1,E2 : 2d ndarray
    U1,U2 : 3d ndarray
    BZ : FBZ
    polarization : 'up' or 'dw'
    '''

    def __init__(self,Eup,Edw,Uup,Udw,BZ=None,polarization='up'):
        '''
        Constructor.

        Parameters
        ----------
        Eup,Edw : 1d/2d ndarray
        Uup,Udw : 2d/3d ndarray
        BZ : FBZ, optional
        polarization : 'up' or 'dw', optional
        '''
        Eup,Edw,Uup,Udw=np.asarray(Eup),np.asarray(Edw),np.asarray(Uup),np.asarray(Udw)
        assert polarization in ('up','dw') and Eup.shape==Edw.shape and Uup.shape==Udw.shape and Eup.shape[-1]==Uup.shape[-1] and Uup.dtype==Udw.dtype
        self.BZ=BZ
        self.polarization=polarization
        if self.BZ is None:
            Eup,Edw=Eup[np.newaxis,...],Edw[np.newaxis,...]
            Uup,Udw=Uup[np.newaxis,...],Udw[np.newaxis,...]
        else:
            assert Eup.shape[0]==len(BZ) and Uup.shape[0]==len(BZ)
        if polarization=='up':
            self.E1=Edw
            self.E2=Eup
            self.U1=Udw
            self.U2=Uup
        else:
            self.E1=Eup
            self.E2=Edw
            self.U1=Uup
            self.U2=Udw

    @property
    def nk(self):
        return 1 if self.BZ is None else len(self.BZ)

    @property
    def nsp(self):
        return self.E1.shape[-1]

    @property
    def dtype(self):
        return self.U1.dtype

def optrep(operator,k,basis):
    '''
    The matrix representation of an operator.

    Parameters
    ----------
    operator : FOperator
    k : QuantumNumber
    basis : FBFMBasis

    Returns
    -------
    2d ndarray
        The matrix representation of the operator.
    '''
    if isinstance(operator,HP.FHubbard):
        return optrep_hubbard(operator,k,basis)
    else:
        raise ValueError('optrep error: not supported operator type(%s).'%operator.__class__.__name__)

def optrep_hubbard(operator,k,basis):
    '''

    Parameters
    ----------
    operator : FHubbard
    k : QuantumNumber
    basis : FBFMBasis

    Returns
    -------
    2d ndarray
        The matrix representation of the Hubbard operator.
    '''
    assert len(set(operator.seqs))==1 and operator.indices[1].replace(nambu=HP.CREATION)==operator.indices[0] and operator.indices[3].replace(nambu=HP.CREATION)==operator.indices[2]
    seq,nk,nsp=next(iter(operator.seqs)),basis.nk,basis.nsp
    #A1,A2=U1[:,seq,:],U1[:,seq,:].conjugate()
    B=np.zeros((nk,nk,nsp,nsp),dtype=basis.dtype)
    diagsum=(basis.U2[:,seq,:].conjugate()*basis.U2[:,seq,:]).sum()
    for i,q in enumerate(basis):
        pass

class FBFM(HP.Engine):
    '''
    Attributes
    ----------
    lattice : Lattice
    config : IDFConfig
    terms : list of Term
    interactions : list of Term
    bz : FBZ
    generator : Generator
    igenerator : Generator
    '''

    def __init__(self,lattice=None,config=None,terms=None,interactions=None,bz=None,**karg):
        assert config.priority==FBFM_PRIORITY
        self.lattice=lattice
        self.config=config
        self.terms=terms
        self.interactions=interactions
        self.bz=bz
        self.generator=HP.Generator(bonds=lattice.bonds,config=config,table=config.table(mask=['nambu']),terms=terms,half=True)
        self.igenerator=HP.Generator(bonds=lattice.bonds,config=config,table=config.table(mask=['spin','nambu']),terms=interactions,half=False,order='density')
        self.status.update(const=self.generator.parameters['const'],alter=self.generator.parameters['alter'])
        self.status.update(const=self.igenerator.parameters['const'],alter=self.igenerator.parameters['alter'])
        self.spdiagonalize()

    @property
    def nsp(self):
        return len(self.generator.table)

    @property
    def nk(self):
        return 1 if self.bz is None else len(self.bz.mesh('k'))

    def spdiagonalize(self):
        def matrix(k=()):
            result=np.zeros((self.nsp,self.nsp),dtype=np.complex128)
            for opt in self.generator.operators.values():
                result[opt.seqs]+=opt.value*(1 if len(k)==0 else np.exp(-1j*np.inner(k,opt.rcoords[0])))
            result+=result.T.conjugate()
            return result
        dwesmesh,dwvsmesh=[],[]
        upesmesh,upvsmesh=[],[]
        for k in self.bz.mesh('k'):
            m=matrix(k)
            es,vs=sl.eigh(m[:self.nsp/2,:self.nsp/2],eigvals=(0,self.nsp/4))
            dwesmesh.append(es)
            dwvsmesh.append(vs)
            es,vs=sl.eigh(m[self.nsp/2:,self.nsp/2:],eigvals=(0,self.nsp/4))
            upesmesh.append(es)
            upvsmesh.append(vs)
        self.esmesh=np.array([dwesmesh,upesmesh])
        self.vsmesh=np.array([dwvsmesh,upvsmesh])

    def update(self,**karg):
        self.generator.update(**karg)
        self.igenerator.update(**karg)
        self.status.update(alter=karg)
        self.spdiagonalize()

    def matrix(self,k=None,**karg):
        self.update(**karg)
        result=np.zeros((self.nk*self.nsp**2/16,self.nk*self.nsp**2/16),dtype=np.complex128)
        pass
