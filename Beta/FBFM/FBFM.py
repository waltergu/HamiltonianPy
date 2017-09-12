'''
Spin wave theory for spin reserved flat band ferromagnets.
    * constants: FBFM_PRIORITY
    * classes: FBFMBasis, FBFM
    * functions: optrep
'''

__all__=['FBFM_PRIORITY','FBFMBasis','optrep','FBFM']

import numpy as np
import HamiltonianPy as HP
import scipy.linalg as sl

FBFM_PRIORITY=('spin','scope','nambu','site','orbital')

class FBFMBasis(object):
    '''
    Attributes
    ----------
    BZ : FBZ
    polarization : 'up'/'dw'
    E1,E2 : 2d ndarray
    U1,U2 : 3d ndarray
    '''

    def __init__(self,BZ=None,polarization='up'):
        '''
        Constructor.

        Parameters
        ----------
        BZ : FBZ, optional
        polarization : 'up'/'dw', optional
        '''
        assert polarization in ('up','dw')
        self.BZ=BZ
        self.polarization=polarization

    def set(self,engine):
        '''
        Set the basis.

        Parameters
        ----------
        engine : Engine
        '''
        Eup,Uup,Edw,Udw=[],[],[],[]
        for k in [()] if self.BZ is None else self.BZ.mesh('k'):
            m=engine.spmatrix(k)
            es,us=sl.eigh(m[:m.shape[0]/2,:m.shape[0]/2],eigvals=(0,m.shape[0]/4))
            Eup.append(es)
            Uup.append(us)
            es,us=sl.eigh(m[m.shape[0]/2:,m.shape[0]/2:],eigvals=(0,m.shape[0]/4))
            Edw.append(es)
            Udw.append(us)
        Eup,Uup=np.asarray(Eup),np.asarray(Uup).transpose(axes=(1,0,2))
        Edw,Udw=np.asarray(Edw),np.asarray(Udw).transpose(axes=(1,0,2))
        if self.polarization=='up':
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
        '''
        Number of k points of the whole Brillouin zone.
        '''
        return 1 if self.BZ is None else len(self.BZ)

    @property
    def nsp(self):
        '''
        Number of single particle states at each k point.
        '''
        return self.E1.shape[-1]

    @property
    def dtype(self):
        '''
        The data type of the basis.
        '''
        return self.U1.dtype

def optrep(operator,k,basis):
    '''
    The matrix representation of an operator.

    Parameters
    ----------
    operator : FOperator or None
    k : QuantumNumber
    basis : FBFMBasis

    Returns
    -------
    2d ndarray
        The matrix representation of the operator.
    '''
    nk,nsp=basis.nk,basis.nsp
    permutation=[0] if basis.BZ is None else np.argsort((basis.BZ+k).sort(history=True)[1])
    if operator is None:
        result=np.zeros((nk,nsp,nsp,nk,nsp,nsp),dtype=basis.dtype)
        for i in xrange(nk):
            for j in xrange(nsp):
                for k in xrange(nsp):
                    result[i,j,k,i,j,k]=basis.E1[permutation[i],j]-basis.E2[i,k]
        return result.reshape((nk*nsp**2,nk*nsp**2))
    elif isinstance(operator,HP.FQuadratic):
        (index1,index2),(seq1,seq2)=operator.indices,operator.seqs
        assert seq1<=basis.nsp and seq2<=basis.nsp
        assert index1.spin==index2.spin and index1.nambu==HP.CREATION and index2.nambu==HP.ANNIHILATION
        result=np.zeros((nk,nk,nsp*nsp,nsp*nsp),dtype=basis.dtype)
        if index1.spin==(0 if basis.polarization=='dw' else 1):
            diag=operator.value*(1 if len(k)==0 else np.exp(-1j*np.inner(basis.kcoord(k),opt.rcoord)))*np.identity(nsp,dtype=basis.dtype)
            for i in xrange(nk):
                result[i,i,:,:]=np.kron(np.kron(basis.U1[seq2,i,:],basis.U1[seq1,i,:].conjugate()),diag)
        else:
            diagsum=(basis.U2[seq1,:,:].conjugate()*basis.U2[seq2,:,:]).sum()*np.identity(nsp,dtype=basis.dtype)
            for i in xrange(nk):
                result[i,i,:,:]=diagsum-np.kron(basis.U2[seq1,i,:].conjugate(),basis.U2[seq2,i,:])
            result*=operator.value*(1 if len(k)==0 else np.exp(-1j*np.inner(basis.kcoord(k),opt.rcoord)))
        return result.transpose(axes=(0,2,1,3)).reshape((nk*nsp**2,nk*nsp**2))
    elif isinstance(operator,HP.FHubbard):
        assert len(set(operator.seqs))==1
        assert operator.indices[1].replace(nambu=HP.CREATION)==operator.indices[0]
        assert operator.indices[3].replace(nambu=HP.CREATION)==operator.indices[2]
        seq=next(iter(operator.seqs))
        diagsum=(basis.U2[seq,:,:].conjugate()*basis.U2[seq,:,:]).sum()*np.identity(nsp,dtype=basis.dtype)
        result=np.zeros((nk,nk,nsp*nsp,nsp*nsp),dtype=basis.dtype)
        for i in xrange(nk):
            for j in xrange(nk):
                A=np.kron(basis.U1[seq,permutation[i]],basis.U1[seq,permutation[j]].conjugate())
                B=(diagsum if i==j else 0)-np.kron(basis.U2[seq,i,:].conjugate(),basis.U2[seq,j,:])
                result[i,j,:,:]=np.kron(A,B)*operator.value/nk
        return result.transpose(axes=(0,2,1,3)).reshape((nk*nsp**2,nk*nsp**2))
    else:
        raise ValueError('optrep error: not supported operator type(%s).'%operator.__class__.__name__)

class FBFM(HP.Engine):
    '''
    Attributes
    ----------
    basis : FBFMBasis
    lattice : Lattice
    config : IDFConfig
    terms : list of Term
    interactions : list of Term
    dtype : np.float64 or np.complex128, etc
    generator : Generator
    igenerator : Generator
    '''

    def __init__(self,basis=None,lattice=None,config=None,terms=None,interactions=None,dtype=np.complex128,**karg):
        '''
        Constructor.

        Parameters
        ----------
        basis : FBFMBasis
        lattice : Lattice
        config : IDFConfig
        terms : list of Term
        interactions : list of Term
        dtype : np.float64 or np.complex128, etc
        '''
        assert config.priority==FBFM_PRIORITY
        self.basis=basis
        self.lattice=lattice
        self.config=config
        self.terms=terms
        self.interactions=interactions
        self.dtype=dtype
        self.generator=HP.Generator(bonds=lattice.bonds,config=config,table=config.table(mask=['nambu']),terms=terms,dtype=dtype,half=True)
        self.igenerator=HP.Generator(bonds=lattice.bonds,config=config,table=config.table(mask=['spin','nambu']),terms=interactions,dtype=dtype,half=False,order='density')
        self.status.update(const=self.generator.parameters['const'],alter=self.generator.parameters['alter'])
        self.status.update(const=self.igenerator.parameters['const'],alter=self.igenerator.parameters['alter'])
        self.basis.set(self)

    def spmatrix(self,k=()):
        '''
        The single particle matrix.
        '''
        result=np.zeros((len(self.generator.table),len(self.generator.table)),dtype=self.dtype)
        for opt in self.generator.operators.values():
            result[opt.seqs]+=opt.value*(1 if len(k)==0 else np.exp(-1j*np.inner(k,opt.rcoord)))
        result+=result.T.conjugate()
        return result

    def update(self,**karg):
        '''
        Update the engine.
        '''
        self.generator.update(**karg)
        self.igenerator.update(**karg)
        self.status.update(alter=karg)
        self.basis.set(self)

    def matrix(self,k=None,**karg):
        '''
        The matrix representation of the spin-1 excitations.
        '''
        self.update(**karg)
        result=0
        for opreator in it.chain([None],self.igenerator.operators.itervalues()):
            result+=optrep(operator,k,self.basis)
        return result
