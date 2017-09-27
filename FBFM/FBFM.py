'''
======================
Flat band ferromagnets
======================

Spin excitations for flat band ferromagnets, including:
    * constants: FBFM_PRIORITY
    * classes: FBFMBasis, FBFM, EB
    * functions: optrep, FBFMEB
'''

__all__=['FBFM_PRIORITY','FBFMBasis','optrep','FBFM','EB','FBFMEB']

import numpy as np
import HamiltonianPy as HP
import HamiltonianPy.Misc as HM
import scipy.linalg as sl
import itertools as it
import matplotlib.pyplot as plt
from collections import OrderedDict

FBFM_PRIORITY=('spin','scope','nambu','site','orbital')

class FBFMBasis(object):
    '''
    The basis of the projected single particle space.

    Attributes
    ----------
    BZ : FBZ
        The first Brillouin zone.
    polarization : 'up'/'dw'
        The polarization of the ground state.
    E1,E2 : 2d ndarray
        The eigenvalues of the projected single particle space.
    U1,U2 : 3d ndarray
        The eigenvectors of the projected single particle space.
    '''

    def __init__(self,BZ=None,polarization='up'):
        '''
        Constructor.

        Parameters
        ----------
        BZ : FBZ, optional
            The first Brillouin zone.
        polarization : 'up'/'dw', optional
            The polarization of the ground state.
        '''
        assert polarization in ('up','dw')
        self.BZ=BZ
        self.polarization=polarization

    def set(self,matrix):
        '''
        Set the basis.

        Parameters
        ----------
        matrix : callable
            The function to get the single particle matrix.
        '''
        Eup,Uup,Edw,Udw=[],[],[],[]
        for k in [()] if self.BZ is None else self.BZ.mesh('k'):
            m=matrix(k)
            es,us=sl.eigh(m[:m.shape[0]/2,:m.shape[0]/2])
            Eup.append(es[:m.shape[0]/4])
            Uup.append(us[:,:m.shape[0]/4])
            es,us=sl.eigh(m[m.shape[0]/2:,m.shape[0]/2:])
            Edw.append(es[:m.shape[0]/4])
            Udw.append(us[:,:m.shape[0]/4])
        Eup,Uup=np.asarray(Eup),np.asarray(Uup).transpose((1,0,2))
        Edw,Udw=np.asarray(Edw),np.asarray(Udw).transpose((1,0,2))
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
        The operator whose matrix representation is wanted.
        ``None`` represents the whole single particle Hamiltonian.
    k : QuantumNumber
        The quantum-number-formed k point.
    basis : FBFMBasis
        The basis of the projected single particle space.

    Returns
    -------
    2d ndarray
        The matrix representation of the operator.
    '''
    nk,nsp=basis.nk,basis.nsp
    permutation=[0] if basis.BZ is None else np.argsort((basis.BZ-k).sort(history=True)[1])
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
            diag=operator.value*(1 if len(k)==0 else np.exp(-1j*np.inner(basis.BZ.kcoord(k),operator.rcoord)))*np.identity(nsp,dtype=basis.dtype)
            for i in xrange(nk):
                result[i,i,:,:]=np.kron(np.kron(basis.U1[seq2,i,:],basis.U1[seq1,i,:].conjugate()),diag)
        else:
            diagsum=(basis.U2[seq1,:,:].conjugate()*basis.U2[seq2,:,:]).sum()*np.identity(nsp,dtype=basis.dtype)
            for i in xrange(nk):
                result[i,i,:,:]=diagsum-np.kron(basis.U2[seq1,i,:].conjugate(),basis.U2[seq2,i,:])
            result*=operator.value*(1 if len(k)==0 else np.exp(-1j*np.inner(basis.BZ.kcoord(k),operator.rcoord)))
        return result.transpose((0,2,1,3)).reshape((nk*nsp**2,nk*nsp**2))
    elif isinstance(operator,HP.FHubbard):
        assert len(set(operator.seqs))==1
        assert operator.indices[1].replace(nambu=HP.CREATION)==operator.indices[0]
        assert operator.indices[3].replace(nambu=HP.CREATION)==operator.indices[2]
        seq=next(iter(operator.seqs))
        diagsum=(basis.U2[seq,:,:].conjugate()*basis.U2[seq,:,:]).sum()*np.identity(nsp,dtype=basis.dtype)
        result=np.zeros((nk,nk,nsp*nsp,nsp*nsp),dtype=basis.dtype)
        for i in xrange(nk):
            for j in xrange(nk):
                A=np.kron(basis.U1[seq,i,:],basis.U1[seq,j,:].conjugate())
                B=(diagsum if i==j else 0)-np.kron(basis.U2[seq,permutation[i],:].conjugate(),basis.U2[seq,permutation[j],:])
                result[i,j,:,:]=np.kron(A,B)*operator.value/nk
        return result.transpose((0,2,1,3)).reshape((nk*nsp**2,nk*nsp**2))
    else:
        raise ValueError('optrep error: not supported operator type(%s).'%operator.__class__.__name__)

class FBFM(HP.Engine):
    '''
    Attributes
    ----------
    basis : FBFMBasis
        The basis of the projected single particle space.
    lattice : Lattice
        The lattice of the system.
    config : IDFConfig
        The configuration of the internal degrees of freedom of the system.
    terms : list of Term
        The single particle terms of the system.
    interactions : list of Term
        The interaction terms of the system.
    dtype : np.float64 or np.complex128, etc
        The data type of the matrix representation of the system's Hamiltonian.
    generator : Generator
        The generator of the single particle part of the system.
    igenerator : Generator
        The generator of the interaction part of the system.

    Supported methods:
        ========     ==================================================
        METHODS      DESCRIPTION
        ========     ==================================================
        `FBFMEB`     calculate the energy spectrums of spin excitations
        ========     ==================================================
    '''

    def __init__(self,basis=None,lattice=None,config=None,terms=None,interactions=None,dtype=np.complex128,**karg):
        '''
        Constructor.

        Parameters
        ----------
        basis : FBFMBasis
            The basis of the projected single particle space.
        lattice : Lattice
            The lattice of the system.
        config : IDFConfig
            The configuration of the internal degrees of freedom of the system.
        terms : list of Term
            The single particle terms of the system.
        interactions : list of Term
            The interaction terms of the system.
        dtype : np.float64 or np.complex128, etc
            The data type of the matrix representation of the system's Hamiltonian.
        '''
        assert config.priority==FBFM_PRIORITY
        self.basis=basis
        self.lattice=lattice
        self.config=config
        self.terms=terms
        self.interactions=interactions
        self.dtype=dtype
        if self.status.map is None: self.status.update(OrderedDict((term.id,term.value) for term in it.chain(terms,interactions)))
        self.generator=HP.Generator(bonds=lattice.bonds,config=config,table=config.table(mask=['nambu']),terms=terms,dtype=dtype,half=True)
        self.igenerator=HP.Generator(bonds=lattice.bonds,config=config,table=config.table(mask=['spin','nambu']),terms=interactions,dtype=dtype,half=False,order='density')
        self.basis.set(self.spmatrix)

    def spmatrix(self,k=()):
        '''
        The single particle matrix.

        Parameters
        ----------
        k : iterable of float
            The k point.

        Returns
        -------
        2d ndarray
            The single particle matrix.
        '''
        result=np.zeros((len(self.generator.table),len(self.generator.table)),dtype=self.dtype)
        for opt in self.generator.operators.values():
            result[opt.seqs]+=opt.value*(1 if len(k)==0 else np.exp(-1j*np.inner(k,opt.icoord)))
        result+=result.T.conjugate()
        return result

    def update(self,**karg):
        '''
        Update the engine.
        '''
        self.status.update(karg)
        karg=self.status.parameters(karg)
        self.generator.update(**karg)
        self.igenerator.update(**karg)
        self.basis.set(self.spmatrix)

    @property
    def nmatrix(self):
        '''
        The dimension of the Hilbert space of spin-1 excitations.
        '''
        return self.basis.nk*self.basis.nsp**2

    def matrix(self,k=None,**karg):
        '''
        The matrix representation of the spin excitations.

        Parameters
        ----------
        k : QuantumNumber, optional

        Returns
        -------
        2d ndarray
            The matrix representation of the spin excitations.
        '''
        if len(karg)>0: self.update(**karg)
        result=0
        for operator in it.chain([None],self.igenerator.operators.itervalues()):
            result+=optrep(operator,k,self.basis)
        return result

    def view(self,path=None,show=True,suspend=False,close=True):
        '''
        View the projected single particle energy levels along a path in the k space.

        Parameters
        ----------
        path : str
            The str-formed path.
        show : logical, optional
            True for showing the view and False for not.
        suspend : logical, optional
            True for suspending the view and False for not.
        close : logical, optional
            True for closing the view and False for not.
        '''
        if path is None:
            assert self.basis.BZ is None
            ks=[0,1]
            e1=np.hstack(self.basis.E1,self.basis.E1)
            e2=np.hstack(self.basis.E2,self.basis.E2)
        else:
            ks=self.basis.BZ.path(HP.KMap(self.basis.BZ.reciprocals,path),mode='I')
            e1=self.basis.E1[ks,:]
            e2=self.basis.E2[ks,:]
        fig,ax=plt.subplots(nrows=1,ncols=2)
        plt.suptitle('%s'%self.status.tostr(mask=[term.id for term in self.interactions]))
        ax[0].plot(ks,e1)
        ax[1].plot(ks,e2)
        ax[0].set_title('Spin down' if self.basis.polarization=='up' else 'Spin up')
        ax[1].set_title('Spin up' if self.basis.polarization=='up' else 'Spin down')
        if show and suspend: plt.show()
        if show and not suspend: plt.pause(1)
        if close: plt.close()

class EB(HP.EB):
    '''
    Energy spectrums of spin excitations.

    Attributes
    ----------
    ne : integer
        The number of energy spectrums.
    method : 'eigh','eigsh'
        The function used to calculate the spectrums. 'eigh' for `sl.eigh` and 'eigsh' for `HM.eigsh`.
    '''

    def __init__(self,ne=6,method='eigh',**karg):
        '''
        Constructor.

        Parameters
        ----------
        ne : integer, optional
            The number of energy spectrums.
        method : 'eigh','eigsh'
            The function used to calculate the spectrums. 'eigh' for `sl.eigh` and 'eigsh' for `HM.eigsh`.
        '''
        super(EB,self).__init__(**karg)
        self.ne=ne
        self.method=method

def FBFMEB(engine,app):
    '''
    This method calculates the energy spectrums of the spin excitations.
    '''
    ne=min(app.ne or engine.nmatrix,engine.nmatrix)
    if app.path is not None:
        indices=engine.basis.BZ.path(HP.KMap(engine.lattice.reciprocals,app.path) if isinstance(app.path,str) else HP.app.path,mode='I')
        result=np.zeros((len(indices),ne+1))
        result[:,0]=np.array(xrange(len(indices)))
        engine.log<<'%s: '%len(indices)
        for i,index in enumerate(indices):
            engine.log<<'%s(%s)..'%(i,index)
            m=engine.matrix(engine.basis.BZ[index])
            result[i,1:]=sl.eigh(m,eigvals_only=True)[:ne] if app.method=='eigh' else HM.eigsh(m,k=ne,return_eigenvectors=False)
        engine.log<<'\n'
    else:
        result=np.zeros((2,ne+1))
        result[:,0]=np.array(xrange(2))
        result[0,1:]=sl.eigh(engine.matrix(),eigvals_only=True)[:ne] if app.method=='eigh' else HM.eigsh(engine.matrix(),k=ne,return_eigenvectors=False)
        result[1,1:]=result[0,1:]
    if app.save_data:
        np.savetxt('%s/%s_EB.dat'%(engine.dout,engine.status),result)
    if app.plot:
        plt.title('%s_EB'%engine.status)
        plt.plot(result[:,0],result[:,1:])
        if app.show and app.suspend: plt.show()
        if app.show and not app.suspend: plt.pause(app.SUSPEND_TIME)
        if app.save_fig: plt.savefig('%s/%s_EB.png'%(engine.dout,engine.status))
        plt.close()
