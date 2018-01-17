'''
===========
TBA and BdG
===========

Tight Binding Approximation for fermionic systems, including:
    * classes: TBA, GSE
    * functions: TBAEB, TBAGSE, TBADOS, TBABC
'''

__all__=['TBA','GSE','TBAGSE','TBAEB','TBADOS','TBABC']

from ..Basics import *
from numpy import *
from scipy.linalg import eigh
from collections import OrderedDict
import HamiltonianPy as HP
import matplotlib.pyplot as plt

class TBA(Engine):
    '''
    Tight-binding approximation for fermionic systems. Also support BdG systems (phenomenological superconducting systems at the mean-field level).

    Attributes
    ----------
    lattice : Lattice
        The lattice of the system.
    config : IDFConfig
        The configuration of the internal degrees of freedom.
    terms : list of Term
        The terms of the system.
    mask : ['nambu'] or []
        ['nambu'] for not using the nambu space and [] for using the nambu space.
    generator : Generator
        The operator generator for the Hamiltonian.


    Supported methods:
        ========    ==============================================
        METHODS     DESCRIPTION
        ========    ==============================================
        `TBAGSE`    calculate the ground state energy
        `TBAEB`     calculate the energy bands
        `TBADOS`    calculate the density of states
        `TBABC`     calculate the Berry curvature and Chern number
        ========    ==============================================
    '''

    def __init__(self,lattice=None,config=None,terms=None,mask=('nambu',),**karg):
        '''
        Constructor.

        Parameters
        ----------
        lattice : Lattice, optional
            The lattice of the system.
        config : IDFConfig, optional
            The configuration of the internal degrees of freedom.
        terms : list of Term, optional
            The terms of the system.
        mask : ['nambu'] or [], optional
            ['nambu'] for not using the nambu space and [] for using the nambu space.
        '''
        self.lattice=lattice
        self.config=config
        self.terms=terms
        self.mask=mask
        if self.map is None: self.parameters.update(OrderedDict((term.id,term.value) for term in terms))
        self.generator=Generator(bonds=lattice.bonds,config=config,table=config.table(mask=mask),terms=terms,half=True)
        self.logging()

    def update(self,**karg):
        '''
        This method update the engine.
        '''
        if len(karg)>0:
            super(TBA,self).update(**karg)
            self.generator.update(**self.data)

    @property
    def nmatrix(self):
        '''
        The dimension of the matrix representation of the Hamiltonian.
        '''
        return len(self.generator.table)

    def matrix(self,k=(),**karg):
        '''
        This method returns the matrix representation of the Hamiltonian.

        Parameters
        ----------
        k : 1D array-like, optional
            The coords of a point in K-space.
        karg : dict, optional
            Other parameters.

        Returns
        -------
        2d ndarray
            The matrix representation of the Hamiltonian.
        '''
        self.update(**karg)
        nmatrix=self.nmatrix
        result=zeros((nmatrix,nmatrix),dtype=complex128)
        for opt in self.generator.operators.values():
            phase=1 if len(k)==0 else exp(-1j*inner(k,opt.rcoord))
            result[opt.seqs]+=opt.value*phase
            if len(self.mask)==0:
                i,j=opt.seqs
                if i<nmatrix/2 and j<nmatrix/2: result[j+nmatrix/2,i+nmatrix/2]+=-opt.value*conjugate(phase)
        result+=conjugate(result.T)
        return result

    def matrices(self,basespace=None,mode='*'):
        '''
        This method returns a generator iterating over the matrix representations of the Hamiltonian defined on the input basespace.

        Parameters
        ----------
        basespace : BaseSpace, optional
            The base space on which the Hamiltonian is defined.
        mode : string, optional
            The mode to iterate over the base space.

        Yields
        ------
        2d ndarray
        '''
        if basespace is None:
            yield self.matrix()
        else:
            for paras in basespace(mode):
                yield self.matrix(**paras)

    def eigvals(self,basespace=None,mode='*'):
        '''
        This method returns all the eigenvalues of the Hamiltonian.

        Parameters
        ----------
        basespace : BaseSpace, optional
            The base space on which the Hamiltonian is defined.
        mode : string,optional
            The mode to iterate over the base space.

        Returns
        -------
        1d ndarray
            All the eigenvalues.
        '''
        if basespace is None:
            result=eigh(self.matrix(),eigvals_only=True)
        else:
            result=asarray([eigh(self.matrix(**paras),eigvals_only=True) for paras in basespace(mode)]).reshape(-1)
        return result

    def filling(self,mu,kspace=None):
        '''
        Return the filling factor of the system.

        Parameters
        ----------
        mu : float
            The chemical potential of the system.
        kspace : BaseSpace, optional
            The first Brillouin zone.

        Returns
        -------
        float
            The filling factor of the system.
        '''
        eigvals=sort(self.eigvals(kspace))
        return searchsorted(eigvals,mu)*1.0/len(eigvals)

    def mu(self,filling,kspace=None):
        '''
        Return the chemical potential of the system.

        Parameters
        ----------
        filling : float
            The filling factor of the system.
        kspace : BaseSpace, optional
            The first Brillouin zone.

        Returns
        -------
        float
            The chemical potential of the system.
        '''
        nelectron,eigvals=int(round(filling*(1 if kspace is None else kspace.rank('k'))*self.nmatrix)),sort(self.eigvals(kspace))
        return (eigvals[nelectron]+eigvals[nelectron-2])/2

    def gse(self,filling,kspace=None):
        '''
        Return the ground state energy of the system.

        Parameters
        ----------
        filling : float
            The filling factor of the system.
        kspace : BaseSpace, optional
            The first Brillouin zone.

        Returns
        -------
        float
            The ground state energy of the system.
        '''
        return sort(self.eigvals(kspace))[0:int(round(filling*(1 if kspace is None else kspace.rank('k'))*self.nmatrix))].sum()

class GSE(HP.App):
    '''
    The ground state energy.

    Attributes
    ----------
    filling : float
        The filling factor of the system.
    kspace : BaseSpace
        The first Brillouin zone.
    '''

    def __init__(self,filling,kspace=None,**karg):
        '''
        Constructor.

        Parameters
        ----------
        filling : float
            The filling factor of the system.
        kspace : BaseSpace, optional
            The first Brillouin zone.
        '''
        super(GSE,self).__init__(**karg)
        self.filling=filling
        self.kspace=kspace

def TBAGSE(engine,app):
    '''
    This method calculates the ground state energy.
    '''
    gse=engine.gse(filling=app.filling,kspace=app.kspace)
    engine.log<<engine<<'\n'
    engine.log<<Sheet.from_ordereddict({'Total':gse,'Site':gse/len(engine.lattice)/(1 if app.kspace is None else app.kspace.rank('k'))})<<'\n'
    if app.returndata: return gse

def TBAEB(engine,app):
    '''
    This method calculates the energy bands of the Hamiltonian.
    '''
    nmatrix=engine.nmatrix
    if app.path is not None:
        assert len(app.path.tags)==1
        result=zeros((app.path.rank(0),nmatrix+1))
        if app.path.mesh(0).ndim==1:
            result[:,0]=app.path.mesh(0)
        else:
            result[:,0]=array(xrange(app.path.rank(0)))
        for i,paras in enumerate(app.path()):
            result[i,1:]=eigh(engine.matrix(**paras),eigvals_only=True)
    else:
        result=zeros((2,nmatrix+1))
        result[:,0]=array(xrange(2))
        result[0,1:]=eigh(engine.matrix(),eigvals_only=True)
        result[1,1:]=result[0,1:]
    name='%s_%s'%(engine.tostr(mask=() if app.path is None else app.path.tags),app.name)
    if app.savedata: savetxt('%s/%s.dat'%(engine.dout,name),result)
    if app.plot: app.figure('L',result,'%s/%s'%(engine.dout,name))
    if app.returndata: return result

def TBADOS(engine,app):
    '''
    This method calculates the density of states of the Hamiltonian.
    '''
    result=zeros((app.ne,2))
    eigvals=engine.eigvals(app.BZ)
    emin=eigvals.min() if app.emin is None else app.emin
    emax=eigvals.max() if app.emax is None else app.emax
    for i,v in enumerate(linspace(emin,emax,num=app.ne)):
       result[i,0]=v
       result[i,1]=sum(app.eta/((v-eigvals)**2+app.eta**2))
    name='%s_%s'%(engine,app.name)
    if app.savedata: savetxt('%s/%s.dat'%(engine.dout,name),result)
    if app.plot: app.figure('L',result,'%s/%s'%(engine.dout,name))
    if app.returndata: return result

def TBABC(engine,app):
    '''
    This method calculates the total Berry curvature and Chern number of the filled bands of the Hamiltonian.
    '''
    bc,cn=app.set(lambda kx,ky: engine.matrix(k=[kx,ky]))
    engine.log<<'Chern number(mu): %s(%s)'%(cn,app.mu)<<'\n'
    if app.savedata or app.plot or app.returndata:
        result=zeros((app.BZ.rank('k'),3))
        result[:,0:2]=app.BZ.mesh('k')
        result[:,2]=bc
        name='%s_%s'%(engine,app.name)
        if app.savedata: savetxt('%s/%s.dat'%(engine.dout,name),result)
        if app.plot: app.figure('P',result.reshape((int(sqrt(result.shape[0])),int(sqrt(result.shape[0])),3)),'%s/%s'%(engine.dout,name),axis='equal')
        if app.returndata: return cn if app.bcoff else cn,result
