'''
--------
App pack
--------

App pack, including:
    * classes: GSE, EB, DOS, GF, FS, BC, GP, CPFF
'''

__all__=['GSE','EB','DOS','GF','FS','BC','GP','CPFF']

import numpy as np
from ..EngineApp import App
from ..Utilities import berry_curvature

class GSE(App):
    '''
    Ground state energy.

    Attributes
    ----------
    gse : np.float64
        The groundstate energy.
    factor : integer
        An extra factor.
    '''

    def __init__(self,factor=1,**karg):
        '''
        Constructor.

        Parameters
        ----------
        factor : integer
        '''
        self.factor=factor
        self.gse=None

class EB(App):
    '''
    Energy bands.

    Attributes
    ----------
    path : BaseSpace
        The path in the basespace along which the energy spectrum is to be computed.
    mu : np.float64
        The base point to measure the energy, usually the chemical potential of the system.
    '''

    def __init__(self,path=None,mu=0.0,**karg):
        '''
        Constructor.

        Parameters
        ----------
        path : BaseSpace, optional
            The path in the basespace along which the energy spectrum is to be computed.
        mu : np.float64, optional
            The base point to measure the energy, usually the chemical potential of the system.
        '''
        self.path=path
        self.mu=mu

class DOS(App):
    '''
    Density of states.

    Attributes
    ----------
    BZ: BaseSpace
        The Brillouin zone.
    emin,emax : np.float64
        The lower/upper bound of the energy range.
    mu : np.float64
        The base point to measure the energy, usually the chemical potential of the system.
    ne : integer
        The number of sample points in the energy range.
    eta : np.float64
        The damping factor.
    '''

    def __init__(self,BZ=None,emin=None,emax=None,mu=0.0,ne=100,eta=0.05,**karg):
        '''
        Constructor.

        Parameters
        ----------
        BZ : BaseSpace, optional
            The Brillouin zone.
        emin,emax : np.float64, optional
            The lower/upper bound of the energy range.
        mu : np.float64, optional
            The base point to measure the energy, usually the chemical potential of the system.
        ne : integer, optional
            The number of sample points in the energy range defined by emin and emax.
        eta : np.float64, optional
            The damping factor.
        '''
        self.BZ=BZ
        self.emin=emin
        self.emax=emax
        self.mu=mu
        self.ne=ne
        self.eta=eta

class GF(App):
    '''
    Green's functions.

    Attributes
    ----------
    operators : list of Operator
        The operators of the GF.
    omega : number
        The frequency of the GF.
    k : 1d ndarray
        The momentum of the GF.
    dtype : np.complex64 or np.complex128
        The data type of the Green's functions.
    gf : 2d ndarray
        The value of the GF.
    '''

    def __init__(self,operators,omega=None,k=None,dtype=np.complex128,**karg):
        '''
        Constructor.

        Parameters
        ----------
        operators : list of Operator
            The operators of the GF.
        omega : number, optional
            The frequency of the GF.
        k : 1d ndarray, optional
            The momentum of the GF.
        dtype : np.complex64 or np.complex128
            The data type of the Green's functions.
        '''
        self.operators=operators
        self.omega=omega
        self.k=k
        self.dtype=dtype
        self.gf=np.zeros((self.nopt,self.nopt),dtype=dtype)

    @property
    def nopt(self):
        '''
        The number of operators.
        '''
        return len(self.operators)

class FS(App):
    '''
    Fermi surface.

    Attributes
    ----------
    BZ : BaseSpace
        The Brillouin zone.
    mu : np.float64
        The Fermi level.
    eta : np.float64
        The damping factor.
    '''

    def __init__(self,BZ,mu=0.0,eta=0.05,**karg):
        '''
        Constructor.

        Parameters
        ----------
        BZ : BaseSpace
            The Brillouin zone.
        mu : np.float64
            The Fermi level.
        eta : np.float64, optional
            The damping factor.
        '''
        self.BZ=BZ
        self.mu=mu
        self.eta=eta

class BC(App):
    '''
    Berry curvature.

    Attributes
    ----------
    BZ : BaseSpace
        The Brillouin zone.
    mu : np.float64
        The Fermi level.
    d : np.float64
        The step used to calculate the directives.
    bc : 1d ndarray
        The values of the Berry curvature.
    cn : np.float64
        The integration of the Berry curvature.
         When BZ is the first Brillouin zone, this number is the first Chern number.
    '''

    def __init__(self,BZ,mu=0.0,d=10**-6,**karg):
        '''
        Constructor.

        Parameters
        ----------
        BZ : BaseSpace
            The Brillouin zone.
        mu : np.float64, optional
            The Fermi level.
        d : np.float64, optional
            The step used to calculate the derivatives.
        '''
        self.BZ=BZ
        self.mu=mu
        self.d=d
        self.bc=np.zeros(BZ.rank('k'))
        self.cn=None

    def set(self,H):
        '''
        Using the Kubo formula to calculate the Berry curvature of the occupied bands for a Hamiltonian with the given chemical potential.

        Parameters
        ----------
        H: function
            Input function which returns the Hamiltonian as a 2D ndarray.
        '''
        for i,ks in enumerate(self.BZ()):
            self.bc[i]=berry_curvature(H,ks['k'][0],ks['k'][1],self.mu,d=self.d)
        self.cn=sum(self.bc)*self.BZ.volume('k')/len(self.bc)/2/np.pi

class GP(App):
    '''
    Grand potential.

    Attributes
    ----------
    BZ : BaseSpace
        The Brillouin zone.
    mu : np.float64
        The Fermi level.
    gp : float64
        The value of the grand potential.
    '''

    def __init__(self,BZ=None,mu=0.0,**karg):
        '''
        Constructor.

        Parameters
        ----------
        BZ : BaseSpace, optional
            The Brillouin zone.
        mu : np.float64, optional
            The Fermi level.
        '''
        self.BZ=BZ
        self.mu=mu
        self.gp=0

class CPFF(App):
    '''
    Chemical potential or filling factor.

    Attributes
    ----------
    task : 'FF', 'CP', optional
        'FF' for filling factor and 'CP' for chemical potential.
    BZ : BaseSpace
        The Brillouin zone.
    filling : np.float64
        The value of the filling factor.
    mu : np.float64
        The value of the chemical potential.
    '''

    def __init__(self,task='FF',BZ=None,filling=0.0,mu=0.0,**karg):
        '''
        Constructor.

        Parameters
        ----------
        task : 'FF', 'CP', optional
            'FF' for filling factor and 'CP' for chemical potential.
        BZ : BaseSpace, optional
            The Brillouin zone.
        filling : np.float64, optional
            The value of the filling factor.
        mu : np.float64, optional
            The value of the chemical potential.
        '''
        assert task in ('FF','CP')
        self.task=task
        self.BZ=BZ
        self.filling=filling
        self.mu=mu
