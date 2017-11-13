'''
--------
App pack
--------

App pack, including:
    * classes: EB, POS, DOS, GF, FS, BC, BP, GP, CPFF
'''

__all__=['EB','POS','DOS','GF','FS','BC','BP','GP','CPFF']

import numpy as np
from ..EngineApp import App
from ..Utilities import berry_curvature,berry_phase

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

class POS(App):
    '''
    Profiles of states.

    Attributes
    ----------
    k : 1d ndarray of int
        The k point at which the profiles of states are wanted.
    ns : iterable of int
        The sequences of the states whose profiles are wanted.
    '''

    def __init__(self,k=(),ns=None,**karg):
        '''
        Constructor.

        Parameters
        ----------
        k : 1d ndarray of int, optional
            The k point at which the profiles of states are wanted.
        ns : iterable of int, optional
            The sequences of the states whose profiles are wanted.
        '''
        self.k=k
        self.ns=ns

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
    bcoff : logical, optional
        When True, only the Chern number will be included in the returned data.
        Otherwise, the Berry curvature will be included as well.
    '''

    def __init__(self,BZ,mu=0.0,d=10**-6,bcoff=True,**karg):
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
        bcoff : logical, optional
            When True, only the Chern number will be included in the returned data.
            Otherwise, the Berry curvature will be included as well.
        '''
        self.BZ=BZ
        self.mu=mu
        self.d=d
        self.bcoff=bcoff

    def set(self,H):
        '''
        Using the Kubo formula to calculate the Berry curvature of the occupied bands for a Hamiltonian with the given chemical potential.

        Parameters
        ----------
        H : callable
            Input function which returns the Hamiltonian as a 2D ndarray.

        Returns
        -------
        bc : 1d ndarray
            The values of the Berry curvature.
        cn : np.float64
            The integration of the Berry curvature.
            When BZ is the first Brillouin zone, this number is the first Chern number.
        '''
        bc=np.zeros(self.BZ.rank('k'))
        for i,ks in enumerate(self.BZ()):
            bc[i]=berry_curvature(H,ks['k'][0],ks['k'][1],self.mu,d=self.d)
        cn=np.sum(bc)*self.BZ.volume('k')/len(bc)/2/np.pi
        return bc,cn

class BP(App):
    '''
    Berry phase.

    Attributes
    ----------
    path : BaseSpace
        The path in the base space along which to calculate the Berry phase.
    ns : iterable of int
        The sequences of bands whose Berry phases are wanted.
    '''

    def __init__(self,path,ns=(0,),**karg):
        '''
        Constructor.

        Parameters
        ----------
        path : BaseSpace
            The path in the base space along which to calculate the Berry phase.
        ns : iterable of int, optional
            The sequences of bands whose Berry phases are wanted.
        '''
        self.path=path
        self.ns=ns

    def set(self,H,path):
        '''
        Set the Berry phases of the wanted bands for the input Hamiltonian.

        Parameters
        ----------
        H : callable
            Input function which returns the Hamiltonian as a 2D ndarray.
        path : list of dict
            The path of parameters passed to `H`.

        Returns
        -------
        1d ndarray of np.float64
            The Berry phases of the bands.
        '''
        return berry_phase(H,path,self.ns)

class GP(App):
    '''
    Grand potential.

    Attributes
    ----------
    BZ : BaseSpace
        The Brillouin zone.
    mu : np.float64
        The Fermi level.
    filling : np.float64
        The filling factor.
    '''

    def __init__(self,BZ=None,mu=0.0,filling=None,**karg):
        '''
        Constructor.

        Parameters
        ----------
        BZ : BaseSpace, optional
            The Brillouin zone.
        mu : np.float64, optional
            The Fermi level.
        filling : np.float64, optional
            The filling factor.
        '''
        self.BZ=BZ
        self.mu=mu
        self.filling=filling

class CPFF(App):
    '''
    Chemical potential or filling factor.

    Attributes
    ----------
    task : 'FF', 'CP'
        'FF' for filling factor and 'CP' for chemical potential.
    BZ : BaseSpace
        The Brillouin zone.
    cf : np.float64
        * When `task` is 'FF': the chemical potential
        * When `task` is 'CP': the filling factor
    '''

    def __init__(self,task='FF',BZ=None,cf=None,**karg):
        '''
        Constructor.

        Parameters
        ----------
        task : 'FF', 'CP', optional
            'FF' for filling factor and 'CP' for chemical potential.
        BZ : BaseSpace, optional
            The Brillouin zone.
        cf : np.float64, optional
            * When `task` is 'FF': the chemical potential
            * When `task` is 'CP': the filling factor
        '''
        assert task in ('FF','CP')
        self.task=task
        self.BZ=BZ
        self.cf=cf
