'''
--------
App pack
--------

App pack, including:
    * classes: EB, POS, DOS, GF, FS, BC, CN, BP, GP, CPFF
'''

__all__=['EB','POS','DOS','GF','FS','BC','CN','BP','GP','CPFF']

import numpy as np
import itertools as it
from scipy.linalg import eigh
from ..EngineApp import App
from ..Utilities import berrycurvature,berryphase

class EB(App):
    '''
    Energy bands.

    Attributes
    ----------
    path : BaseSpace
        The path in the basespace along which the energy spectrum is to be computed.
    mu : float
        The base point to measure the energy, usually the chemical potential of the system.
    '''

    def __init__(self,path=None,mu=0.0,**karg):
        '''
        Constructor.

        Parameters
        ----------
        path : BaseSpace, optional
            The path in the basespace along which the energy spectrum is to be computed.
        mu : float, optional
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
    emin,emax : float
        The lower/upper bound of the energy range.
    mu : float
        The base point to measure the energy, usually the chemical potential of the system.
    ne : int
        The number of sample points in the energy range.
    eta : float
        The damping factor.
    '''

    def __init__(self,BZ=None,emin=None,emax=None,mu=0.0,ne=100,eta=0.05,**karg):
        '''
        Constructor.

        Parameters
        ----------
        BZ : BaseSpace, optional
            The Brillouin zone.
        emin,emax : float, optional
            The lower/upper bound of the energy range.
        mu : float, optional
            The base point to measure the energy, usually the chemical potential of the system.
        ne : int, optional
            The number of sample points in the energy range defined by emin and emax.
        eta : float, optional
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
    mu : float
        The Fermi level.
    eta : float
        The damping factor.
    '''

    def __init__(self,BZ,mu=0.0,eta=0.05,**karg):
        '''
        Constructor.

        Parameters
        ----------
        BZ : BaseSpace
            The Brillouin zone.
        mu : float
            The Fermi level.
        eta : float, optional
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
    mu : float
        The Fermi level.
    d : float
        The step used to calculate the directives.
    bcoff : logical
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
        mu : float, optional
            The Fermi level.
        d : float, optional
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
        cn : float
            The integration of the Berry curvature.
            When BZ is the first Brillouin zone, this number is the first Chern number.
        '''
        bc=np.zeros(self.BZ.rank('k'))
        for i,ks in enumerate(self.BZ()):
            bc[i]=berrycurvature(H,ks['k'][0],ks['k'][1],self.mu,d=self.d)
        cn=np.sum(bc)*self.BZ.volume('k')/len(bc)/2/np.pi
        return bc,cn

class CN(App):
    '''
    Chern number.

    Attributes
    ----------
    BZ : FBZ
        The first Brillouin zone.
    ns : Tuple of int
        The energy bands whose Chern numbers are to be computed.
    '''

    def __init__(self,BZ,ns,**karg):
        '''
        Constructor.

        Parameters
        ----------
        BZ : FBZ
            The first Brillouin zone.
        ns : Tuple of int
            The energy bands whose Chern numbers are to be computed.
        '''
        self.BZ=BZ
        self.ns=ns

    def set(self,H):
        '''
        Using the plaquette integral method to calculate the Chern number of energy bands.

        Parameters
        ----------
        H : Callable
            Input function which returns the Hamiltonian as a 2D ndarray.

        Returns
        -------
        Tuple of float
            The calculated Chern numbers of the inquired energy bands.
        '''
        assert len(self.BZ.type.periods)==2
        smesh={}
        N1,N2=self.BZ.type.periods
        for i,j in it.product(range(N1),range(N2)):
            vs=eigh(H(i,j))[1]
            smesh[(i,j)]=vs[:,self.ns]
        phases=np.zeros(len(self.ns),dtype=np.float64)
        for i,j in it.product(range(N1),range(N2)):
            i1,j1=i,j
            i2,j2=(i+1)%N1,(j+1)%N2
            vs1,vs2,vs3,vs4=smesh[(i1,j1)],smesh[(i2,j1)],smesh[(i2,j2)],smesh[(i1,j2)]
            for k in range(len(self.ns)):
                phases[k]+=np.angle(np.vdot(vs1[:,k],vs2[:,k])*np.vdot(vs2[:,k],vs3[:,k])*np.vdot(vs3[:,k],vs4[:,k])*np.vdot(vs4[:,k],vs1[:,k]))
        return phases/np.pi/2

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

    def set(self,H):
        '''
        Set the Berry phases of the wanted bands for the input Hamiltonian.

        Parameters
        ----------
        H : callable
            Input function which returns the Hamiltonian as a 2D ndarray.

        Returns
        -------
        1d ndarray of float
            The Berry phases of the bands.
        '''
        return berryphase(H,list(self.path('+')),self.ns)

class GP(App):
    '''
    Grand potential.

    Attributes
    ----------
    BZ : BaseSpace
        The Brillouin zone.
    mu : float
        The Fermi level.
    '''

    def __init__(self,BZ=None,mu=0.0,**karg):
        '''
        Constructor.

        Parameters
        ----------
        BZ : BaseSpace, optional
            The Brillouin zone.
        mu : float, optional
            The Fermi level.
        '''
        self.BZ=BZ
        self.mu=mu

class CPFF(App):
    '''
    Chemical potential or filling factor.

    Attributes
    ----------
    task : 'FF', 'CP'
        'FF' for filling factor and 'CP' for chemical potential.
    BZ : BaseSpace
        The Brillouin zone.
    cf : float
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
        cf : float, optional
            * When `task` is 'FF': the chemical potential
            * When `task` is 'CP': the filling factor
        '''
        assert task in ('FF','CP')
        self.task=task
        self.BZ=BZ
        self.cf=cf
