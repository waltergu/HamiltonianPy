'''
App pack, including:
1) classes: EB, DOS, GF, FS, BC, GP, CP, FF
'''

__all__=['EB','DOS','GF','FS','BC','GP','CP','FF']

from EngineApp import App
from numpy import *
from FermionicPackage import F_Linear
from ..Math import berry_curvature

class EB(App):
    '''
    Energy bands.
    Attributes:
        path: BaseSpace
            The path in basespace along which the energy spectrum is to be computed.
    '''

    def __init__(self,path=None,**karg):
        '''
        Constructor.
        Parameters:
            path: BaseSpace, optional
                The path in basespace along which the energy spectrum is to be computed.
        '''
        self.path=path

class DOS(App):
    '''
    Density of states.
    Attributes:
        BZ: BaseSpace
            The Brillouin zone.
        emin,emax: float
            The lower/upper bound of the energy range.
        ne: integer
            The number of sample points in the energy range.
        eta: float
            The damping factor.
    '''

    def __init__(self,BZ=None,ne=100,eta=0.05,emin=None,emax=None,**karg):
        '''
        Constructor.
        Parameters:
            BZ: BaseSpace,optional
                The Brillouin zone.
            emin,emax: float, optional
                They define the range of the energy within which the DOS is to be computed.
            ne: int, optional
                The number of sample points in the energy range defined by emin and emax.
            eta: float, optional
                The damping factor.
        '''
        self.BZ=BZ
        self.ne=ne
        self.eta=eta
        self.emin=emin
        self.emax=emax

class GF(App):
    '''
    Green's functions.
    Attribues:
        operators: list of Operator
            The operators of the GF.
        omega: number
            The frequency of the GF.
        k: 1D ndarray
            The momentum of the GF.
        gf: 2d ndarray
            The value of the GF.
    '''

    def __init__(self,operators=None,omega=None,k=None,**karg):
        '''
        Constructor.
        Parameters:
            operators: list of Operator, optional
                The operators of the GF.
            omega: number, optional
                The frequency of the GF.
            k: 1D array-like, optional
                The momentum of the GF.
            shape: tuple, optional
                The shape of the Green's function.
        '''
        self.operators=operators
        self.omega=omega
        self.k=k
        self.gf=None if operators is None else zeros((self.nopt,self.nopt),dtype=complex128)

    def reinitialization(self,operators):
        '''
        Reinitialize the GF.
        Parameters:
            operators: list of Operator.
        '''
        self.operators=operators
        self.gf=zeros((self.nopt,self.nopt),dtype=complex128)

    @property
    def nopt(self):
        '''
        The number of operators.
        '''
        return len(self.operators)

    @staticmethod
    def fsp_operators(table,lattice):
        '''
        Generate the fermionic single particle operators corresponding to a table.
        Parameters:
            table: Table
                The index-sequence table of the fermionic single particle operators.
            lattice: Lattice
                The lattice on which the fermionic single particle operators are defined.
        Returns: list of OperatorF
            The fermionic single particle operators corresponding to the table.
        '''
        result=[]
        for ndx in sorted(table,key=table.get):
            result.append(F_Linear(1,indices=[ndx],rcoords=[lattice[ndx.pid].rcoord],icoords=[lattice[ndx.pid].icoord],seqs=[table[ndx]]))
        return result

class FS(App):
    '''
    Fermi surface.
    Attribues:
        BZ: BaseSpace
            The Brillouin zone.
        eta: float64
            The damping factor.
    '''

    def __init__(self,BZ,eta=0.05,**karg):
        '''
        Constructor.
        Parameters:
            BZ: BaseSpace
                The Brillouin zone.
            eta: float, optional
                The damping factor.
        '''
        self.BZ=BZ
        self.eta=eta

class BC(App):
    '''
    Berry curvature.
    Attribues:
        BZ: BaseSpace
            The Brillouin zone.
        d: float64
            The difference used to calculate the partial directive.
        bc: 1d ndarray
            The values of the Berry curvature.
        cn: float64
            The integration of the Berry curvature.
            When BZ is the first Brillouin zone, this number is the first Chern number.
    '''

    def __init__(self,BZ,d=10**-6,**karg):
        '''
        Constructor.
        Parameters:
            BZ: BaseSpace
                The Brillouin zone.
            d: float, optional
                The difference used to calculate the derivates.
        '''
        self.BZ=BZ
        self.d=d
        self.bc=zeros(BZ.rank['k'])
        self.cn=None

    def set(self,H,mu):
        '''
        Using the Kubo formula to calculate the Berry curvature of the occupied bands for a Hamiltonian with the given chemical potential.
        Parameters:
            H: function
                Input function which returns the Hamiltonian as a 2D ndarray.
            mu: float64
                The chemical potential.
        '''
        for i,ks in enumerate(self.BZ()):
            self.bc[i]=berry_curvature(H,ks['k'][0],ks['k'][1],mu,d=self.d)
        self.cn=sum(self.bc)*self.BZ.volume['k']/len(self.bc)/2/pi

class GP(App):
    '''
    Grand potential.
    Attribues:
        BZ: BaseSpace
            The Brillouin zone.
        gp: float64
            The value of the grand potential.
    '''

    def __init__(self,BZ=None,**karg):
        '''
        Constructor.
        Parameters:
            BZ: BaseSpace, optional
                The Brillouin zone.
        '''
        self.BZ=BZ
        self.gp=0

class CP(App):
    '''
    Chemical potential.
    Attribues:
        BZ: BaseSpace
            The Brillouin zone.
        mu: float64
            The value of the chemical potential.
    '''

    def __init__(self,BZ=None,**karg):
        '''
        Constructor.
        Parameters:
            BZ: BaseSpace, optional
                The Brillouin zone.
        '''
        self.BZ=BZ
        self.mu=None

class FF(App):
    '''
    Filling factor.
    Attribues:
        BZ: BaseSpace
            The Brillouin zone.
        filling: float64
            The value of the filling factor.
    '''

    def __init__(self,BZ=None,**karg):
        '''
        Constructor.
        Parameter:
            BZ: BaseSpace, optional
                The first Brillouin zone.
        '''
        self.BZ=BZ
        self.filling=0
