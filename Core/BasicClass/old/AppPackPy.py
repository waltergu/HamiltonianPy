'''
App pack.
'''
from EngineAppPy import *
from numpy import *
class EB(App):
    '''
    Energy bands.
    '''
    def __init__(self,path=None,emax=10.0,emin=-10.0,ne=400,eta=0.05,ns=6,ts=None,**karg):
        '''
        Constructor.
        Parameters:
            path: BaseSpace, optional
                The path in basespace along which the energy spectrum is to be computed.
            emax,emin: float, optional
                They define the range of the energy within which the EB is to be computed.
                Not supported by all engines.
            ne: integer, optional
                The number of sample points in the energy range defined by emin and emax.
                Not necessary for all engines.
            eta: float, optional
                The damping factor.
                Not necessary for all engines.
            ns: integer, optional
                The number of energy levels to be computed.
                Not supported for all engines.
            ts: BaseSpace, optional
                Only used for FLQT.
        '''
        self.path=path
        self.emax=emax
        self.emin=emin
        self.ne=ne
        self.eta=eta
        self.ns=ns
        self.ts=ts
        
class DOS(App):
    '''
    Density of states.
    '''
    def __init__(self,BZ=None,ne=100,eta=0.05,emin=-10.0,emax=10.0,**karg):
        '''
        Constructor.
        Parameters:
            BZ: BaseSpace,optional
                The first Brillouin zone.
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

class OP(App):
    '''
    Order parameter.
    '''
    def __init__(self,terms,BZ=None,p=1.0,**karg):
        '''
        Constructor.
        Parameters:
            term: list of Term
                The terms representing the order parameter.
            BZ: BaseSpace, optional
                The first Brillouin zone.
            p: float, optional
                A tunale parameter used in the calculation.
                For details, please refer arXiv:0806.2690.
        '''
        self.terms=terms
        self.BZ=BZ
        self.ms=0
        self.ops=0
        self.p=p

    def matrix(self,bonds,table,nambu):
        '''
        '''
        pass

class FF(App):
    '''
    Filling factor.
    '''
    def __init__(self,BZ=None,p=1.0,**karg):
        '''
        Constructor.
        Parameter:
            BZ: BaseSpace, optional
                The first Brillouin zone.
            p: float, optional
                A tunale parameter used in the calculation.
                For details, please refer arXiv:0806.2690.
        '''
        self.BZ=BZ
        self.p=p
        self.filling=0

class CP(App):
    '''
    Chemical potential.
    '''
    def __init__(self,BZ=None,p=1.0,error=10**-6,**karg):
        '''
        Constructor.
        Parameters:
            BZ: BaseSpace, optional
                The first Brillouin zone.
            p: float, optional
                A tunale parameter used in the calculation.
                For details, please refer arXiv:0806.2690.
            error: float, optional
                The error of the result.
        '''
        self.BZ=BZ
        self.p=p
        self.error=10**-6
        self.mu=0

class FS(App):
    '''
    Fermi surface.
    '''
    def __init__(self,BZ,eta=0.05,**karg):
        '''
        Constructor.
        Parameters:
            BZ: BaseSpace
                The first Brillouin zone.
            eta: float, optional
                The damping factor.
        '''
        self.BZ=BZ
        self.eta=eta

class GP(App):
    '''
    Grand potential.
    '''
    def __init__(self,BZ=None,**karg):
        '''
        Constructor.
        Parameters:
            BZ: BaseSpace, optional
                The first Brillouin zone.
        '''
        self.BZ=BZ
        self.gp=0

class GPS(App):
    '''
    Grand potential surface.
    '''
    def __init__(self,BS,**karg):
        '''
        Constructor.
        Parameters:
            BS: BaseSpace
                The basespace on which the grand potential is to be computed.
        '''
        self.BS=BS

class CN(App):
    '''
    Chern number.
    '''
    def __init__(self,BZ,d=10**-6,**karg):
        '''
        Constructor.
        Parameters:
            BZ: BaseSpace
                The first Brillouin zone.
            d: float, optional
                The difference used to calculate the derivatives.
        '''
        self.BZ=BZ
        self.d=d
        self.bc=None

    @property
    def cn(self):
        '''
        Returns the integration of the Berry curvature.
        '''
        return sum(self.bc)*self.BZ.volume['k']/self.BZ.rank['k']/2/pi

class GFC(App):
    '''
    The coefficients of Green's functions.
    '''
    def __init__(self,nstep=200,method='python',vtype='rd',error=0,**karg):
        '''
        Constructor.
        Parameters:
            nstep: integer, optional
                The max number of steps for the Lanczos iteration.
            method: string,optional
                It specifies the method the engine uses to compute the ground state.
                'python' means scipy.sparse.linalg.eigsh, and 'user' means Hamiltonian.Core.BasicAlgorithm.LanczosPy.Lanczos.eig.
            vtype: string,optional
                It specifies the initial vector type used for the calculation of the ground state.
                It only makes sense when method is 'user'.
                'rd' means random and 'sy' means symmetric.
            error: float, optional
                The error used to terminate the iteration.
        '''
        self.nstep=nstep
        self.method=method
        self.vtype=vtype
        self.error=error
        self.gse=0
        self.coeff=array([])

class GF(App):
    '''
    Green's functions.
    '''
    def __init__(self,shape,omega=0.0,k=None,**karg):
        '''
        Constructor.
        Parameters:
            shape: tuple
                The shape of the Green's function.
            omega: float or complex, optional
                The frequency.
            k: 1D ndarray like, optional
                The k points in the reciprocal space.
        '''
        self.omega=omega
        self.k=k
        self.gf=zeros(shape,dtype=complex128)
