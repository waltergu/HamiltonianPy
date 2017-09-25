'''
=================================
Self-consistent mean field theory
=================================

Self-consistent mean field theory for fermionic systems, including:
    * classes: OP, SCMF
'''

__all__=['OP','SCMF']

from numpy import *
from ..Basics import RZERO,Generator,Timers,Info
from TBA import *
from copy import deepcopy
from collections import OrderedDict
from scipy.optimize import broyden2
from scipy.linalg import eigh

class OP(object):
    '''
    Order parameter.

    Attributes
    -----------
    value : number
        The value of the order parameter.
    matrix : 2d ndarray
        The matrix representation on the TBA basis of the order parameter.
    dtype : float64, complex128
        The data type of the value of the order parameter.
    '''

    def __init__(self,value,matrix,dtype=float64):
        '''
        Constructor.

        Parameters
        ----------
        value : number
            The value of the order parameter.
        matrix : 2d ndarray
            The matrix representation on the TBA basis of the order parameter.
        dtype : float64, complex128, optional
            The data type of the value of the order parameter.
        '''
        self.value=value
        self.matrix=matrix
        self.dtype=dtype

class SCMF(TBA):
    '''
    Self-consistent mean field theory for fermionic systems.

    Attributes
    ----------
    filling : float64
        The filling factor of the system.
    mu : float64
        The chemical potential of the system.
    temperature : float64
        The temperature of the system.
    orders : list of Term
        The terms representing the order parameters of the system.
    ops : OrderedDict in the form (key,value)
        * key: string
            The name of the term representing an order parameter of the system.
        * value: OP
            The corresponding order parameter of the system.
    timers : Timers
        The timer to record the consumed time of the iteration.
    '''

    def __init__(self,filling=0.5,temperature=0,lattice=None,config=None,terms=None,orders=None,mask=('nambu',),**karg):
        '''
        Constructor.

        Parameters
        ----------
        filling : float, optional
            The filling factor of the system.
        temperature : float64, optional
            The temperature of the system.
        lattice : Lattice, optional
            The lattice of the system.
        config : IDFConfig, optional
            The configuration of degrees of freedom.
        terms : list of Term, optional
            The terms of the system.
        orders : list of Term, optional
            The terms representing the order parameters of the system.
        mask : ['nambu'] or [], optional
            ['nambu'] for not using the nambu space and [] for using the nambu space.
        '''
        self.filling=filling
        self.temperature=temperature
        self.mu=None
        self.status.update(const={'filling':filling,'temperature':temperature})
        self.lattice=lattice
        self.config=config
        self.terms=terms
        self.orders=orders
        self.mask=mask
        self.generator=Generator(bonds=lattice.bonds,config=config,table=config.table(mask=mask),terms=terms+orders,half=True)
        self.status.update(**self.generator.parameters)
        self.ops=OrderedDict()
        for order in self.orders:
            m=zeros((self.nmatrix,self.nmatrix),dtype=complex128)
            term=deepcopy(order)
            term.value=1
            for opt in Generator(bonds=lattice.bonds,config=config,table=config.table(mask=mask),terms=[term],half=True).operators.values():
                m[opt.seqs]+=opt.value
            m+=conjugate(m.T)
            self.ops[order.id]=OP(order.value,m)
        self.timers=Timers('Iteration')

    def update_ops(self,kspace=None):
        '''
        Update the order parameters of the system.

        Parameters
        ----------
        kspace : BaseSpace, optional
            The Brillouin zone of the system.
        '''
        self.generator.update(**{name:order.value for name,order in self.ops.iteritems()})
        self.status.update(alter=self.generator.parameters['alter'])
        self.mu=super(SCMF,self).mu(self.filling,kspace)
        nmatrix=self.nmatrix
        f=(lambda e,mu: 1 if e<=mu else 0) if abs(self.temperature)<RZERO else (lambda e,mu: 1/(exp((e-mu)/self.temperature)+1))
        m=zeros((nmatrix,nmatrix),dtype=complex128)
        for matrix in self.matrices(kspace):
            eigs,eigvecs=eigh(matrix)
            for eig,eigvec in zip(eigs,eigvecs.T):
                m+=dot(eigvec.conj().reshape((nmatrix,1)),eigvec.reshape((1,nmatrix)))*f(eig,self.mu)
        nstate=(1 if kspace is None else kspace.rank('k'))*nmatrix/self.config.values()[0].nspin
        for key in self.ops.keys():
            self.ops[key].value=sum(m*self.ops[key].matrix)/nstate
            if self.ops[key].dtype in (float32,float64): self.ops[key].value=self.ops[key].value.real

    def iterate(self,kspace=None,tol=10**-6,maxiter=200):
        '''
        Iterate the SCMF to get converged order parameters.

        Parameters
        ----------
        kspace : BaseSpace, optional
            The Brillouin zone of the system.
        tol : float64, optional
            The tolerance of the order parameter.
        maxiter : integer, optional
            The maximum times of the iteration.
        '''
        def gx(values):
            for op,value in zip(self.ops.values(),values):
                op.value=value
            self.update_ops(kspace)
            return array([self.ops[key].value for key in self.ops.keys()])-values
        with self.timers.get('Iteration'):
            x0=array([self.ops[key].value for key in self.ops.keys()])
            ops=broyden2(gx,x0,verbose=True,reduction_method='svd',maxiter=maxiter,x_tol=tol)
        self.log<<'Order parameters:\n%s\n'%Info.from_ordereddict(OrderedDict([(name,op) for name,op in zip(self.ops.keys(),ops)]))
        self.log<<'Iterate: time consumed %ss.\n\n'%self.timers.time('Iteration')
