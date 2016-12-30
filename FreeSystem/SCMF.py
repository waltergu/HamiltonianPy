'''
Self-consistent mean field theory for fermionic systems, including:
1) classes: OP, SCMF
'''

__all__=['OP','SCMF']

from numpy import *
from ..Basics import RZERO,Generator,Timers
from TBA import *
from copy import deepcopy
from collections import OrderedDict
from scipy.optimize import broyden2
from scipy.linalg import eigh
import time

class OP(object):
    '''
    Order parameter.
    Attribues:
        value: number
            The value of the order parameter.
        matrix: 2d ndarray
            The matrix representation on the TBA basis of the order parameter.
    '''

    def __init__(self,value,matrix):
        '''
        Constructor.
        Parameters:
            value: number
                The value of the order parameter.
            matrix: 2d ndarray
                The matrix representation on the TBA basis of the order parameter.
        '''
        self.value=value
        self.matrix=matrix

class SCMF(TBA):
    '''
    Self-consistent mean field theory for fermionic systems.
    Attribues:
        temperature: float64
            The temperature of the system.
        orders: list of Term
            The terms representing the order parameters of the system.
        ops: OrderedDict in the form (key,value)
            key: string
                The name of the term representing an order parameter of the system.
            value: OP
                The corresponding order parameter of the system.
    '''

    def __init__(self,filling=0,mu=0,temperature=0,lattice=None,config=None,terms=None,orders=None,mask=['nambu'],**karg):
        '''
        Constructor.
        Parameters:
            filling: float
                The filling factor of the system.
            mu: float
                The chemical potential of the system.
            temperature: float64
                The temperature of the system.
            lattice: Lattice
                The lattice of the system.
            config: Configuration
                The configuration of degrees of freedom.
            terms: list of Term
                The terms of the system.
            orders: list of Term
                The terms representing the order parameters of the system.
            mask: list of string
                A list to tell whether or not to use the nambu space.
        '''
        self.filling=filling
        self.mu=mu
        self.temperature=temperature
        self.lattice=lattice
        self.config=config
        self.terms=terms
        self.orders=orders
        self.mask=mask
        self.generators={}
        self.generators['h']=Generator(bonds=lattice.bonds,config=config,table=config.table(mask=mask),terms=terms+orders)
        self.status.update(const=self.generators['h'].parameters['const'])
        self.status.update(alter=self.generators['h'].parameters['alter'])
        self.ops=OrderedDict()
        for order in self.orders:
            v=order.value
            m=zeros((len(self.generators['h'].table),len(self.generators['h'].table)),dtype=complex128)
            buff=deepcopy(order)
            buff.value=1
            for opt in Generator(bonds=lattice.bonds,config=config,table=config.table(mask=mask),terms=[buff]).operators.values():
                m[opt.seqs]+=opt.value
            m+=conjugate(m.T)
            self.ops[order.id]=OP(v,m)
        self.log.timers['Iteration']=Timers(['Iteration'],str_form='s')

    def update_ops(self,kspace=None):
        '''
        Update the order parameters of the system.
        Parameters:
            kspace: BaseSpace, optional
                The Brillouin zone of the system.
        '''
        self.generators['h'].update(**{key:self.ops[key].value for key in self.ops.keys()})
        self.status.update(alter=self.generators['h'].parameters['alter'])
        self.set_mu(kspace)
        f=(lambda e,mu: 1 if e<=mu else 0) if abs(self.temperature)<RZERO else (lambda e,mu: 1/(exp((e-mu)/self.temperature)+1))
        nmatrix=len(self.generators['h'].table)
        buff=zeros((nmatrix,nmatrix),dtype=complex128)
        for matrix in self.matrices(kspace):
            eigs,eigvecs=eigh(matrix)
            for eig,eigvec in zip(eigs,eigvecs.T):
                buff+=dot(eigvec.conj().reshape((nmatrix,1)),eigvec.reshape((1,nmatrix)))*f(eig,self.mu)
        nstate=(1 if kspace is None else kspace.rank['k'])*nmatrix/self.config.values()[0].nspin
        for key in self.ops.keys():
            self.ops[key].value=sum(buff*self.ops[key].matrix)/nstate

    def iterate(self,kspace=None,tol=10**-6,maxiter=200):
        '''
        Iterate the SCMF to get converged order parameters.
        Parameters:
            kspace: BaseSpace, optional
                The Brillouin zone of the system.
            tol: float64, optional
                The tolerance of the order parameter.
            n: integer, optional
                The maximum times of the iteration.
        '''
        def gx(values):
            for op,value in zip(self.ops.values(),values):
                op.value=value
            self.update_ops(kspace)
            return array([self.ops[key].value for key in self.ops.keys()])-values
        self.log.timers['Iteration'].start('Iteration')
        x0=array([self.ops[key].value for key in self.ops.keys()])
        ops=broyden2(gx,x0,verbose=True,reduction_method='svd',maxiter=maxiter,x_tol=tol)
        self.log.timers['Iteration'].stop('Iteration')
        self.log<<'Order parameters: %s\n'%(ops,)
        self.log<<'Iterate: time consumed %ss.\n\n'%(self.log.timers['Iteration'].time('Iteration'))
