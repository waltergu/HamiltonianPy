'''
Self-consistent mean field theory for fermionic systems, including:
1) classes: OP, SCMF
'''

__all__=['op','SCMF']

from numpy import *
from ..Basics import *
from TBA import *
from copy import deepcopy
from collections import OrderedDict
from scipy.optimize import broyden1,broyden2
from scipy.linalg import eigh
import time

class op:
    '''
    '''
    def __init__(self,value,matrix):
        self.value=value
        self.matrix=matrix

class SCMF(TBA):
    '''
    '''
    def __init__(self,filling=0,mu=0,temperature=0,lattice=None,config=None,terms=None,orders=None,mask=['nambu'],**karg):
        '''
        Constructor.
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
        self.name.update(const=self.generators['h'].parameters['const'])
        self.name.update(alter=self.generators['h'].parameters['alter'])
        self.ops=OrderedDict()
        self.init_ops()

    def init_ops(self):
        nmatrix=len(self.generators['h'].table)
        for order in self.orders:
            v=order.value
            m=zeros((nmatrix,nmatrix),dtype=complex128)
            buff=deepcopy(order);buff.value=1
            for opt in Generator(bonds=self.lattice.bonds,config=self.config,table=self.config.table(mask=self.mask),terms=[buff]).operators.values():
                m[opt.seqs]+=opt.value
            m+=conjugate(m.T)
            self.ops[order.id]=op(v,m)

    def update_ops(self,kspace=None):
        self.generators['h'].update(**{key:self.ops[key].value for key in self.ops.keys()})
        self.name.update(alter=self.generators['h'].parameters['alter'])
        self.set_mu(kspace)
        f=(lambda e,mu: 1 if e<=mu else 0) if abs(self.temperature)<RZERO else (lambda e,mu: 1/(exp((e-mu)/self.temperature)+1))
        nmatrix=len(self.generators['h'].table)
        nspin=self.config.values()[0].nspin
        buff=zeros((nmatrix,nmatrix),dtype=complex128)
        for matrix in self.matrices(kspace):
            eigs,eigvecs=eigh(matrix)
            for eig,eigvec in zip(eigs,eigvecs.T):
                buff+=dot(eigvec.conj().reshape((nmatrix,1)),eigvec.reshape((1,nmatrix)))*f(eig,self.mu)
        nstate=(1 if kspace is None else kspace.rank['k'])*nmatrix
        for key in self.ops.keys():
            self.ops[key].value=sum(buff*self.ops[key].matrix)/(nstate/nspin)

    def iterate_sec(self,kspace=None,error=10**-4,n=200):
        stime=time.time()
        err,count=inf,0
        op_old=array([self.ops[key].value for key in self.ops.keys()])
        while err>=error:
            count+=1
            if count==1:
                self.update_ops(kspace)
                op_new=array([self.ops[key].value for key in self.ops.keys()])
                gx_old=op_new
            elif count>n:
                raise ValueError("SCMF iterate error: the iterations has exceeded the max step.")
            else:
                self.update_ops(kspace)
                gx_new=array([self.ops[key].value for key in self.ops.keys()])-op_new
                buff=op_new-(op_new-op_old)/(gx_new-gx_old)*gx_new
                op_old=op_new
                op_new=buff
                gx_old=gx_new
                err=max(abs(op_old-op_new))
            print 'Step,op,error: ',count,',',op_old,err
        #print 'Order parameters:',op_old
        etime=time.time()
        print 'Iterate: time consumed ',etime-stime,'s.'

    def iterate(self,kspace=None,error=10**-6,n=200):
        stime=time.time()
        def gx(values):
            for op,value in zip(self.ops.values(),values):
                op.value=value
            self.update_ops(kspace)
            return array([self.ops[key].value for key in self.ops.keys()])-values
        x0=array([self.ops[key].value for key in self.ops.keys()])
        buff=broyden2(gx,x0,verbose=True,reduction_method='svd',maxiter=n,x_tol=error)
        print 'Order parameters:',buff
        etime=time.time()
        print 'Iterate: time consumed ',etime-stime,'s.'
