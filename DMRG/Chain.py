'''
Chain for DMRG algorithm, including:
1) classes: Block,Chain
2) functions: EmptyChain
'''

__all__=['Block','Chain','EmptyChain']

import numpy as np
import scipy.sparse as sp
import time
from scipy.sparse.linalg import eigsh
from ..Basics import *
from ..Math import dagger,Tensor
from MPS import *
from MPO import *
from linalg import kron,kronsum,vblock_svd
from copy import copy,deepcopy

class Block(int):
    '''
    The block of a chain.
    Attribues:
        form: 'L', 'S' or 'R'
            The form of the block.
            'L' for left, 'R' for right and 'S' for site.
        pos: integer
            The position of the block in the chain.
        label: Label
            The label of the block.
    '''

    def __new__(cls,nsite,form,pos,label):
        '''
        Constructor.
        Parameters:
            nsite: integer
                The number of sites of the superblock.
            form: 'L','S','R'
                The form of the block.
            pos: integer
                The position of the block in the superblock.
            label: Label
                The Label of the block.
        '''
        assert form in 'LSR'
        self=int.__new__(cls,0 if pos is None else (pos if form=='S' else (pos+1 if form=='L' else nsite-pos)))
        self.form=form
        self.pos=pos
        self.label=label
        return self

    def __copy__(self):
        '''
        Copy.
        '''
        result=int.__new__(self.__class__,self)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self,memo):
        '''
        Deepcopy.
        '''
        result=int.__new__(self.__class__,self)
        result.__dict__.update(deepcopy(self.__dict__))
        return result

    @property
    def nbasis(self):
        '''
        The number of basis on this block.
        '''
        qnc=self.label.qnc
        if isinstance(qnc,QuantumNumberCollection):
            return qnc.n
        else:
            return qnc

    @property
    def qnc(self):
        '''
        The quantum number collections of the block.
        '''
        return self.label.qnc

    @qnc.setter
    def qnc(self,value):
        '''
        Reset the quantum number collections of the block.
        Parameters:
            value: integer or QuantumNumberCollection
                The new quantum number collection of the block.
        '''
        self.label.qnc=value

class Chain(MPS):
    '''
    The chain of blocks.
    Attribues:
        optstrs: List of OptStr
            The optstrs on the chain.
        target: QuantumNumber
            The target space of the chain.
        _Hs_: dict
            It has three items:
            1) _Hs_["L"]: list of 2d ndarray
                The Hamiltonians of the left blocks.
            2) _Hs_["S"]: list of 2d ndarray
                The Hamiltonians of the single-site blocks.
            3) _Hs_["R"]: list of 2d ndarray
                The Hamiltonians of the right blocks.
        blocks: dict
            It has three items:
            1) blocks["L"]: list of OptStr
                The optstrs of the left blocks.
            2) blocks["S"]: list of OptStr
                The optstrs of the single-site blocks.
            3) blocks["R"]: list of OptStr
                The optstrs of the right blocks.
        connections: dict
            It has three items:
            1) connections["LR"]: list of OptStr
                The connecting optstrs between the system block and environment block.
            2) connections["L"]: list of OptStr
                The connecting optstrs between the single-site block and the A block.
            3) connections["R"]: list of OptStr
                The connecting optstrs between the single-site block and the B block.
        logger: TimerLogger
            The timers to record the time that every process consumes.
        info: dict
            It has five items up to now:
            1) info['gse']: float64
                The ground state energy per site of the chain.
            2) info['overlap']: complex128
                The overlap between the new state and the predicted state.
            3) info['nbasis']: integer
                The number of bond dimension at the current position.
            4) info['err']: float64
                The truncation error.
            5) info['nnz']: integer
                The number of non-zeros of the whole chain's Hamiltonian.
        cache: dict
            It has only one item up to now:
            1) cache['qnc']: QuantumNumberCollection
                The quantum number collection of the whole chain.
    '''

    def __init__(self,optstrs=[],mode='QN',ms=[],labels=None,Lambda=None,cut=None,target=None):
        '''
        Constructor.
        Parameters:
            optstrs: list of OptStr, optional
                The optstrs in the chain.
            mode,ms,labels,Lambda,cut: optional
                Please see HamiltonianPy.DMRG.MPS.MPS.__init__ for details.
            target: QuantumNumber
                The target space of the chain.
        '''
        assert mode in ('QN','NB')
        MPS.__init__(self,mode=mode,ms=ms,labels=labels,Lambda=Lambda,cut=cut)
        self.optstrs=optstrs
        self.set_blocks_and_connections()
        self.set_Hs_()
        self.target=target
        self.cache={'qnc':None}
        self.info={'gse':None,'overlap':None,'nbasis':None}
        self.logger=TimerLogger('Preparation','Hamiltonian','Diagonalization','Truncation','Total')
        self.logger.proceed('Total')

    def set_blocks_and_connections(self):
        '''
        Set the blocks and connections of the chain.
        '''
        temp=[[] for i in xrange(self.nsite)]
        self.blocks={"L":deepcopy(temp),"S":deepcopy(temp),"R":deepcopy(temp)}
        self.connections={"LR":deepcopy(temp),"L":deepcopy(temp),"R":deepcopy(temp)}
        for optstr in self.optstrs:
            temp=sorted([self.table[label] for label in optstr.labels])
            if len(temp)==1:
                self.blocks["S"][temp[0]].append(optstr)
            else:
                if temp[0]>0:
                    self.connections["R"][self.nsite-temp[0]].append(optstr)
                if temp[-1]+1<self.nsite:
                    self.connections["L"][temp[-1]+1].append(optstr)
                for i in xrange(len(temp)-1):
                    for j in xrange(temp[i],temp[i+1]):
                        self.connections["LR"][j+1].append(optstr)
        for i in xrange(1,self.nsite):
            self.blocks["L"][i]=self.blocks["L"][i-1]+self.blocks["S"][i-1]+self.connections["L"][i]
            self.blocks["R"][i]=self.blocks["R"][i-1]+self.blocks["S"][i-1]+self.connections["R"][i]

    def set_Hs_(self):
        '''
        Set the Hamiltonians of blocks.
        '''
        self._Hs_={"L":[None]*self.nsite,"S":[None]*self.nsite,"R":[None]*self.nsite}
        if self.cut is not None:
            for i in xrange(self.cut-1):
                new=Block(nsite=self.nsite,form='L',pos=i,label=self[i].labels[MPS.R])
                site=Block(nsite=self.nsite,form='S',pos=i,label=self[i].labels[MPS.S])
                if i==0:
                    self._Hs_['L'][new]=np.zeros((new.nbasis,new.nbasis))
                else:
                    self._Hs_['L'][new]=np.einsum('ikm,ij,jkn->mn',np.asarray(self[new.pos]).conjugate(),self.H(old),np.asarray(self[new.pos]))
                us=self.us(new)
                for optstr in self.blocks['S'][site]+self.connections['L'][new]:
                    self._Hs_['L'][new]+=np.asarray(optstr.matrix(us=us,form='L'))
                old=new
            for i in reversed(range(self.cut,self.nsite)):
                new=Block(nsite=self.nsite,form='R',pos=i,label=self[i].labels[MPS.L])
                site=Block(nsite=self.nsite,form='S',pos=i,label=self[i].labels[MPS.S])
                if i==self.nsite-1:
                    self._Hs_['R'][new]=np.zeros((new.nbasis,new.nbasis))
                else:
                    self._Hs_['R'][new]=np.einsum('mki,ij,nkj->mn',np.asarray(self[new.pos]).conjugate(),self.H(old),np.asarray(self[new.pos]))
                us=self.us(new)
                for optstr in self.blocks['S'][site]+self.connections['R'][new]:
                    self._Hs_['R'][new]+=np.asarray(optstr.matrix(us=us,form='R'))
                old=new

    def update(self,optstrs):
        '''
        Update the chain.
        Parameters:
            optstrs: list of OptStr
                The new optstrs of the chain.
        '''
        self.optstrs=optstrs
        self.set_blocks_and_connections()
        self.set_Hs_()

    @property
    def sys(self):
        '''
        The system block.
        '''
        return Block(nsite=self.nsite,form='L',pos=self.cut-1,label=self[self.cut-1].labels[MPS.R])

    @property
    def env(self):
        '''
        The environment block.
        '''
        return Block(nsite=self.nsite,form='R',pos=self.cut,label=self[self.cut].labels[MPS.L])

    @property
    def A(self):
        '''
        The A block.
        '''
        assert self.cut>0
        if self.cut==1:
            return Block(nsite=self.nsite,form='L',pos=None,label=self[0].labels[MPS.L])
        else:
            return Block(nsite=self.nsite,form='L',pos=self.cut-2,label=self[self.cut-2].labels[MPS.R])

    @property
    def Asite(self):
        '''
        The Asite block.
        '''
        return Block(nsite=self.nsite,form='S',pos=self.cut-1,label=self[self.cut-1].labels[MPS.S])

    @property
    def Bsite(self):
        '''
        The Bsite block.
        '''
        return Block(nsite=self.nsite,form='S',pos=self.cut,label=self[self.cut].labels[MPS.S])

    @property
    def B(self):
        '''
        The B block.
        '''
        assert self.cut<self.nsite
        if self.cut==self.nsite-1:
            return Block(nsite=self.nsite,form='R',pos=None,label=self[-1].labels[MPS.R])
        else:
            return Block(nsite=self.nsite,form='R',pos=self.cut+1,label=self[self.cut+1].labels[MPS.L])

    @property
    def graph(self):
        '''
        The graph representation of the chain.
        '''
        return ''.join(['A'*(self.cut-1),'..','B'*(self.nsite-self.cut-1)])

    def us(self,block):
        '''
        Get the sub-mps corresponding to a block.
        Parameters:
            block: Block
                The block.
        Returns: MPS
            The sub-mps corresponding to a block.
        '''
        assert isinstance(block,Block)
        if block.pos is None:
            return MPS(ms=[])
        elif block.form=='L':
            return MPS(ms=self[0:block.pos+1])
        elif block.form=='R':
            return MPS(ms=self[block.pos:])
        else:
            return MPS(ms=[self[block.pos]])

    def H(self,block,force=False):
        '''
        Get the block Hamiltonian.
        Parameters:
            block: Block
                The block.
            force: logical, optional
                When True, the H are forced to be recalculated.
                Otherwise not.
        Returns: 2d ndarray
            The block Hamiltonian.
        '''
        assert isinstance(block,Block)
        if self._Hs_[block.form][block] is None or force:
            if block==0 and block.form in "LR":
                self._Hs_[block.form][block]=0.0
            else:
                self._Hs_[block.form][block]=np.zeros((block.nbasis,block.nbasis))
                us,form=self.us(block),block.form
                for optstr in self.blocks[block.form][block]:
                    self._Hs_[block.form][block]+=np.asarray(optstr.matrix(us=us,form=form))
        return self._Hs_[block.form][block]

    @property
    def matrix(self):
        '''
        The Hamiltonian of the whole chain.
        '''
        sys,env,qnc=self.sys,self.env,self.cache['qnc']
        ussys,usenv=self.us(sys),self.us(env)
        target=None if self.mode=='NB' else self.target.zeros
        result=kronsum(self.H(env),self.H(sys),qnc1=env.qnc,qnc2=sys.qnc,qnc=qnc,target=target,format='csr')
        for optstr in self.connections['LR'][self.cut]:
            a,b=optstr.split(ussys.table,usenv.table,coeff='A')
            result+=kron(a.matrix(ussys,'L'),b.matrix(usenv,'R'),qnc1=sys.qnc,qnc2=env.qnc,qnc=qnc,target=target,format='csr')
        return result

    def two_site_update(self):
        '''
        The two site update, which resets the central two mps and Hamiltonians of the chain.
        '''
        self.logger.proceed('Preparation')
        A,Asite,sys=self.A,self.Asite,self.sys
        usa,usasite=self.us(A),self.us(Asite)
        u=np.identity(A.nbasis*Asite.nbasis).reshape((A.nbasis,Asite.nbasis,-1))
        ha=np.kron(self.H(A),np.identity(Asite.nbasis))+np.kron(np.identity(A.nbasis),self.H(Asite))
        for optstr in self.connections[sys.form][sys]:
            a,b=optstr.split(usa.table,usasite.table,coeff='B')
            ha+=np.kron(a.matrix(usa,'L'),b.matrix(usasite,'S'))
        B,Bsite,env=self.B,self.Bsite,self.env
        usb,usbsite=self.us(B),self.us(Bsite)
        v=np.identity(Bsite.nbasis*B.nbasis).reshape((-1,Bsite.nbasis,B.nbasis))
        hb=np.kron(self.H(Bsite),np.identity(B.nbasis))+np.kron(np.identity(Bsite.nbasis),self.H(B))
        for optstr in self.connections[env.form][env]:
            a,b=optstr.split(usbsite.table,usb.table,coeff='A')
            hb+=np.kron(a.matrix(usbsite,'S'),b.matrix(usb,'R'))
        if self.mode=='QN':
            QuantumNumberCollection.clear_history(sys.qnc,env.qnc,self.cache['qnc'])
            sys.qnc=A.qnc.kron(Asite.qnc,'+',history=True)
            env.qnc=Bsite.qnc.kron(B.qnc,'-',history=True)
            self.cache['qnc']=sys.qnc.kron(env.qnc,'+',history=True)
            self[sys.pos]=Tensor(sys.qnc.reorder(u,axes=[2]),labels=self[sys.pos].labels)
            self[env.pos]=Tensor(env.qnc.reorder(v,axes=[0]),labels=self[env.pos].labels)
            self._Hs_[sys.form][sys]=sys.qnc.reorder(ha,axes=[0,1])
            self._Hs_[env.form][env]=env.qnc.reorder(hb,axes=[0,1])
        else:
            sys.qnc=A.qnc*Asite.qnc
            env.qnc=Bsite.qnc*B.qnc
            self.cache['qnc']=sys.qnc*env.qnc
            self[sys.pos]=Tensor(u,labels=self[sys.pos].labels)
            self[env.pos]=Tensor(v,labels=self[env.pos].labels)
            self._Hs_[sys.form][sys]=ha
            self._Hs_[env.form][env]=hb
        self.logger.suspend('Preparation')

    def two_site_truncate(self,v0=None,nmax=200,tol=5*10**-14):
        '''
        The two site truncation, which truncates the central two site mps and Hamiltonians.
        Parameters:
            v0: 1d ndarray
                The initial state used to diagonalize the Hamiltonian of the whole chain.
            nmax: integer
                The maximum singular values to be kept.
            tol: float64
                The tolerance of the singular values.
        '''
        sys,env,qnc=self.sys,self.env,self.cache['qnc']
        self.logger.proceed('Hamiltonian')
        matrix=self.matrix
        self.info['nnz']=matrix.nnz
        self.logger.suspend('Hamiltonian')
        self.logger.proceed('Diagonalization')
        gse,Psi=eigsh(matrix,which='SA',v0=v0,k=1)
        self.info['gse']=gse/self.nsite
        self.info['overlap']=None if v0 is None else Psi[:,0].conjugate().dot(v0)
        self.logger.suspend('Diagonalization')
        self.logger.proceed('Truncation')
        if self.mode=='QN':
            tsys,tenv=[],[]
            for qnsys,qnenv in qnc.pairs(self.target.zeros):
                tsys.append(qnsys)
                tenv.append(qnenv)
            U,S,V,new,err=vblock_svd(Psi,sys.qnc.subset(tsys),env.qnc.subset(tenv),nmax=nmax,tol=tol,return_truncation_err=True)
            sysslice=sys.qnc.subslice(tsys)
            envslice=env.qnc.subslice(tenv)
            self[sys.pos]=Tensor(np.einsum('ijk,kl->ijl',np.asarray(self[sys.pos])[:,:,sysslice],U),labels=self[sys.pos].labels)
            self[env.pos]=Tensor(np.einsum('lk,kji->lji',V,np.asarray(self[env.pos])[envslice,:,:]),labels=self[env.pos].labels)
            self._Hs_[sys.form][sys]=dagger(U).dot(self.H(sys)[:,sysslice][sysslice,:]).dot(U)
            self._Hs_[env.form][env]=V.dot(self.H(env)[:,envslice][envslice,:]).dot(dagger(V))
        else:
            U,S,V,new,err=vblock_svd(Psi,sys.qnc,env.qnc,nmax=nmax,tol=tol,return_truncation_err=True)
            self[sys.pos]=Tensor(np.einsum('ijk,kl->ijl',np.asarray(self[sys.pos]),U),labels=self[sys.pos].labels)
            self[env.pos]=Tensor(np.einsum('lk,kji->lji',V,np.asarray(self[env.pos])),labels=self[env.pos].labels)
            self._Hs_[sys.form][sys]=dagger(U).dot(self.H(sys)).dot(U)
            self._Hs_[env.form][env]=V.dot(self.H(env)).dot(dagger(V))
        self.Lambda=Tensor(S,labels=[self[sys.pos].labels[MPS.R]])
        QuantumNumberCollection.clear_history(sys.qnc,env.qnc)
        sys.qnc=new
        env.qnc=new
        self.logger.suspend('Truncation')
        self.info['nbasis']=sys.nbasis
        self.info['err']=err

    def two_site_grow(self,AL,BL,optstrs,target=None,nmax=200,tol=5*10**-14):
        '''
        Two site grow of the chain.
        Parameters:
            AL,BL: Label
                The labels for the two added sites.
            optstrs: list of OptStr
                The optstrs of the new chain.
            target: QuantumNumber, optional
                The new target space.
            nmax: integer
                The maximum singular values to be kept.
            tol: float64
                The tolerance of the singular values.
        '''
        assert self.cut==None or self.cut==self.nsite/2
        assert AL not in self.table and BL not in self.table
        if self.mode=='QN':
            diff=QuantumNumberCollection([] if target==self.target else [(target if self.target is None else target-self.target,1)])
        for m in self[self.nsite/2:]:
            L,R=m.labels[MPS.L],m.labels[MPS.R]
            m.labels[MPS.L]=L.replace(identifier=L.identifier+2,qnc=L.qnc.kron(diff,'+') if self.mode=='QN' else L.qnc)
            m.labels[MPS.R]=R.replace(identifier=0 if R.identifier==0 else R.identifier+2,qnc=R.qnc.kron(diff,'+') if self.mode=='QN' else R.qnc)
        alabels,blabels=[None]*3,[None]*3
        if self.cut is None:
            alabels[MPS.L]=Label(identifier=0,qnc=1 if target is None else QuantumNumberCollection([(target.zeros,1)]))
            blabels[MPS.R]=Label(identifier=0,qnc=1 if target is None else QuantumNumberCollection([(target,1)]))
            self.cut=0
        else:
            alabels[MPS.L]=deepcopy(self[self.cut-1].labels[MPS.R])
            blabels[MPS.R]=deepcopy(self[self.cut].labels[MPS.L])
        alabels[MPS.S],blabels[MPS.S]=AL,BL
        alabels[MPS.R]=Label(identifier=self.cut+1,qnc=None)
        blabels[MPS.L]=Label(identifier=self.cut+1,qnc=None)
        self.insert(self.cut,Tensor([[[0.0]]],labels=alabels))
        self.insert(self.cut+1,Tensor([[[0.0]]],labels=blabels))
        self.cut+=1
        self.optstrs=optstrs
        self.set_blocks_and_connections()
        self._Hs_["L"].extend([None,None])
        self._Hs_["S"].extend([None,None])
        self._Hs_["R"].extend([None,None])
        self.target=target
        print 'ChainRep: %s(++)'%(self.graph)
        self.two_site_update()
        self.two_site_truncate(nmax=nmax,tol=tol)
        self.logger.record()
        print self.logger
        print 'nnz,nbasis,err: %s,%s,%s.'%(self.info['nnz'],self.info['nbasis'],self.info['err'])
        print 'gse: %s.'%(self.info['gse'][0])
        print

    def two_site_sweep(self,direction,nmax=200,tol=5*10**-14):
        '''
        Two site sweep of the chain.
        Parameters:
            direction: 'L' or 'R'
                The direction of the sweep.
            nmax: integer
                The maximum singular values to be kept.
            tol: float64
                The tolerance of the singular values.
        '''
        if direction=='L':
            self<<=1
            print 'ChainRep: %s(<<)'%(self.graph)
        else:
            self>>=1
            print 'ChainRep: %s(>>)'%(self.graph)
        A,Asite,sys=self.A,self.Asite,self.sys
        B,Bsite,env=self.B,self.Bsite,self.env
        ml=np.asarray(self[sys.pos]).reshape((A.nbasis*Asite.nbasis,sys.nbasis))
        mr=np.asarray(self[env.pos]).reshape((env.nbasis,Bsite.nbasis*B.nbasis))
        self.two_site_update()
        if self.mode=='QN':
            ml=sys.qnc.reorder(ml,axes=[0])
            mr=env.qnc.reorder(mr,axes=[1])
        v0=np.einsum('ik,k,kj->ij',ml,np.asarray(self.Lambda),mr).ravel()
        if self.mode=='QN':
            v0=self.cache['qnc'].reorder(v0,axes=[0])
            v0=v0[self.cache['qnc'][self.target.zeros]]
        v0/=np.linalg.norm(v0)
        self.two_site_truncate(v0=v0,nmax=nmax,tol=tol)
        self.logger.record()
        print self.logger
        print 'nnz,nbasis,overlap,err: %s,%s,%s,%s.'%(self.info['nnz'],self.info['nbasis'],self.info['overlap'],self.info['err'])
        print 'gse: %s.'%(self.info['gse'][0])
        print

def EmptyChain(mode='QN'):
    '''
    Construt an empty chain.
    Parameters:
        mode: optional
            Please see Chain.__init__ for details.
    Returns: Chain
        An empty chain.
    '''
    return Chain(mode=mode)
