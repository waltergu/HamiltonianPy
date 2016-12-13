'''
DMRG, including:
1) classes: Block,DMRG,TSG,TSS
2) function: DMRGTSG,DMRGTSS
'''

__all__=['Block','DMRG','TSG','DMRGTSG','TSS','DMRGTSS']

import numpy as np
import pickle as pk
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from ..Basics import *
from ..Math import kron,kronsum,dagger,Tensor
from MPS import *
from MPO import *
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

    def __getnewargs__(self):
        '''
        Return the arguments for Block.__new__, required by copy and pickle.
        '''
        return (None if self.form in 'LS' else self.pos+self,self.form,self.pos,self.label)

    def __getstate__(self):
        '''
        Since Block.__new__ constructs everything, self.__dict__ can be omitted for copy and pickle.
        '''
        pass

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return 'Block(%s,form=%s,pos=%s,label=%s)'%(int(self),self.form,self.pos,self.label)

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

class DMRG(Engine):
    '''
   Density matrix renormalization group method.
    Attribues:
        mps: MPS
            The matrix product state of the DMRG.
        lattice: SuperLattice
            The final lattice of the DMRG.
        terms: list of Term
            The terms of the DMRG.
        config: IDFConfig
            The configuration of the internal degrees of freedom on the lattice.
        degfres: DegFreTree
            The physical degrees of freedom tree.
        layer: tuple of string
            The layer on which the DMRG works.
        mask: [] or ['nambu']
            [] for spin systems and ['nambu'] for fermionic systems.
        target: QuantumNumber
            The target space of the DMRG.
        dtype: np.float64, np.complex128
            The data type.
        generators: dict
            entry 'h': Generator
                The generator of the Hamiltonian.
        operators: dict
            entry 'h': OperatorCollection
                The operator of the Hamiltonian.
        optstrs: dict
            entry 'h': list of OptStr
                The optstrs of the Hamiltonian.
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
            It has three items up to now:
            1) cache['qnc']: QuantumNumberCollection
                The quantum number collection of the whole chain.
            2) subslice: list of integer
                The subslice of the DMRG's target space in the kron order.
            3) permutation: list of integer
                The permutation of the DMRG's target space to be in the order with respect to cache['qnc'].
    '''

    def __init__(self,name,mps,lattice,terms,config,degfres,layer=None,mask=[],target=None,dtype=np.complex128,**karg):
        '''
        Constructor.
        Parameters:
            name: sting
                The prefix of the name of the DMRG.
            mps: MPS
                The matrix product state of the DMRG.
            lattice: Lattice
                The lattice of the DMRG.
            terms: list of Term
                The terms of the DMRG.
            config: IDFConfig
                The configuration of the internal degrees of freedom on the lattice.
            degfres: DegFreTree
                The physical degrees of freedom tree.
            layer: tuple of string
                The layer on which the DMRG works.
            mask: [] or ['nambu']
                [] for spin systems and ['nambu'] for fermionic systems.
            dtype: np.float64,np.complex128, optional
                The data type.
        '''
        self.mps=mps
        self.lattice=lattice
        self.terms=terms
        self.config=config
        self.degfres=degfres
        self.layer=degfres.layers[0] if layer is None else degfres.layers[degfres.layers.index(layer)]
        self.mask=mask
        self.target=target
        self.dtype=dtype
        self.set_generators_operators_optstrs()
        self.status.update(const=self.generators['h'].parameters['const'])
        self.status.update(alter=self.generators['h'].parameters['alter'])
        self.set_blocks_and_connections()
        self.set_Hs_()
        self.cache={'qnc':None}
        self.info={'gse':None,'overlap':None,'nbasis':None}
        self.logger=TimerLogger('Preparation','Hamiltonian','Diagonalization','Truncation','Total')
        self.logger.proceed('Total')

    def update(self,**karg):
        '''
        Update the DMRG with new parameters.
        '''
        self.generators['h'].update(**karg)
        self.operators['h']=self.generators['h'].operators
        self.optstrs['h']=[OptStr.from_operator(opt,self.degfres,self.layer) for opt in self.operators['h'].itervalues()]
        self.status.update(alter=self.generators['h'].parameters['alter'])
        self.set_blocks_and_connections()
        self.set_Hs_()

    def set_generators_operators_optstrs(self):
        '''
        Set the generators, operators and optstrs of the DMRG.
        '''
        self.generators={'h':Generator(bonds=self.lattice.bonds,config=self.config,terms=self.terms,dtype=self.dtype)}
        self.operators={'h':self.generators['h'].operators}
        if self.mask==['nambu']:
            for operator in self.operators['h'].values():
                self.operators['h']+=operator.dagger
        self.optstrs={'h':[OptStr.from_operator(opt,self.degfres,self.layer) for opt in self.operators['h'].itervalues()]}

    def set_blocks_and_connections(self):
        '''
        Set the blocks and connections of the DMRG.
        '''
        temp=[[] for i in xrange(self.mps.nsite+1)]
        self.blocks={"L":deepcopy(temp),"S":deepcopy(temp),"R":deepcopy(temp)}
        self.connections={"LR":deepcopy(temp),"L":deepcopy(temp),"R":deepcopy(temp)}
        for optstr in self.optstrs['h']:
            temp=sorted([self.mps.table[label] for label in optstr.labels])
            if len(temp)==1:
                self.blocks["S"][temp[0]].append(optstr)
            else:
                if temp[0]>0:
                    self.connections["R"][self.mps.nsite-temp[0]].append(optstr)
                if temp[-1]+1<self.mps.nsite:
                    self.connections["L"][temp[-1]+1].append(optstr)
                for i in xrange(len(temp)-1):
                    for j in xrange(temp[i],temp[i+1]):
                        self.connections["LR"][j+1].append(optstr)
        for i in xrange(1,self.mps.nsite):
            self.blocks["L"][i]=self.blocks["L"][i-1]+self.blocks["S"][i-1]+self.connections["L"][i]
            self.blocks["R"][i]=self.blocks["R"][i-1]+self.blocks["S"][i-1]+self.connections["R"][i]

    def set_Hs_(self):
        '''
        Set the Hamiltonians of blocks.
        '''
        self._Hs_={"L":[None]*(self.mps.nsite+1),"S":[None]*self.mps.nsite,"R":[None]*(self.mps.nsite+1)}
        if self.mps.cut is not None:
            for i in xrange(self.mps.cut-1):
                new=Block(nsite=self.mps.nsite,form='L',pos=i,label=self.mps[i].labels[MPS.R])
                site=Block(nsite=self.mps.nsite,form='S',pos=i,label=self.mps[i].labels[MPS.S])
                if i==0:
                    self._Hs_['L'][new]=np.zeros((new.nbasis,new.nbasis))
                else:
                    self._Hs_['L'][new]=np.einsum('ikm,ij,jkn->mn',np.asarray(self.mps[new.pos]).conjugate(),self.H(old),np.asarray(self.mps[new.pos]))
                us=self.us(new)
                for optstr in self.blocks['S'][site]+self.connections['L'][new]:
                    self._Hs_['L'][new]+=np.asarray(optstr.matrix(us=us,form='L'))
                old=new
            for i in reversed(range(self.mps.cut,self.mps.nsite)):
                new=Block(nsite=self.mps.nsite,form='R',pos=i,label=self.mps[i].labels[MPS.L])
                site=Block(nsite=self.mps.nsite,form='S',pos=i,label=self.mps[i].labels[MPS.S])
                if i==self.mps.nsite-1:
                    self._Hs_['R'][new]=np.zeros((new.nbasis,new.nbasis))
                else:
                    self._Hs_['R'][new]=np.einsum('mki,ij,nkj->mn',np.asarray(self.mps[new.pos]).conjugate(),self.H(old),np.asarray(self.mps[new.pos]))
                us=self.us(new)
                for optstr in self.blocks['S'][site]+self.connections['R'][new]:
                    self._Hs_['R'][new]+=np.asarray(optstr.matrix(us=us,form='R'))
                old=new

    @property
    def sys(self):
        '''
        The system block.
        '''
        return Block(nsite=self.mps.nsite,form='L',pos=self.mps.cut-1,label=self.mps[self.mps.cut-1].labels[MPS.R])

    @property
    def env(self):
        '''
        The environment block.
        '''
        return Block(nsite=self.mps.nsite,form='R',pos=self.mps.cut,label=self.mps[self.mps.cut].labels[MPS.L])

    @property
    def A(self):
        '''
        The A block.
        '''
        assert self.mps.cut>0
        if self.mps.cut==1:
            return Block(nsite=self.mps.nsite,form='L',pos=None,label=self.mps[0].labels[MPS.L])
        else:
            return Block(nsite=self.mps.nsite,form='L',pos=self.mps.cut-2,label=self.mps[self.mps.cut-2].labels[MPS.R])

    @property
    def Asite(self):
        '''
        The Asite block.
        '''
        return Block(nsite=self.mps.nsite,form='S',pos=self.mps.cut-1,label=self.mps[self.mps.cut-1].labels[MPS.S])

    @property
    def Bsite(self):
        '''
        The Bsite block.
        '''
        return Block(nsite=self.mps.nsite,form='S',pos=self.mps.cut,label=self.mps[self.mps.cut].labels[MPS.S])

    @property
    def B(self):
        '''
        The B block.
        '''
        assert self.mps.cut<self.mps.nsite
        if self.mps.cut==self.mps.nsite-1:
            return Block(nsite=self.mps.nsite,form='R',pos=None,label=self.mps[-1].labels[MPS.R])
        else:
            return Block(nsite=self.mps.nsite,form='R',pos=self.mps.cut+1,label=self.mps[self.mps.cut+1].labels[MPS.L])

    @property
    def graph(self):
        '''
        The graph representation of the DMRG.
        '''
        result=[]
        temp=''.join(['A'*(self.mps.cut-1),'..','B'*(self.mps.nsite-self.mps.cut-1)])
        while temp:
            result.append(temp[0:87])
            temp=temp[87:]
        return '\n'.join(result)

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
            return MPS(ms=self.mps[0:block.pos+1])
        elif block.form=='R':
            return MPS(ms=self.mps[block.pos:])
        else:
            return MPS(ms=[self.mps[block.pos]])

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
        The Hamiltonian of the whole DMRG.
        '''
        sys,env=self.sys,self.env
        ussys,usenv=self.us(sys),self.us(env)
        rows,cols=(None,None) if self.mps.mode=='NB' else (self.cache['subslice'],self.cache['subslice'])
        result=kronsum(self.H(env),self.H(sys),rows=rows,cols=cols,format='csr')
        for optstr in self.connections['LR'][self.mps.cut]:
            a,b=optstr.split(ussys.table,usenv.table,coeff='A')
            result+=kron(a.matrix(ussys,'L'),b.matrix(usenv,'R'),rows=rows,cols=cols,format='csr')
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
        if self.mps.mode=='QN':
            sys.qnc=A.qnc.kron(Asite.qnc,'+',history=True)
            env.qnc=Bsite.qnc.kron(B.qnc,'-',history=True)
            self.cache['qnc']=sys.qnc.kron(env.qnc,'+',history=True)
            self.mps[sys.pos]=Tensor(sys.qnc.reorder(u,axes=[2]),labels=self.mps[sys.pos].labels)
            self.mps[env.pos]=Tensor(env.qnc.reorder(v,axes=[0]),labels=self.mps[env.pos].labels)
            self._Hs_[sys.form][sys]=sys.qnc.reorder(ha,axes=[0,1])
            self._Hs_[env.form][env]=env.qnc.reorder(hb,axes=[0,1])
            permutation=self.cache['qnc'].permutation(targets=[self.target.zeros])
            antipermutation=np.argsort(permutation)
            self.cache['subslice']=np.array(permutation)[antipermutation]
            self.cache['permutation']=np.argsort(antipermutation)
        else:
            sys.qnc=A.qnc*Asite.qnc
            env.qnc=Bsite.qnc*B.qnc
            self.cache['qnc']=sys.qnc*env.qnc
            self.mps[sys.pos]=Tensor(u,labels=self.mps[sys.pos].labels)
            self.mps[env.pos]=Tensor(v,labels=self.mps[env.pos].labels)
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
        self.info['gse']=gse/self.mps.nsite
        self.info['overlap']=None if v0 is None else Psi[:,0].conjugate().dot(v0)
        self.logger.suspend('Diagonalization')
        self.logger.proceed('Truncation')
        if self.mps.mode=='QN':
            tsys,tenv=[],[]
            for qnsys,qnenv in qnc.pairs(self.target.zeros):
                tsys.append(qnsys)
                tenv.append(qnenv)
            U,S,V,new,err=vb_svd(Psi[self.cache['permutation']],sys.qnc.subset(tsys),env.qnc.subset(tenv),nmax=nmax,tol=tol,return_truncation_err=True)
            sysslice=sys.qnc.subslice(tsys)
            envslice=env.qnc.subslice(tenv)
            self.mps[sys.pos]=Tensor(np.einsum('ijk,kl->ijl',np.asarray(self.mps[sys.pos])[:,:,sysslice],U),labels=self.mps[sys.pos].labels)
            self.mps[env.pos]=Tensor(np.einsum('lk,kji->lji',V,np.asarray(self.mps[env.pos])[envslice,:,:]),labels=self.mps[env.pos].labels)
            self._Hs_[sys.form][sys]=dagger(U).dot(self.H(sys)[:,sysslice][sysslice,:]).dot(U)
            self._Hs_[env.form][env]=V.dot(self.H(env)[:,envslice][envslice,:]).dot(dagger(V))
        else:
            U,S,V,new,err=vb_svd(Psi,sys.qnc,env.qnc,nmax=nmax,tol=tol,return_truncation_err=True)
            self.mps[sys.pos]=Tensor(np.einsum('ijk,kl->ijl',np.asarray(self.mps[sys.pos]),U),labels=self.mps[sys.pos].labels)
            self.mps[env.pos]=Tensor(np.einsum('lk,kji->lji',V,np.asarray(self.mps[env.pos])),labels=self.mps[env.pos].labels)
            self._Hs_[sys.form][sys]=dagger(U).dot(self.H(sys)).dot(U)
            self._Hs_[env.form][env]=V.dot(self.H(env)).dot(dagger(V))
        self.mps.Lambda=Tensor(S,labels=[self.mps[sys.pos].labels[MPS.R]])
        QuantumNumberCollection.clear_history(sys.qnc,env.qnc,self.cache['qnc'])
        sys.qnc=new
        env.qnc=new
        self.logger.suspend('Truncation')
        self.info['nbasis']=sys.nbasis
        self.info['err']=err

    def level_up(self,n=1):
        '''
        Move the physical indices n-level up in self.degfres.
        Parameters:
            n: integer, optional
                The degree of level to go up.
        '''
        self.mps=MPS.level_up(self.mps,self.degfres,n=n)
        self.layer=self.degfres.layers[self.degfres.layers.index(self.layer)+n]
        self.optstrs['h']=[OptStr.from_operator(opt,self.degfres,self.layer) for opt in self.operators['h'].itervalues()]
        self.set_blocks_and_connections()
        self.set_Hs_()

    def level_down(self,n=1):
        '''
        Move the physical indices n-level down in self.degfres.
        Parameters:
            n: integer, optional
                The degree of level to go down.
        '''
        raise NotImplementedError()

class TSG(App):
    '''
    Two site growth of a DMRG.
    Attribues:
        block: Lattice
            The building block of the lattice of a DMRG.
        vector: 1d ndarray
            The translation vector of the left blocks and right blocks before the addition of two new ones.
        scopes: list of hashable objects
            The scopes of the blocks to be added two-by-two into the lattice of the DMRG.
        targets: sequence of QuantumNumber
            The target space at each growth of the DMRG.
        nmax: integer
            The maximum singular values to be kept.
        tol: float64
            The tolerance of the singular values.
    '''

    def __init__(self,block,vector,scopes,targets,nmax=200,tol=5*10**-14,**karg):
        '''
        Constructor.
        Parameters:
            block: Lattice
                The building block of the lattice of a DMRG.
            vector: 1d ndarray
                The translation vector of the left blocks and right blocks before the addition of two new ones.
            scopes: list of hashable objects
                The scopes of the blocks to be added two-by-two into the lattice of the DMRG.
            targets: sequence of QuantumNumber
                The target space at each growth of the DMRG.
            nmax: integer, optional
                The maximum number of singular values to be kept.
            tol: float64, optional
                The tolerance of the singular values.
        '''
        assert len(scopes)==len(targets)*2
        self.block=block
        self.vector=vector
        self.scopes=scopes
        self.targets=targets
        self.nmax=nmax
        self.tol=tol

    @property
    def lattices(self):
        '''
        Return a generator over the sequence of the SuperLattices of the DMRG with blocks added two-by-two into the center.
        '''
        scopes=copy(self.scopes)
        assert len(scopes)%2==0
        for i in xrange(len(scopes)/2):
            A=scopes.pop(0)
            B=scopes.pop(-1)
            if i==0:
                aps=[Point(p.pid._replace(scope=A),rcoord=p.rcoord-self.vector/2,icoord=p.icoord) for p in self.block.values()]
                bps=[Point(p.pid._replace(scope=B),rcoord=p.rcoord+self.vector/2,icoord=p.icoord) for p in self.block.values()]
            else:
                aps=[Point(p.pid,rcoord=p.rcoord-self.vector,icoord=p.icoord) for p in aps]
                bps=[Point(p.pid,rcoord=p.rcoord+self.vector,icoord=p.icoord) for p in bps]
                aps.extend([Point(p.pid._replace(scope=A),rcoord=p.rcoord-self.vector/2,icoord=p.icoord) for p in self.block.values()])
                bps.extend([Point(p.pid._replace(scope=B),rcoord=p.rcoord+self.vector/2,icoord=p.icoord) for p in self.block.values()])
            yield SuperLattice.compose(
                name=                   self.block.name,
                points=                 aps+bps,
                vectors=                self.block.vectors,
                nneighbour=             self.block.nneighbour,
                max_coordinate_number=  self.block.max_coordinate_number
                )

def DMRGTSG(engine,app):
    '''
    This method iterativey update the DMRG by increasing its lattice in the center by 2 blocks at each iteration.
    '''
    engine.rundependence(app.status.name)
    for i,lattice in enumerate(app.lattices):
        engine.lattice=lattice
        engine.layer=engine.degfres.layers[0]
        engine.config.reset(pids=engine.lattice)
        QuantumNumberCollection.history.clear()
        engine.degfres.reset(leaves=engine.config.table(mask=engine.mask).keys())
        engine.set_generators_operators_optstrs()
        indices=engine.degfres.indices()
        AL=Label(identifier=indices[i],qnc=engine.degfres[indices[i]])
        BL=Label(identifier=indices[i+1],qnc=engine.degfres[indices[i+1]])
        assert engine.mps.cut==None or engine.mps.cut==engine.mps.nsite/2
        assert AL not in engine.mps.table and BL not in engine.mps.table
        target=app.targets[i]
        if engine.mps.mode=='QN':
            diff=QuantumNumberCollection([] if target==engine.target else [(target if engine.target is None else target-engine.target,1)])
        for m in engine.mps[engine.mps.nsite/2:]:
            L,R=m.labels[MPS.L],m.labels[MPS.R]
            m.labels[MPS.L]=L.replace(identifier=L.identifier+2,qnc=L.qnc.kron(diff,'+') if engine.mps.mode=='QN' else L.qnc)
            m.labels[MPS.R]=R.replace(identifier=0 if R.identifier==0 else R.identifier+2,qnc=R.qnc.kron(diff,'+') if engine.mps.mode=='QN' else R.qnc)
        alabels,blabels=[None]*3,[None]*3
        if engine.mps.cut is None:
            alabels[MPS.L]=Label(identifier=0,qnc=1 if target is None else QuantumNumberCollection([(target.zeros,1)]))
            blabels[MPS.R]=Label(identifier=0,qnc=1 if target is None else QuantumNumberCollection([(target,1)]))
            engine.mps.cut=0
        else:
            alabels[MPS.L]=deepcopy(engine.mps[engine.mps.cut-1].labels[MPS.R])
            blabels[MPS.R]=deepcopy(engine.mps[engine.mps.cut].labels[MPS.L])
        alabels[MPS.S],blabels[MPS.S]=AL,BL
        alabels[MPS.R]=Label(identifier=engine.mps.cut+1,qnc=None)
        blabels[MPS.L]=Label(identifier=engine.mps.cut+1,qnc=None)
        engine.mps.insert(engine.mps.cut,Tensor([[[0.0]]],labels=alabels))
        engine.mps.insert(engine.mps.cut+1,Tensor([[[0.0]]],labels=blabels))
        engine.mps.cut+=1
        engine.set_blocks_and_connections()
        engine._Hs_["L"].extend([None,None])
        engine._Hs_["S"].extend([None,None])
        engine._Hs_["R"].extend([None,None])
        engine.target=target
        print '%s(++)\n%s'%(engine.status,engine.graph)
        engine.two_site_update()
        engine.two_site_truncate(nmax=app.nmax,tol=app.tol)
        engine.logger.record()
        print engine.logger
        print 'nnz,nbasis,err: %s,%s,%s.'%(engine.info['nnz'],engine.info['nbasis'],engine.info['err'])
        print 'gse: %s.'%(engine.info['gse'][0])
        print
    if app.save_data:
        with open('%s/%s_mps.dat'%(engine.din,engine.status),'wb') as fout:
            pk.dump(engine.mps,fout,2)

class TSS(App):
    '''
    Two site sweep of a DMRG.
    Attribues:
        BS: BaseSpace
            The basespace of the DMRG's parametes for the sweeps.
        nmaxs: list of integer
            The maximum numbers of singular values to be kept for the sweeps.
        tol: np.float64
            The tolerance of the singular values.
    '''

    def __init__(self,BS,nmaxs,tol=5*10**-14,**karg):
        '''
        Constructor.
        Parameters:
            BS: list of dict
                The parameters of the DMRG for each sweep.
            nmaxs: list of integer
                The maximum numbers of singular values to be kept for each sweep.
            tol: np.float64
                The tolerance of the singular values.
        '''
        self.BS=BS
        self.nmaxs=nmaxs
        self.tol=tol

def DMRGTSS(engine,app):
    '''
    This method iterativey sweep the DMRG with 2 sites updated at each iteration.
    '''
    def two_site_sweep(direction,info,nmax,tol):
        if direction=='<<':
            engine.mps<<=1
        else:
            engine.mps>>=1
        print '%s\n%s'%(info,engine.graph)
        A,Asite,sys=engine.A,engine.Asite,engine.sys
        B,Bsite,env=engine.B,engine.Bsite,engine.env
        ml=np.asarray(engine.mps[sys.pos]).reshape((A.nbasis*Asite.nbasis,sys.nbasis))
        mr=np.asarray(engine.mps[env.pos]).reshape((env.nbasis,Bsite.nbasis*B.nbasis))
        engine.two_site_update()
        if engine.mps.mode=='QN':
            ml=sys.qnc.reorder(ml,axes=[0])
            mr=env.qnc.reorder(mr,axes=[1])
        v0=np.einsum('ik,k,kj->ij',ml,np.asarray(engine.mps.Lambda),mr).ravel()
        if engine.mps.mode=='QN': v0=v0[engine.cache['subslice']]
        v0/=np.linalg.norm(v0)
        engine.two_site_truncate(v0=v0,nmax=nmax,tol=tol)
        engine.logger.record()
        print engine.logger
        print 'nnz,nbasis,overlap,err: %s,%s,%s,%s.'%(engine.info['nnz'],engine.info['nbasis'],engine.info['overlap'],engine.info['err'])
        print 'gse: %s.'%(engine.info['gse'][0])
        print
    for i,(parameters,nmax) in enumerate(zip(app.BS,app.nmaxs)):
        app.status.update(alter=parameters)
        cmp=app.status<=engine.status
        if not cmp: engine.update(parameters)
        suffix='st'if i==0 else ('nd' if i==1 else ('rd' if i==2 else 'th'))
        while engine.mps.cut>1:
            two_site_sweep(info='%s %s%s sweep(<<)'%(engine.status,i+1,suffix),direction='<<',nmax=nmax,tol=app.tol)
        while engine.mps.cut<engine.mps.nsite-1:
            two_site_sweep(info='%s %s%s sweep(>>)'%(engine.status,i+1,suffix),direction='>>',nmax=nmax,tol=app.tol)
    if app.save_data:
        with open('%s/%s_mps.dat'%(engine.din,engine.status),'wb') as fout:
            pk.dump(engine.mps,fout,2)
