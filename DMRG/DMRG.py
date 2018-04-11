'''
====================================
Density matrix renormalization group
====================================

DMRG, including:
    * classes: DMRG, TSG, TSS
    * function: pattern, DMRGTSG, DMRGTSS
'''

__all__=['pattern','DMRG','TSG','DMRGTSG','TSS','DMRGTSS']

import os
import re
import numpy as np
import pickle as pk
import itertools as it
import HamiltonianPy.Misc as hm
import matplotlib.pyplot as plt
from numpy.linalg import norm
from HamiltonianPy import *
from HamiltonianPy.TensorNetwork import *
from copy import deepcopy
from collections import OrderedDict

class Block(object):
    '''
    A block of DMRG tensor network.

    Attributes
    ----------
    mpo : MPO
        The MPO of the block.
    mps : MPS
        The MPS of the block.
    _Hs_ : dict
        * entry 'L': list of 3d Tensor
            The contraction of mpo and mps from the left.
        * entry 'R': list of 3d Tensor
            The contraction of mpo and mps from the right.
    '''

    def __init__(self,mpo,mps):
        '''
        Constructor.

        Parameters
        ----------
        mpo : MPO
            The MPO of the block.
        mps : MPS
            The MPS of the block.
        '''
        self.mpo=mpo
        self.mps=mps

    def relabel(self,sites,obonds,sbonds):
        '''

        '''
        pass


    def set_HL_(self,pos,job='contract',tol=hm.TOL):
        '''
        Set a certain left contraction.

        Parameters
        ----------
        pos : int
            The position of the left block Hamiltonian.
        job : 'contract' or 'relabel'
            'contract' for a complete calculation from the contraction of `self.mpo` and `self.mps`;
            'relabel' for a sole labels change according to `self.mpo` and `self.mps`.
        tol : np.float64, optional
            The tolerance of the non-zeros.
        '''
        assert job in ('contract','relabel')
        if job=='contract':
            if pos==-1:
                self._Hs_['L'][0]=Tensor([[[1.0]]],labels=[self.mps[+0].labels[MPS.L].prime,self.mpo[+0].labels[MPO.L],self.mps[+0].labels[MPS.L]])
            else:
                u,m=self.mps[pos],self.mpo[pos]
                L,S,R=u.labels
                up=u.copy(copy_data=False).conjugate()
                up.relabel(news=[L.prime,S.prime,R.prime])
                temp=contract([self._Hs_['L'][pos],up,m,u],engine='tensordot')
                temp[np.abs(temp)<tol]=0.0
                self._Hs_['L'][pos+1]=temp
        else:
            if pos==-1:
                self._Hs_['L'][0].relabel(news=[self.mps[+0].labels[MPS.L].prime,self.mpo[+0].labels[MPO.L],self.mps[+0].labels[MPS.L]])
            else:
                self._Hs_['L'][pos+1].relabel(news=[self.mps[pos].labels[MPS.R].prime,self.mpo[pos].labels[MPO.R],self.mps[pos].labels[MPS.R]])

    def set_HR_(self,pos,job='contract',tol=hm.TOL):
        '''
        Set a certain right block Hamiltonian.

        Parameters
        ----------
        pos : integer
            The position of the right block Hamiltonian.
        job : 'contract' or 'relabel'
            'contract' for a complete calculation from the contraction of `self.mpo` and `self.mps`;
            'relabel' for a sole labels change according to `self.mpo` and `self.mps`.
        tol : np.float64, optional
            The tolerance of the non-zeros.
        '''
        assert job in ('contract','relabel')
        if job=='contract':
            if pos==self.mps.nsite:
                self._Hs_['R'][0]=Tensor([[[1.0]]],labels=[self.mps[-1].labels[MPS.R].prime,self.mpo[-1].labels[MPO.R],self.mps[-1].labels[MPS.R]])
            else:
                v,m=self.mps[pos],self.mpo[pos]
                L,S,R=v.labels
                vp=v.copy(copy_data=False).conjugate()
                vp.relabel(news=[L.prime,S.prime,R.prime])
                temp=contract([self._Hs_['R'][self.mps.nsite-pos-1],vp,m,v],engine='tensordot')
                temp[np.abs(temp)<tol]=0.0
                self._Hs_['R'][self.mps.nsite-pos]=temp
        else:
            if pos==self.mps.nsite:
                self._Hs_['R'][0].relabel(news=[self.mps[-1].labels[MPS.R].prime,self.mpo[-1].labels[MPO.R],self.mps[-1].labels[MPS.R]])
            else:
                self._Hs_['R'][self.mps.nsite-pos].relabel(news=[self.mps[pos].labels[MPS.L].prime,self.mpo[pos].labels[MPO.L],self.mps[pos].labels[MPS.L]])

    def set_Hs_(self,SL=None,EL=None,SR=None,ER=None,keep=False,tol=hm.TOL):
        '''
        Set the Hamiltonians of blocks.

        Parameters
        ----------
        SL,EL : integer, optional
            The start/end position of the left Hamiltonians to be set.
        SR,ER : integer, optional
            The start/end position of the right Hamiltonians to be set.
        keep : logical, optional
            True for keeping the old `_Hs_` and False for not.
        tol : np.float64, optional
            The tolerance of the zeros.
        '''
        SL=-1 if SL is None else SL
        SR=self.mps.nsite if SR is None else SR
        if keep:
            for pos in xrange(-1,SL):
                self.set_HL_(pos,job='relabel')
            for pos in xrange(self.mps.nsite,SR,-1):
                self.set_HR_(pos,job='relabel')
        else:
            self._Hs_={'L':[None]*(self.mps.nsite+1),'R':[None]*(self.mps.nsite+1)}
        if self.mps.cut is not None:
            EL=self.mps.cut-1 if EL is None else EL
            ER=self.mps.cut if ER is None else ER
            for pos in xrange(SL,EL+1):
                self.set_HL_(pos,job='contract',tol=tol)
            for pos in xrange(SR,ER-1,-1):
                self.set_HR_(pos,job='contract',tol=tol)

def pattern(name,parameters,target,nsite,mode='re'):
    '''
    Return the pattern of data files for match.

    Parameters
    ----------
    name : str
        The name of the DMRG.
    parameters : Parameters
        The parameters of the DMRG.
    target : QuantumNumber
        The target of the DMRG.
    nsite : integer
        The number of sites of the DMRG.
    mode : 're','py'
        're' for regular and 'py' for python.

    Returns
    -------
    string
        The pattern.
    '''
    assert mode in ('re','py')
    result='%s_%s_DMRG_%s_%s'%(name,parameters,tuple(target) if isinstance(target,QuantumNumber) else None,nsite)
    if mode=='re':
        ss=['(',')','[',']','^','+']
        rs=['\(','\)','\[','\]','\^','\+']
        for s,r in zip(ss,rs):
            result=result.replace(s,r)
    return result

class DMRG(Engine):
    '''
    Density matrix renormalization group method.

    Attributes
    ----------
    mps : MPS
        The matrix product state of the DMRG.
    lattice : Cylinder, Lattice
        The lattice of the DMRG.
    terms : list of Term
        The terms of the DMRG.
    config : IDFConfig
        The configuration of the internal degrees of freedom on the lattice.
    degfres : DegFreTree
        The physical degrees of freedom tree.
    matvec : 'csr' or 'lo'
        The matrix-vector multiplication method. 'csr' for csr-matrix and 'lo' for linear operator.
    mask : [] or ['nambu']
        [] for spin systems and ['nambu'] for fermionic systems.
    target : QuantumNumber
        The target space of the DMRG.
    dtype : np.float64, np.complex128
        The data type.
    generator : Generator
        The generator of the Hamiltonian.
    operators : Operators
        The operators of the Hamiltonian.
    mpo : MPO
        The MPO-formed Hamiltonian.

    timers : Timers
        The timers of the dmrg processes.
    info : Sheet
        The info of the dmrg processes.
    cache : dict
        * entry 'osvs': 1d ndarray
            The old singular values of the DMRG.
    '''

    def __init__(self,mps,lattice,terms,config,degfres,matvec='lo',mask=(),target=None,dtype=np.complex128,**karg):
        '''
        Constructor.

        Parameters
        ----------
        mps : MPS
            The matrix product state of the DMRG.
        lattice : Cylinder
            The lattice of the DMRG.
        terms : list of Term
            The terms of the DMRG.
        config : IDFConfig
            The configuration of the internal degrees of freedom on the lattice.
        degfres : DegFreTree
            The physical degrees of freedom tree.
        matvec : 'csr' or 'lo'
            The matrix-vector multiplication method. 'csr' for csr-matrix and 'lo' for linear operator.
        mask : [] or ['nambu']
            [] for spin systems and ['nambu'] for fermionic systems.
        target : QuantumNumber
            The target space of the DMRG.
        dtype : np.float64,np.complex128, optional
            The data type.
        '''
        assert config.priority==degfres.priority
        assert matvec.lower() in ('csr','lo')
        self.mps=mps
        self.lattice=lattice
        self.terms=terms
        self.config=config
        self.degfres=degfres
        self.matvec=matvec.lower()
        self.mask=mask
        self.target=target
        self.dtype=dtype
        self.generator=Generator(bonds=lattice.bonds,config=config,terms=terms,dtype=dtype,half=False)
        if self.map is None: self.parameters.update(OrderedDict((term.id,term.value) for term in terms))
        self.set_operators()
        self.set_mpo()
        self.set_Hs_()
        if self.matvec=='csr':
            self.timers=Timers('Preparation','Hamiltonian','Diagonalization','Truncation')
            self.timers.add(parent='Hamiltonian',name='kron')
            self.timers.add(parent='Hamiltonian',name='sum')
            if self.mps.mode=='QN':
                self.timers.add(parent='kron',name='csr')
                self.timers.add(parent='kron',name='fkron')
            self.info=Sheet(('Etotal','Esite','dE/E','nmatvec','nbasis','nslice','nnz','nz','density','overlap','err'))
        else:
            self.timers=Timers('Preparation','Diagonalization','Truncation')
            self.timers.add('Diagonalization','matvec')
            self.info=Sheet(('Etotal','Esite','dE/E','nmatvec','nbasis','nslice','overlap','err'))
        self.cache={}
        self.logging()

    @property
    def graph(self):
        '''
        The graph representation of the DMRG.
        '''
        sites=''.join([' ','A-'*(self.mps.cut-1),'.-.','-B'*(self.mps.nsite-self.mps.cut-1)])
        bonds=''.join(str(bond.dim)+('=' if i in (self.mps.cut-1,self.mps.cut) else ('-' if i<self.mps.nsite else '')) for i,bond in enumerate(self.mpo.bonds))
        return '\n'.join([sites,bonds])

    @property
    def state(self):
        '''
        The current state of the dmrg.
        '''
        return '%s(target=%s,nsite=%s,cut=%s)'%(self,self.target,self.mps.nsite,self.mps.cut)

    def update(self,**karg):
        '''
        Update the DMRG with new parameters.
        '''
        if len(karg)>0:
            super(DMRG,self).update(**karg)
            self.generator.update(**self.data)
            self.set_operators()
            self.set_mpo()
            self.set_Hs_()

    def set_operators(self):
        '''
        Set the operators of the DMRG.
        '''
        self.operators=self.generator.operators

    def set_mpo(self):
        '''
        Set the mpo of the DMRG.
        '''
        if len(self.operators)>0:
            self.mpo=OptMPO([OptStr.from_operator(operator,self.degfres) for operator in self.operators.itervalues()],self.degfres).to_mpo()

    @property
    def nspb(self):
        '''
        The number of site labels per block.
        '''
        config,degfres=deepcopy(self.config),deepcopy(self.degfres)
        config.reset(pids=self.lattice(['__DMRG_NSPB__']).pids)
        degfres.reset(leaves=config.table(mask=self.mask).keys())
        return len(degfres.indices())

    def reset(self,lattice,mps):
        '''
        Reset the core of the dmrg.

        Parameters
        ----------
        lattice : Lattice
            The new lattice of the dmrg.
        mps : MPS
            The new mps of the dmrg.
        '''
        self.lattice=lattice
        self.config.reset(pids=self.lattice.pids)
        self.degfres.reset(leaves=self.config.table(mask=self.mask).keys())
        self.generator.reset(bonds=self.lattice.bonds,config=self.config)
        self.set_operators()
        self.set_mpo()
        self.mps=mps
        sites,bonds=self.degfres.labels('S'),self.degfres.labels('B')
        self.mps.relabel(sites=sites,bonds=[bond.replace(qns=obond.qns) for bond,obond in zip(bonds,mps.bonds)])
        self.target=next(iter(mps[-1].labels[MPS.R].qns)) if mps.mode=='QN' else None
        self.set_Hs_()

    def iterate(self,info='',sp=True,nmax=200,tol=hm.TOL,piechart=True):
        '''
        The two site dmrg step.

        Parameters
        ----------
        info : str, optional
            The information string passed to self.log.
        sp : logical, optional
            True for state prediction False for not.
        nmax : integer, optional
            The maximum singular values to be kept.
        tol : np.float64, optional
            The tolerance of the singular values.
        piechart : logical, optional
            True for showing the piechart of self.timers while False for not.
        '''
        self.log<<'%s%s\n%s\n'%(self.state,info,self.graph)
        eold=self.info['Esite']
        with self.timers.get('Preparation'):
            Ha,Hb=self._Hs_['L'][self.mps.cut-1],self._Hs_['R'][self.mps.nsite-self.mps.cut-1]
            Hasite,Hbsite=self.mpo[self.mps.cut-1],self.mpo[self.mps.cut]
            La,Sa,Ra=self.mps[self.mps.cut-1].labels
            Lb,Sb,Rb=self.mps[self.mps.cut].labels
            Oa,Ob=Hasite.labels[MPO.R],Hbsite.labels[MPO.L]
            assert Ra==Lb
            if self.mps.mode=='QN':
                sysqns,syspt=QuantumNumbers.kron([La.qns,Sa.qns],signs='++').sort(history=True)
                envqns,envpt=QuantumNumbers.kron([Sb.qns,Rb.qns],signs='-+').sort(history=True)
                sysantipt,envantipt=np.argsort(syspt),np.argsort(envpt)
                subslice=QuantumNumbers.kron([sysqns,envqns],signs='+-').subslice(targets=(self.target.zero(),))
                qns=QuantumNumbers.mono(self.target.zero(),count=len(subslice))
                self.info['nslice']=len(subslice)
            else:
                sysqns,syspt,sysantipt=np.product([La.qns,Sa.qns]),None,None
                envqns,envpt,envantipt=np.product([Sb.qns,Rb.qns]),None,None
                subslice,qns=slice(None),sysqns*envqns
                self.info['nslice']=qns
            Lpa,Spa,Spb,Rpb=La.prime,Sa.prime,Sb.prime,Rb.prime
            Lsys,Lenv,new=Label('__DMRG_TWO_SITE_STEP_SYS__',qns=sysqns),Label('__DMRG_TWO_SITE_STEP_ENV__',qns=envqns),Ra.replace(qns=None)
            Lpsys,Lpenv=Lsys.prime,Lenv.prime
            Hsys=contract([Ha,Hasite],engine='tensordot').transpose([Oa,Lpa,Spa,La,Sa]).merge(([Lpa,Spa],Lpsys,syspt),([La,Sa],Lsys,syspt))
            Henv=contract([Hbsite,Hb],engine='tensordot').transpose([Ob,Spb,Rpb,Sb,Rb]).merge(([Spb,Rpb],Lpenv,envpt),([Sb,Rb],Lenv,envpt))
        if self.matvec=='csr':
            with self.timers.get('Hamiltonian'):
                if isinstance(subslice,slice):
                    rcs=None
                else:
                    rcs=(np.divide(subslice,Henv.shape[1]),np.mod(subslice,Henv.shape[1]),np.zeros(Hsys.shape[1]*Henv.shape[1],dtype=np.int64))
                    rcs[2][subslice]=xrange(len(subslice))
                matrix=0
                for hsys,henv in zip(Hsys,Henv):
                    with self.timers.get('kron'):
                        temp=hm.kron(hsys,henv,rcs=rcs,timers=self.timers)
                    with self.timers.get('sum'):
                        matrix+=temp
                self.info['nnz']=matrix.nnz
                self.info['nz']=(len(np.argwhere(np.abs(matrix.data)<tol))*100.0/matrix.nnz) if matrix.nnz>0 else 0,'%1.1f%%'
                self.info['density']=1.0*self.info['nnz']/self.info['nslice']**2,'%.1e'
                matrix=hm.LinearOperator(shape=matrix.shape,matvec=matrix.dot,dtype=self.dtype)
        else:
            with self.timers.get('Preparation'):
                if self.mps.mode=='QN':
                    sysod,envod=sysqns.to_ordereddict(),envqns.to_ordereddict()
                    qnpairs=[[(tuple(qn),tuple(qn-oqn)) for qn in sysqns if tuple(qn) in envod and tuple(qn-oqn) in sysod and tuple(qn-oqn) in envod] for oqn in Oa.qns]
                    assert len(qnpairs)==len(Hsys)
                    self.cache['vecold']=np.zeros(Hsys.shape[1]*Henv.shape[1],dtype=self.dtype)
                    self.cache['vecnew']=np.zeros((Hsys.shape[1],Henv.shape[1]),dtype=self.dtype)
                    def matvec(v):
                        with self.timers.get('matvec'):
                            vec,result=self.cache['vecold'],self.cache['vecnew']
                            vec[subslice]=v;result[...]=0.0
                            vec=vec.reshape((Hsys.shape[1],Henv.shape[1]))
                            for hsys,henv,pairs in zip(Hsys,Henv,qnpairs):
                                for qn1,qn2 in pairs:
                                    result[sysod[qn1],envod[qn1]]+=hsys[sysod[qn1],sysod[qn2]].dot(vec[sysod[qn2],envod[qn2]]).dot(henv.T[envod[qn2],envod[qn1]])
                        return result.reshape(-1)[subslice] 
                    matrix=hm.LinearOperator(shape=(len(subslice),len(subslice)),matvec=matvec,dtype=self.dtype)
                else:
                    def matvec(v):
                        v=v.reshape((Hsys.shape[1],Henv.shape[1]))
                        result=np.zeros_like(v)
                        for hsys,henv in zip(Hsys,Henv):
                            result+=hsys.dot(v).dot(henv.T)
                        return result.reshape(-1)
                    matrix=hm.LinearOperator(shape=(Hsys.shape[1]*Henv.shape[1],Hsys.shape[1]*Henv.shape[1]),matvec=matvec,dtype=self.dtype)
        with self.timers.get('Diagonalization'):
            u,s,v=self.mps[self.mps.cut-1],self.mps.Lambda,self.mps[self.mps.cut]
            if sp and norm(s)>RZERO:
                v0=np.asarray(contract([u,s,v],engine='einsum').merge(([La,Sa],Lsys,syspt),([Sb,Rb],Lenv,envpt))).reshape(-1)[subslice]
            else:
                v0=None
            es,vs=hm.eigsh(matrix,which='SA',v0=v0,k=1)
            energy,Psi=es[0],vs[:,0]
            self.info['Etotal']=energy,'%.6f'
            self.info['Esite']=energy/self.mps.nsite,'%.8f'
            self.info['dE/E']=None if eold is None else (norm(self.info['Esite']-eold)/norm(self.info['Esite']+eold),'%.1e')
            self.info['nmatvec']=matrix.count
            self.info['overlap']=np.inf if v0 is None else np.abs(Psi.conjugate().dot(v0)/norm(v0)/norm(Psi)),'%.6f'
        with self.timers.get('Truncation'):
            u,s,v,err=Tensor(Psi,labels=[Label('__DMRG_TWO_SITE_STEP__',qns=qns)]).partitioned_svd(Lsys,new,Lenv,nmax=nmax,tol=tol,return_truncation_err=True)
            self.mps[self.mps.cut-1]=u.split((Lsys,[La,Sa],sysantipt))
            self.mps[self.mps.cut]=v.split((Lenv,[Sb,Rb],envantipt))
            self.mps.Lambda=s
            self.set_HL_(self.mps.cut-1,tol=tol)
            self.set_HR_(self.mps.cut,tol=tol)
            self.info['nbasis']=len(s)
            self.info['err']=err,'%.1e'
        self.timers.record()
        self.log<<'timers of the dmrg:\n%s\n'%self.timers.tostr(Timers.ALL)
        self.log<<'info of the dmrg:\n%s\n\n'%self.info
        if piechart: self.timers.graph(parents=Timers.ALL)

    def insert(self,A,B,news=None,target=None):
        '''
        Insert two blocks of points into the center of the lattice.

        Parameters
        ----------
        A,B : any hashable object
            The scopes of the insert block points.
        news : list of any hashable object, optional
            The new scopes of the original points before the insertion.
        target : QuantumNumber, optional
            The new target of the DMRG.
        '''
        self.lattice.insert(A,B,news=news)
        self.config.reset(pids=self.lattice.pids)
        self.degfres.reset(leaves=self.config.table(mask=self.mask).keys())
        self.generator.reset(bonds=self.lattice.bonds,config=self.config)
        self.set_operators()
        sites,mpsbonds,mpobonds=self.degfres.labels('S'),self.degfres.labels('B'),self.degfres.labels('O')
        niter,ob,nb=len(self.lattice)/len(self.lattice.block)/2,self.mps.nsite/2+1,(len(mpsbonds)+1)/2
        if niter>self.lattice.nneighbour+2:
            DMRG.impo_generate(self.mpo,sites,mpobonds)
        else:
            self.set_mpo()
        if niter>1 or nb-ob==1:
            self.cache['osvs']=DMRG.imps_predict(self.mps,sites,mpsbonds,self.cache.get('osvs',np.array([1.0])),target=target,dtype=self.dtype)
            self._Hs_['L'].extend([None]*((nb-ob)*2))
            self._Hs_['R'].extend([None]*((nb-ob)*2))
            SL,SR,keep=(ob-1,self.mps.nsite-ob,True) if niter>self.lattice.nneighbour+2 else (None,None,False)
            self.set_Hs_(SL=SL,EL=nb-3,SR=SR,ER=nb,keep=keep)
        else:
            assert self.mps.nsite==0
            mpsbonds[+0]=mpsbonds[+0].replace(qns=QuantumNumbers.mono(target.zero()) if isinstance(target,QuantumNumber) else 1)
            mpsbonds[-1]=mpsbonds[-1].replace(qns=QuantumNumbers.mono(target) if isinstance(target,QuantumNumber) else 1)
            self.mps=MPS.random(sites=sites,bonds=mpsbonds,cut=len(sites)/2,nmax=20,dtype=self.dtype)
            self.set_Hs_(EL=self.mps.cut-2,ER=self.mps.cut+1)
        self.target=target

    def sweep(self,info='',path=None,**karg):
        '''
        Perform a sweep over the mps of the dmrg under the guidance of `path`.

        Parameters
        ----------
        info : str, optional
            The info passed to self.log.
        path : list of str, optional
            The path along which the sweep is performed.
        karg : 'nmax','tol','piechart'
            See DMRG.iterate for details.
        '''
        for move in it.chain(['<<']*(self.mps.cut-1),['>>']*(self.mps.nsite-2),['<<']*(self.mps.nsite-self.mps.cut-1)) if path is None else path:
            self.mps<<=1 if '<<' in move else -1
            self.iterate(info='%s(%s)'%(info,move),sp=True,**karg)

    def coredump(self):
        '''
        Use pickle to dump the core of the dmrg.
        '''
        with open('%s/%s_%s.dat'%(self.din,pattern(self.name,self.parameters,self.target,self.mps.nsite,mode='py'),self.mps.nmax),'wb') as fout:
            core=   {
                        'lattice':      self.lattice,
                        'target':       self.target,
                        'operators':    self.operators,
                        'mpo':          self.mpo,
                        'mps':          self.mps,
                        '_Hs_':         self._Hs_,
                        'info':         self.info,
                        'cache':        self.cache
                        }
            pk.dump(core,fout,2)

    @staticmethod
    def coreload(din,pattern,nmax):
        '''
        Use pickle to load the core of the dmrg from existing data files.

        Parameters
        ----------
        din : string
            The directory where the data files are searched.
        pattern : string
            The matching pattern of the data files.
        nmax : integer
            The maximum number of singular values kept in the mps.

        Returns
        -------
        dict
            The loaded core of the dmrg.
        '''
        candidates={}
        names=[name for name in os.listdir(din) if re.match(pattern,name)]
        for name in names:
            split=name.split('_')
            cnmax=int(split[-1][0:-4])
            if cnmax<=nmax:candidates[name]=cnmax
        if len(candidates)>0:
            with open('%s/%s'%(din,sorted(candidates.keys(),key=candidates.get)[-1]),'rb') as fin:
                result=pk.load(fin)
        else:
            result=None
        return result

class TSG(App):
    '''
    Two site growth of a DMRG.

    Attributes
    ----------
    targets : sequence of QuantumNumber
        The target space at each growth of the DMRG.
    nmax : integer
        The maximum singular values to be kept.
    npresweep : integer
        The number of presweeps to make a random mps converged to the target state.
    nsweep : integer
        The number of sweeps to make the predicted mps converged to the target state.
    tol : float64
        The tolerance of the target state energy.
    '''

    def __init__(self,targets,nmax,npresweep=10,nsweep=4,tol=10**-6,**karg):
        '''
        Constructor.

        Parameters
        ----------
        targets : sequence of QuantumNumber
            The target space at each growth of the DMRG.
        nmax : integer
            The maximum number of singular values to be kept.
        npresweep : integer, optional
            The number of presweeps to make a random mps converged to the target state.
        nsweep : integer, optional
            The number of sweeps to make the predicted mps converged to the target state.
        tol : float64, optional
            The tolerance of the target state energy.
        '''
        self.targets=targets
        self.nmax=nmax
        self.npresweep=npresweep
        self.nsweep=nsweep
        self.tol=tol

    def recover(self,engine):
        '''
        Recover the core of a dmrg engine.

        Parameters
        ----------
        engine : DMRG
            The dmrg engine whose core is to be recovered.

        Returns
        -------
        integer
            The recover code.
        '''
        for i,target in enumerate(reversed(self.targets)):
            core=DMRG.coreload(din=engine.din,pattern=pattern(engine.name,engine.parameters,target,(len(self.targets)-i)*engine.nspb*2,mode='re'),nmax=self.nmax)
            if core:
                for key,value in core.iteritems():
                    setattr(engine,key,value)
                engine.config.reset(pids=engine.lattice.pids)
                engine.degfres.reset(leaves=engine.config.table(mask=engine.mask).keys())
                engine.generator.reset(bonds=engine.lattice.bonds,config=engine.config)
                code=len(self.targets)-i-1
                break
        else:
            code=-1
        return code

def DMRGTSG(engine,app):
    '''
    This method iterative update the DMRG by increasing its lattice in the center by 2 blocks at each iteration.
    '''
    engine.log.open()
    num=app.recover(engine)
    scopes,nspb=range(len(app.targets)*2),engine.nspb
    def TSGSWEEP(nsweep):
        assert engine.mps.cut==engine.mps.nsite/2
        nold,nnew=engine.mps.nsite-2*nspb,engine.mps.nsite
        path=list(it.chain(['++<<']*((nnew-nold-2)/2),['++>>']*(nnew-nold-2),['++<<']*((nnew-nold-2)/2)))
        for sweep in xrange(nsweep):
            seold=engine.info['Esite']
            engine.sweep(info=' No.%s'%(sweep+1),path=path,nmax=app.nmax,piechart=app.plot)
            senew=engine.info['Esite']
            if norm(seold-senew)/norm(seold+senew)<app.tol: break
    for i,target in enumerate(app.targets[num+1:]):
        pos=i+num+1
        engine.insert(scopes[pos],scopes[-pos-1],news=scopes[:pos]+scopes[-pos:] if pos>0 else None,target=target)
        engine.iterate(info='(++)',sp=True if pos>0 else False,nmax=app.nmax,piechart=app.plot)
        TSGSWEEP(app.npresweep if pos==0 else app.nsweep)
        if nspb>1 and len(app.targets)>1 and pos==0 and app.savedata: engine.coredump()
    if num==len(app.targets)-1 and app.nmax>engine.mps.nmax: TSGSWEEP(app.nsweep)
    if app.plot and app.savefig:
        plt.savefig('%s/%s_%s.png'%(engine.log.dir,engine,repr(engine.target)))
        plt.close()
    if app.savedata: engine.coredump()
    engine.log.close()

class TSS(App):
    '''
    Two site sweep of a DMRG.

    Attributes
    ----------
    target : QuantumNumber
        The target of the DMRG's mps.
    nsite : integer
        The length of the DMRG's mps.
    nmaxs : list of integer
        The maximum numbers of singular values to be kept for the sweeps.
    BS : BaseSpace
        The basespace of the DMRG's parameters for the sweeps.
    paths : list of list of '<<' or '>>'
        The paths along which the sweeps are performed.
    force_sweep : logical
        When True, the sweep will be taken at least once even if the mps are recovered from existing data files perfectly.
        When False, no real sweep will be taken if the mps can be perfectly recovered from existing data files.
    '''

    def __init__(self,target,nsite,nmaxs,BS=None,paths=None,force_sweep=False,**karg):
        '''
        Constructor.

        Parameters
        ----------
        target : QuantumNumber
            The target of the DMRG's mps.
        nsite : integer
            The length of the DMRG's mps.
        nmaxs : list of integer
            The maximum numbers of singular values to be kept for each sweep.
        BS : list of dict, optional
            The parameters of the DMRG for each sweep.
        paths : list of list of '<<' or '>>', optional
            The paths along which the sweeps are performed.
        force_sweep : logical, optional
            When True, the sweep will be taken at least once even if the mps are recovered from existing data files perfectly.
            When False, no real sweep will be taken if the mps can be perfectly recovered from existing data files.
        '''
        self.target=target
        self.nsite=nsite
        self.nmaxs=nmaxs
        self.BS=[{}]*len(nmaxs) if BS is None else BS
        self.paths=[None]*len(nmaxs) if paths is None else paths
        assert len(nmaxs)==len(self.BS) and len(nmaxs)==len(self.paths)
        self.force_sweep=force_sweep

    def recover(self,engine):
        '''
        Recover the core of a dmrg engine.

        Parameters
        ----------
        engine : DMRG
            The dmrg engine whose core is to be recovered.

        Returns
        -------
        integer
            The recover code.
        '''
        parameters=deepcopy(engine.parameters)
        for i,(nmax,paras) in enumerate(reversed(zip(self.nmaxs,self.BS))):
            parameters.update(paras)
            core=DMRG.coreload(din=engine.din,pattern=pattern(self.name,parameters,self.target,self.nsite,mode='re'),nmax=nmax)
            if core:
                for key,value in core.iteritems():
                    setattr(engine,key,value)
                engine.parameters=parameters
                engine.config.reset(pids=engine.lattice.pids)
                engine.degfres.reset(leaves=engine.config.table(mask=engine.mask).keys())
                engine.mps>>=1 if engine.mps.cut==0 else (-1 if engine.mps.cut==engine.mps.nsite else 0)
                code=len(self.nmaxs)-1-i
                if self.force_sweep or engine.mps.nmax<nmax: code-=1
                break
        else:
            code=None
        return code

def DMRGTSS(engine,app):
    '''
    This method iterative sweep the DMRG with 2 sites updated at each iteration.
    '''
    engine.log.open()
    num=app.recover(engine)
    if num is None:
        if app.name in engine.apps: engine.rundependences(app.name)
        num=-1
    for i,(nmax,parameters,path) in enumerate(zip(app.nmaxs[num+1:],app.BS[num+1:],app.paths[num+1:])):
        app.parameters.update(parameters)
        if not (app.parameters.match(engine.parameters)): engine.update(**parameters)
        engine.sweep(info=' No.%s'%(i+1),path=path,nmax=nmax,piechart=app.plot)
        if app.savedata: engine.coredump()
    if app.plot and app.savefig:
        plt.savefig('%s/%s_%s.png'%(engine.log.dir,engine,repr(engine.target)))
        plt.close()
    engine.log.close()
