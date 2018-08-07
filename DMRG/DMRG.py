'''
====================================
Density matrix renormalization group
====================================

DMRG, including:
    * classes: Block, DMRG, TSG, TSS, GSE
    * functions: DMRGMatVec
'''

__all__=['DMRGMatVec','Block','DMRG','TSG','TSS','GSE']

import os,re
import numpy as np
import pickle as pk
import itertools as it
import HamiltonianPy.Misc as hm
from numpy.linalg import norm
from HamiltonianPy import *
from HamiltonianPy.TensorNetwork import *
from copy import deepcopy
from collections import OrderedDict

def DMRGMatVec(Hsys,Henv):
    '''
    DMRG matrix-vector multiplication.

    Parameters
    ----------
    Hsys : 3d DTensor/STensor
        The system (left) Hamiltonian.
    Henv : 3d DTensor/STensor
        The environment (right) Hamiltonian.

    Returns
    -------
    callable
        The multiplication function of the whole Hamiltonian on a dense vector.
    '''
    Oa,Lsys=Hsys.labels[0],Hsys.labels[2]
    Ob,Lenv=Henv.labels[0],Henv.labels[2]
    assert Oa==Ob
    if Hsys.qnon:
        sysod=Lsys.qns.toordereddict(protocol=QuantumNumbers.INDPTR)
        envod=Lenv.qns.toordereddict(protocol=QuantumNumbers.INDPTR)
        records,count={},0
        for qn in it.ifilter(sysod.has_key,envod):
            sysslice,envslice=sysod[qn],envod[qn]
            inc=(sysslice.stop-sysslice.start)*(envslice.stop-envslice.start)
            records[qn]=slice(count,count+inc)
            count+=inc
        if isinstance(Hsys,STensor):
            def matvec(v):
                result=np.zeros(v.shape,dtype=v.dtype)
                for qns in it.ifilter(Henv.data.has_key,Hsys.data):
                    newslice,oldslice=records[qns[1]],records[qns[2]]
                    for sysblock,envblock in zip(Hsys.data[qns],Henv.data[qns]):
                        result[newslice]+=sysblock.dot(v[oldslice].reshape((sysblock.shape[1],envblock.shape[1]))).dot(envblock.T).reshape(-1)
                return result
        else:
            qnpairs=[[(sqn,sqn-oqn) for sqn in Lsys.qns if sqn in envod and sqn-oqn in sysod and sqn-oqn in envod] for oqn in Oa.qns]
            def matvec(v):
                result=np.zeros(v.shape,dtype=v.dtype)
                for hsys,henv,pairs in zip(Hsys.data,Henv.data,qnpairs):
                    for qn1,qn2 in pairs:
                        newslice,oldslice=records[qn1],records[qn2]
                        sysblock,envblock=hsys[sysod[qn1],sysod[qn2]],henv.T[envod[qn2],envod[qn1]]
                        result[newslice]+=sysblock.dot(v[oldslice].reshape((sysblock.shape[1],envblock.shape[0]))).dot(envblock).reshape(-1)
                return result
    else:
        def matvec(v):
            v=v.reshape((Hsys.shape[1],Henv.shape[1]))
            result=np.zeros_like(v)
            for hsys,henv in zip(Hsys.data,Henv.data):
                result+=hsys.dot(v).dot(henv.T)
            return result.reshape(-1)
    return matvec

class Block(object):
    '''
    A block of DMRG tensor network.

    Attributes
    ----------
    mpo : MPO
        The MPO of the block.
    mps : MPS
        The MPS of the block.
    target : QuantumNumber
        The target of the block.
    ttype : 'D'/'S'
        Tensor type. 'D' for dense and 'S' for sparse.
    lcontracts : list of 3d DTensor/STensor
        The contraction of mpo and mps from the left.
    rcontracts : list of 3d DTensor/STensor
        The contraction of mpo and mps from the right.
    timers : Timers
        The timers of the block.
    info : Sheet
        The info of the block.
    '''

    def __init__(self,mpo,mps,target=None,LEND=None,REND=None,ttype='D'):
        '''
        Constructor.

        Parameters
        ----------
        mpo : MPO
            The MPO of the block.
        mps : MPS
            The MPS of the block.
        target : QuantumNumber, optional
            The target of the block.
        LEND/REND : 3d DTensor/STensor, optional
            The leftmost/rightmost end of the contraction of mpo and mps.
        ttype : 'D'/'S', optional
            Tensor type. 'D' for dense and 'S' for sparse.
        '''
        self.reset(mpo,mps,target=target,LEND=LEND,REND=REND,ttype=ttype)
        self.timers=Timers('Preparation','Diagonalization','Truncation')
        self.timers.add('Diagonalization','matvec')
        self.info=Sheet(('Etotal','Esite','nmatvec','nbasis','nslice','overlap','err'))

    @property
    def nsite(self):
        '''
        The number of sites in the block.
        '''
        return self.mps.nsite

    @property
    def cut(self):
        '''
        The cut of the block.
        '''
        return self.mps.cut

    @property
    def graph(self):
        '''
        The graph representation of the block.
        '''
        sites=' %s.-.%s'%('A-'*(self.mps.cut-1),'-B'*(self.nsite-self.mps.cut-1))
        bonds=''.join(str(bond.dim)+('=' if i in (self.mps.cut-1,self.mps.cut) else ('-' if i<self.mps.nsite else '')) for i,bond in enumerate(self.mpo.bonds))
        return '\n'.join([sites,bonds])

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return 'Block(%s,%s,%s)'%(self.target,self.nsite,self.cut)

    __repr__=__str__

    def __getstate__(self):
        '''
        Used for pickle.
        '''
        return {    'mpo':          self.mpo,
                    'mps':          self.mps,
                    'target':       self.target,
                    'ttype':        self.ttype,
                    'lcontracts':   self.lcontracts,
                    'rcontracts':   self.rcontracts,
                    'info':self.info
                    }

    def __setstate__(self,state):
        '''
        Used for pickle.
        '''
        for key,value in state.iteritems(): setattr(self,key,value)
        self.timers=Timers('Preparation','Diagonalization','Truncation')
        self.timers.add('Diagonalization','matvec')

    def __getslice__(self,i,j):
        '''
        Operator "[]" for slicing.
        '''
        result=object.__new__(type(self))
        result.mpo=self.mpo[i:j]
        result.mps=self.mps[i:j]
        result.target=self.target
        result.ttype=self.ttype
        result.lcontracts=self.lcontracts[i:j+1]
        result.rcontracts=self.rcontracts[i:j+1]
        result.timers=self.timers
        result.info=self.info
        return result

    def reset(self,mpo=None,mps=None,target=None,LEND=None,REND=None,ttype=None):
        '''
        Constructor.

        Parameters
        ----------
        mpo,mps,target,LEND,REND,ttype :
            See Block.__init__ for details.
        '''
        if mpo is None: mpo=getattr(self,'mpo')
        if mps is None: mps=getattr(self,'mps')
        if target is None: target=getattr(self,'target')
        if ttype is None: ttype=getattr(self,'ttype')
        assert len(mpo)==len(mps)
        self.ttype=ttype
        self.mpo=mpo if mpo.ttype in (self.ttype,None) else mpo.todense() if self.ttype=='D' else mpo.tosparse()
        self.mps=mps if mps.ttype in (self.ttype,None) else mps.todense() if self.ttype=='D' else mps.tosparse()
        self.target=target
        if self.nsite>0:
            if LEND is None:
                C=self.mpo[0].labels[MPO.L].inverse
                D=self.mps[0].labels[MPS.L].inverse
                assert C.dim==1==D.dim
                LEND=Tensor([[[1.0]]],labels=[D.P,C,D],ttype=self.ttype)
            if REND is None:
                C=self.mpo[self.nsite-1].labels[MPO.R].inverse
                D=self.mps[self.nsite-1].labels[MPS.R].inverse
                assert C.dim==1==D.dim
                REND=Tensor([[[1.0]]],labels=[D.P,C,D],ttype=self.ttype)
            assert LEND.ndim==3==REND.ndim
            self.setcontractions(LEND=LEND,REND=REND)

    def relabel(self,sites,obonds,sbonds):
        '''
        Change the labels of the block.

        Parameters
        ----------
        sites : list of str/Label
            The new site identifiers/labels of the block.
        obonds : list of str/Label
            The new mpo bond identifiers/labels of the block.
        sbonds : list of str/Label
            The new mps bond identifiers/labels of the block.
        '''
        self.mpo.relabel(sites,obonds)
        self.mps.relabel(sites,sbonds)
        for pos in xrange(self.nsite+1):
            self.setlcontract(pos,job='relabel')
            self.setrcontract(pos,job='relabel')

    def setlcontract(self,pos,job='contract'):
        '''
        Set a certain left contraction.

        Parameters
        ----------
        pos : int
            The position of the left contraction.
        job : 'contract' or 'relabel'
            'contract' for a complete calculation from the contraction of `self.mpo` and `self.mps`;
            'relabel' for a sole labels change according to `self.mpo` and `self.mps`.
        '''
        if pos<0: pos+=self.nsite
        assert job in ('contract','relabel') and 0<=pos<=self.nsite
        if job=='contract':
            assert pos>=1
            self.lcontracts[pos]=self.lcontracts[pos-1]*self.mps[pos-1].dagger*self.mpo[pos-1]*self.mps[pos-1]
        else:
            C=self.mpo[pos].labels[MPO.L].inverse if pos<self.nsite else self.mpo[self.nsite-1].labels[MPO.R]
            D=self.mps[pos].labels[MPS.L].inverse if pos<self.nsite else self.mps[self.nsite-1].labels[MPS.R]
            self.lcontracts[pos].relabel([D.P,C,D])

    def setrcontract(self,pos,job='contract'):
        '''
        Set a certain right contraction.

        Parameters
        ----------
        pos : int
            The position of the right contraction.
        job : 'contract' or 'relabel'
            'contract' for a complete calculation from the contraction of `self.mpo` and `self.mps`;
            'relabel' for a sole labels change according to `self.mpo` and `self.mps`.
        '''
        if pos<0: pos+=self.nsite
        assert job in ('contract','relabel') and 0<=pos<=self.nsite
        if job=='contract':
            assert pos<self.nsite
            self.rcontracts[pos]=self.rcontracts[pos+1]*self.mps[pos].dagger*self.mpo[pos]*self.mps[pos]
        else:
            C=self.mpo[pos].labels[MPO.L] if pos<self.nsite else self.mpo[self.nsite-1].labels[MPO.R].inverse
            D=self.mps[pos].labels[MPS.L] if pos<self.nsite else self.mps[self.nsite-1].labels[MPS.R].inverse
            self.rcontracts[pos].relabel([D.P,C,D])

    def setcontractions(self,LEND=None,REND=None,SL=None,EL=None,SR=None,ER=None):
        '''
        Set the Hamiltonians of blocks.

        Parameters
        ----------
        LEND/REND : 3d DTensor/STensor, optional
            The leftmost/rightmost end of the contraction of mpo and mps.
        SL,EL : int, optional
            The start/end position of the left contractions to be set.
        SR,ER : int, optional
            The start/end position of the right contractions to be set.
        '''
        SL=1 if SL is None else SL
        SR=self.nsite-1 if SR is None else SR
        if LEND is None:
            for pos in xrange(SL):
                self.setlcontract(pos,job='relabel')
        else:
            self.lcontracts=[None]*(self.nsite+1)
            self.lcontracts[0]=LEND
            self.setlcontract(0,job='relabel')
        if REND is None:
            for pos in xrange(self.nsite,SR,-1):
                self.setrcontract(pos,job='relabel')
        else:
            self.rcontracts=[None]*(self.nsite+1)
            self.rcontracts[self.nsite]=REND
            self.setrcontract(self.nsite,job='relabel')
        if self.cut is not None:
            EL=self.cut if EL is None else EL
            ER=self.cut if ER is None else ER
            for pos in xrange(SL,EL+1):
                self.setlcontract(pos,job='contract')
            for pos in xrange(SR,ER-1,-1):
                self.setrcontract(pos,job='contract')

    def predict(self,sites,obonds,sbonds,osvs,qn=0):
        '''
        Infinite block prediction.

        Parameters
        ----------
        sites : list of str/Label
            The new site identifiers/labels of the block after prediction.
        obonds : list of str/Label
            The new mpo bond identifiers/labels of the block after prediction.
        sbonds : list of str/Label
            The new mps bond identifiers/labels of the block after prediction.
        osvs : 1d ndarray
            The old singular values.
        qn : QuantumNumber, optional
            The injected quantum number of the block.
        '''
        LEND=self.lcontracts[self.nsite/2]
        REND=self.rcontracts[self.nsite/2]
        self.mpo=self.mpo.impoprediction(sites,obonds)
        self.mps=self.mps.impsprediction(sites,sbonds,osvs,qn)
        self.target=qn+self.target if isinstance(qn,QuantumNumber) else None
        self.lcontracts=[None]*(self.nsite+1)
        self.rcontracts=[None]*(self.nsite+1)
        self.lcontracts[+0]=LEND
        self.rcontracts[-1]=REND
        self.setcontractions()

    def grow(self,sites,obonds,sbonds,osvs,qn=0):
        '''
        Infinite block growth.

        Parameters
        ----------
        sites : list of str/Label
            The new site identifiers/labels of the block after growth.
        obonds : list of str/Label
            The new mpo bond identifiers/labels of the block after growth.
        sbonds : list of str/Label
            The new mps bond identifiers/labels of the block after growth.
        osvs : 1d ndarray
            The old singular values.
        qn : QuantumNumber, optional
            The injected quantum number of the block.
        '''
        flag,onsite=self.nsite>0,self.nsite
        if flag: oldlcontracts=self.lcontracts[0:onsite/2+1]
        if flag: oldrcontracts=self.rcontracts[-onsite/2-1:]
        self.mpo=self.mpo.impogrowth(sites,obonds,ttype=self.ttype)
        self.mps=self.mps.impsgrowth(sites,sbonds,osvs,qn=qn,ttype=self.ttype)
        self.target=qn+self.target if isinstance(qn,QuantumNumber) else None
        self.lcontracts=[None]*(self.nsite+1)
        self.rcontracts=[None]*(self.nsite+1)
        if flag: self.lcontracts[0:onsite/2+1]=oldlcontracts
        if flag: self.rcontracts[-onsite/2-1:]=oldrcontracts
        self.setcontractions(SL=onsite/2+1,EL=self.nsite/2,SR=self.nsite-onsite/2-1,ER=self.nsite/2)

    def iterate(self,log,info='',sp=True,nmax=200,tol=hm.TOL,ebase=None,piechart=True):
        '''
        The two site dmrg step.

        Parameters
        ----------
        log : Log
            The log file.
        info : str, optional
            The information string passed to self.log.
        sp : logical, optional
            True for state prediction False for not.
        nmax : int, optional
            The maximum singular values to be kept.
        tol : float, optional
            The tolerance of the singular values.
        ebase : float, optional
            The base for calculating the ground state energy per site.
        piechart : logical, optional
            True for showing the piechart of self.timers while False for not.
        '''
        log<<'%s(%s)\n%s\n'%(info,self.ttype,self.graph)
        with self.timers.get('Preparation'):
            Ha,Hasite=self.lcontracts[self.cut-1],self.mpo[self.cut-1]
            Hb,Hbsite=self.rcontracts[self.cut+1],self.mpo[self.cut]
            Oa,(La,Sa,Ra)=Hasite.labels[MPO.R],self.mps[self.cut-1].labels
            Ob,(Lb,Sb,Rb)=Hbsite.labels[MPO.L],self.mps[self.cut].labels
            assert Ra==Lb and Oa==Ob
            Lsys,sysinfo=Label.union([La,Sa],'__DMRG_ITERATE_SYS__',flow=+1 if self.mps.qnon else 0,mode=+2 if self.ttype=='S' else +1)
            Lenv,envinfo=Label.union([Sb,Rb],'__DMRG_ITERATE_ENV__',flow=-1 if self.mps.qnon else 0,mode=+2 if self.ttype=='S' else +1)
            subslice=QuantumNumbers.kron([Lsys.qns,Lenv.qns],signs=[1,-1]).subslice(targets=(self.target.zero(),)) if self.mps.qnon else slice(None)
            shape=(len(subslice),len(subslice)) if self.mps.qnon else (Lsys.qns*Lenv.qns,Lsys.qns*Lenv.qns)
            Hsys=(Ha*Hasite).transpose([Oa,La.P,Sa.P,La,Sa]).merge(([La.P,Sa.P],Lsys.P.inverse,sysinfo),([La,Sa],Lsys.inverse,sysinfo))
            Henv=(Hbsite*Hb).transpose([Ob,Sb.P,Rb.P,Sb,Rb]).merge(([Sb.P,Rb.P],Lenv.P.inverse,envinfo),([Sb,Rb],Lenv.inverse,envinfo))
            matvec=DMRGMatVec(Hsys,Henv)
            def timedmatvec(v):
                with self.timers.get('matvec'): return matvec(v)
            matrix=hm.LinearOperator(shape=shape,matvec=timedmatvec,dtype=Hsys.dtype)
        with self.timers.get('Diagonalization'):
            u,s,v=self.mps[self.cut-1],self.mps.Lambda,self.mps[self.cut]
            v0=(u*s*v).merge(([La,Sa],Lsys,sysinfo),([Sb,Rb],Lenv,envinfo)).toarray().reshape(-1)[subslice] if sp and s.norm>RZERO else None
            es,vs=hm.eigsh(matrix,which='SA',v0=v0,k=1)
            energy,Psi=es[0],vs[:,0]
            self.info['Etotal']=energy,'%.6f'
            self.info['Esite']=(energy-(ebase or 0.0))/self.nsite,'%.8f'
            self.info['nmatvec']=matrix.count
            self.info['overlap']=np.inf if v0 is None else np.abs(Psi.conjugate().dot(v0)/norm(v0)/norm(Psi)),'%.6f'
        with self.timers.get('Truncation'):
            sysantiinfo=sysinfo if self.ttype=='S' else np.argsort(sysinfo) if self.mps.qnon else None
            envantiinfo=envinfo if self.ttype=='S' else np.argsort(envinfo) if self.mps.qnon else None
            qns=QuantumNumbers.mono(self.target.zero(),count=len(subslice)) if self.mps.qnon else Lsys.qns*Lenv.qns
            Lgs,new=Label('__DMRG_ITERATE_GS__',qns=qns),Ra.replace(qns=None)
            u,s,v,err=partitionedsvd(Tensor(Psi,labels=[Lgs]),Lsys,new,Lenv,nmax=nmax,tol=tol,ttype=self.ttype,returnerr=True)
            self.mps[self.cut-1]=u.split((Lsys,[La,Sa],sysantiinfo))
            self.mps[self.cut]=v.split((Lenv,[Sb,Rb],envantiinfo))
            self.mps.Lambda=s
            self.setlcontract(self.cut)
            self.setrcontract(self.cut)
            self.info['nslice']=Lgs.dim
            self.info['nbasis']=s.shape[0]
            self.info['err']=err,'%.1e'
        self.timers.record()
        log<<'timers of the dmrg:\n%s\n'%self.timers.tostr(Timers.ALL)
        log<<'info of the dmrg:\n%s\n\n'%self.info
        if piechart: self.timers.graph(parents=Timers.ALL)

class DMRG(Engine):
    '''
    Density matrix renormalization group method.

    Attributes
    ----------
    lattice : Cylinder/Lattice
        The lattice of the DMRG.
    terms : list of Term
        The terms of the DMRG.
    config : IDFConfig
        The configuration of internal degrees of freedom on a lattice.
    degfres : DegFreTree
        The tree stucture of the physical degrees of freedom.
    mask : [] or ['nambu']
        [] for spin systems and ['nambu'] for fermionic systems.
    dtype : np.float64, np.complex128
        The data type.
    generator : Generator
        The generator of the Hamiltonian.
    block : Block
        The block of the DMRG.
    cache : dict
        The cache of the DMRG.
    '''
    DTRP=2
    CORE=('lattice','block','cache')

    def __init__(self,lattice,terms,config,degfres,mask=(),ttype='D',dtype=np.complex128,target=0,**karg):
        '''
        Constructor.

        Parameters
        ----------
        lattice : Cylinder
            The lattice of the DMRG.
        terms : list of Term
            The terms of the DMRG.
        config : IDFConfig
            The configuration of the internal degrees of freedom on the lattice.
        degfres : DegFreTree
            The physical degrees of freedom tree.
        mask : [] or ['nambu']
            [] for spin systems and ['nambu'] for fermionic systems.
        ttype : 'D'/'S'
            Tensor type. 'D' for dense and 'S' for sparse.
        dtype : np.float64,np.complex128, optional
            The data type.
        target : QuantumNumber, optional
            The target of the block of the DMRG.
        '''
        assert config.priority==degfres.priority
        self.lattice=lattice
        self.terms=terms
        self.config=config
        self.degfres=degfres
        self.mask=mask
        self.dtype=dtype
        self.generator=Generator(bonds=lattice.bonds,config=config,terms=terms,boundary=self.boundary,dtype=dtype,half=False)
        if self.map is None: self.parameters.update(OrderedDict((term.id,term.value) for term in terms))
        self.initblock(target=target,ttype=ttype)
        self.cache={}
        self.logging()

    @property
    def nspb(self):
        '''
        The number of site labels per block.
        '''
        assert isinstance(self.lattice,Cylinder)
        return DMRG.NS(self.config,self.degfres,self.lattice,self.mask)

    def update(self,**karg):
        '''
        Update the DMRG with new parameters.
        '''
        if len(karg)>0:
            super(DMRG,self).update(**karg)
            self.generator.update(**self.data)

    def initblock(self,target=None,ttype='D'):
        '''
        Init the block of the DMRG.

        Parameters
        ----------
        target : QuantumNumber, optional
            The target of the block of the DMRG.
        ttype : 'D'/'S', optional

        '''
        if len(self.lattice)>0:
            mpo=MPO.fromoperators(self.generator.operators,self.degfres,ttype=ttype)
            sites,bonds=self.degfres.labels('S'),self.degfres.labels('B')
            if ttype=='S': sites=[site.replace(qns=site.qns.sorted()) for site in sites]
            if isinstance(target,QuantumNumber):
                bonds[+0]=Label(bonds[+0],qns=QuantumNumbers.mono(target.zero()),flow=+1)
                bonds[-1]=Label(bonds[-1],qns=QuantumNumbers.mono(target),flow=-1)
            mps=MPS.random(sites,bonds,cut=len(sites)/2,nmax=10,ttype=ttype,dtype=self.dtype)
        else:
            mpo=MPO()
            mps=MPS()
        self.block=Block(mpo,mps,target=target,ttype=ttype)

    def sweep(self,info='',path=None,**karg):
        '''
        Perform a sweep over the mps of the dmrg under the guidance of `path`.

        Parameters
        ----------
        info : str, optional
            The info passed to self.log.
        path : list of str, optional
            The path along which the sweep is performed.
        '''
        for move in it.chain(['<<']*(self.block.cut-1),['>>']*(self.block.nsite-2),['<<']*(self.block.nsite-self.block.cut-1)) if path is None else path:
            self.block.mps<<=1 if '<<' in move else -1
            self.block.iterate(self.log,info='%s_%s %s(%s)'%(self,self.block,info,move),sp=True,**karg)

    def dump(self):
        '''
        Use pickle to dump the core of the dmrg.
        '''
        with open('%s/%s_%s_%s_%s.dat'%(self.din,self,self.block.nsite,self.block.target,self.block.mps.nmax),'wb') as fout:
            pk.dump({key:getattr(self,key) for key in self.CORE},fout,2)

    @staticmethod
    def NS(config,degfres,lattice,mask):
        '''
        The number of site labels.

        Parameters
        ----------
        config : IDFConfig
            The configuration of internal degrees of freedom on a lattice.
        degfres : DegFreTree
            The tree stucture of the physical degrees of freedom.
        lattice : Cylinder/Lattice
            The lattice.
        mask : [] or ['nambu']
            [] for spin systems and ['nambu'] for fermionic systems.

        Returns
        -------
        int
            The number of site labels.

        Notes
        -----
        When lattice is a cyliner, return the number of site labels per block;
        When lattice is a lattice, return the number of site labels for the whole lattice.
        '''
        config,degfres=deepcopy(config),deepcopy(degfres)
        config.reset(pids=(lattice(['__DMRG_NSPB__']) if isinstance(lattice,Cylinder) else lattice).pids)
        degfres.reset(leaves=config.table(mask=mask).keys())
        return len(degfres.indices())

    @staticmethod
    def rematch(pattern):
        '''
        Convert a data-filename pattern to regular-expression-match pattern.
        '''
        ss=['(',')','[',']','^','+']
        rs=['\(','\)','\[','\]','\^','\+']
        for s,r in zip(ss,rs):
            pattern=pattern.replace(s,r)
        return pattern

    @staticmethod
    def load(din,pattern,nmax):
        '''
        Use pickle to load the core of the dmrg from existing data files.

        Parameters
        ----------
        din : str
            The directory where the data files are searched.
        pattern : str
            The  pattern of the data files to match.
        nmax : int
            The maximum number of singular values kept in the mps.

        Returns
        -------
        dict
            The loaded core of the dmrg.
        '''
        candidates={}
        names=[name for name in os.listdir(din) if re.match(DMRG.rematch(pattern),name)]
        for name in names:
            split=name.split('_')
            cnmax=int(re.findall(r'\d+',split[-1])[0])
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
    target : callable that returns QuantumNumber
        This function returns the target space at each growth of the DMRG.
    maxiter : int
        The maximum times of growth.
    nmax : int
        The maximum singular values to be kept.
    npresweep : int
        The number of presweeps to make a random mps converged to the target state.
    nsweep : int
        The number of sweeps to make the predicted mps converged to the target state.
    tol : float
        The tolerance of the target state energy.
    '''

    def __init__(self,target=None,maxiter=10,nmax=400,npresweep=10,nsweep=4,tol=10**-5,**karg):
        '''
        Constructor.

        Parameters
        ----------
        target : callable that returns QuantumNumber, optional
            This function returns the target space at each growth of the DMRG.
        maxiter : int, optional
            The maximum times of growth.
        nmax : int, optional
            The maximum number of singular values to be kept.
        npresweep : int, optional
            The number of presweeps to make a random mps converged to the target state.
        nsweep : int, optional
            The number of sweeps to make the predicted mps converged to the target state.
        tol : float, optional
            The tolerance of the target state energy.
        '''
        self.target=(lambda niter: None) if target is None else target
        self.maxiter=maxiter
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
        int
            The recover code.
        '''
        if engine.__class__.__name__=='fDMRG':
            for niter in xrange(self.maxiter-1,-1,-1):
                core=DMRG.load(din=engine.din,pattern='%s_%s_%s'%(engine,(niter+1)*engine.nspb*2,self.target(niter)),nmax=self.nmax)
                if core: break
        else:
            core=DMRG.load(din=engine.din,pattern='%s_%s'%(engine,engine.nspb*2),nmax=self.nmax)
            niter=None
        if core:
            for key,value in core.iteritems(): setattr(engine,key,value)
            engine.config.reset(pids=engine.lattice.pids)
            engine.degfres.reset(leaves=engine.config.table(mask=engine.mask).keys())
            engine.generator.reset(bonds=engine.lattice.bonds,config=engine.config)
            code=getattr(engine,'niter',niter)
            assert engine.block.target==self.target(code)
        else:
            code=-1
        return code

class TSS(App):
    '''
    Two site sweep of a DMRG.

    Attributes
    ----------
    target : QuantumNumber
        The target of the DMRG's mps.
    nsite : int
        The length of the DMRG's mps.
    nmaxs : list of int
        The maximum numbers of singular values to be kept for the sweeps.
    paths : list of list of '<<' or '>>'
        The paths along which the sweeps are performed.
    '''

    def __init__(self,target,nsite,nmaxs,paths=None,**karg):
        '''
        Constructor.

        Parameters
        ----------
        target : QuantumNumber
            The target of the DMRG's mps.
        nsite : int
            The length of the DMRG's mps.
        nmaxs : list of int
            The maximum numbers of singular values to be kept for each sweep.
        paths : list of list of '<<' or '>>', optional
            The paths along which the sweeps are performed.
        '''
        self.target=target
        self.nsite=nsite
        self.nmaxs=nmaxs
        self.paths=[None]*len(nmaxs) if paths is None else paths
        assert len(nmaxs)==len(self.paths)

    def recover(self,engine):
        '''
        Recover the core of a dmrg engine.

        Parameters
        ----------
        engine : DMRG
            The dmrg engine whose core is to be recovered.

        Returns
        -------
        int
            The recover code.
        '''
        for i,nmax in enumerate(reversed(self.nmaxs)):
            core=DMRG.load(din=engine.din,pattern='%s_%s_%s'%(engine,self.nsite,self.target),nmax=nmax)
            if core:
                for key,value in core.iteritems(): setattr(engine,key,value)
                engine.config.reset(pids=engine.lattice.pids)
                engine.degfres.reset(leaves=engine.config.table(mask=engine.mask).keys())
                engine.generator.reset(bonds=engine.lattice.bonds,config=engine.config)
                engine.block.mps>>=1 if engine.block.cut==0 else -1 if engine.block.cut==engine.block.nsite else 0
                code=len(self.nmaxs)-1-i
                if engine.block.mps.nmax<nmax: code-=1
                break
        else:
            code=None
        return code

class GSE(App):
    '''
    Ground state energy.

    Attributes
    ----------
    path : BaseSpace
        The path in the parameter space to calculate the ground state energy.
    '''

    def __init__(self,path,**karg):
        '''
        Constructor.

        Parameters
        ----------
        path : BaseSpace
            The path in the parameter space to calculate the ground state energy.
        '''
        self.path=path