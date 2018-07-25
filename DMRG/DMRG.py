'''
====================================
Density matrix renormalization group
====================================

DMRG, including:
    * classes: Block, DMRG, TSG, TSS
    * function: pattern
'''

__all__=['Block','pattern','DMRG','TSG','TSS']

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
    divisor : int
        The extra divisor of extensive quantities and intensive quantities.
    lcontracts : list of 3d DTensor/STensor
        The contraction of mpo and mps from the left.
    rcontracts : list of 3d DTensor/STensor
        The contraction of mpo and mps from the right.
    timers : Timers
        The timers of the block.
    info : Sheet
        The info of the block.
    '''

    def __init__(self,mpo,mps,target=None,divisor=1,LEND=None,REND=None):
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
        divisor : int, optional
            The extra divisor of extensive quantities and intensive quantities.
        LEND/REND : 3d DTensor/STensor, optional
            The leftmost/rightmost end of the contraction of mpo and mps.
        '''
        self.reset(mpo,mps,target=target,divisor=divisor,LEND=LEND,REND=REND)
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
        return {'mpo':self.mpo,'mps':self.mps,'target':self.target,'lcontracts':self.lcontracts,'rcontracts':self.rcontracts,'info':self.info}

    def __setstate__(self,state):
        '''
        Used for pickle.
        '''
        for key,value in state.itervalues(): setattr(self,key,value)
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
        result.divisor=self.divisor
        result.lcontracts=self.lcontracts[i:j+1]
        result.rcontracts=self.rcontracts[i:j+1]
        result.timers=self.timers
        result.info=self.info
        return result

    def reset(self,mpo=None,mps=None,target=None,divisor=1,LEND=None,REND=None):
        '''
        Constructor.

        Parameters
        ----------
        mpo : MPO, optional
            The MPO of the block.
        mps : MPS, optional
            The MPS of the block.
        target : QuantumNumber, optional
            The target of the block.
        divisor : int, optional
            The extra divisor of extensive quantities and intensive quantities.
        LEND/REND : 3d DTensor/STensor, optional
            The leftmost/rightmost end of the contraction of mpo and mps.
        '''
        if mpo is None: mpo=self.mpo
        if mps is None: mps=self.mps
        if target is None: target=self.target
        assert len(mpo)==len(mps)
        self.mpo=mpo
        self.mps=mps
        self.target=target
        self.divisor=divisor
        if self.nsite>0:
            if LEND is None:
                C=self.mpo[0].labels[MPO.L].inverse
                D=self.mps[0].labels[MPS.L].inverse
                assert C.dim==1==D.dim
                LEND=Tensor([[[1.0]]],labels=[D.P,C,D])
            if REND is None:
                C=self.mpo[self.nsite-1].labels[MPO.R].inverse
                D=self.mps[self.nsite-1].labels[MPS.R].inverse
                assert C.dim==1==D.dim
                REND=Tensor([[[1.0]]],labels=[D.P,C,D])
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
        #self.setlcontract(+0,job='relabel')
        #self.setrcontract(-1,job='relabel')
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
        self.mpo=self.mpo.impogrowth(sites,obonds)
        self.mps=self.mps.impsgrowth(sites,sbonds,osvs,qn=qn)
        self.target=qn+self.target if isinstance(qn,QuantumNumber) else None
        self.lcontracts=[None]*(self.nsite+1)
        self.rcontracts=[None]*(self.nsite+1)
        if flag: self.lcontracts[0:onsite/2+1]=oldlcontracts
        if flag: self.rcontracts[-onsite/2-1:]=oldrcontracts
        self.setcontractions(SL=onsite/2+1,EL=self.nsite/2,SR=self.nsite-onsite/2-1,ER=self.nsite/2)

    def iterate(self,log,info='',sp=True,nmax=200,tol=hm.TOL,piechart=True):
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
        piechart : logical, optional
            True for showing the piechart of self.timers while False for not.
        '''
        log<<'%s\n%s\n'%(info,self.graph)
        with self.timers.get('Preparation'):
            Ha,Hasite=self.lcontracts[self.cut-1],self.mpo[self.cut-1]
            Hb,Hbsite=self.rcontracts[self.cut+1],self.mpo[self.cut]
            Oa,(La,Sa,Ra)=Hasite.labels[MPO.R],self.mps[self.cut-1].labels
            Ob,(Lb,Sb,Rb)=Hbsite.labels[MPO.L],self.mps[self.cut].labels
            assert Ra==Lb and Oa==Ob
            Lsys,syspt=Label.union([La,Sa],'__DMRG_ITERATE_SYS__',flow=+1 if self.mps.mode=='QN' else 0,mode=+1)
            Lenv,envpt=Label.union([Sb,Rb],'__DMRG_ITERATE_ENV__',flow=-1 if self.mps.mode=='QN' else 0,mode=+1)
            if self.mps.mode=='QN':
                zero=self.target.zero()
                sysantipt,sysod=np.argsort(syspt),Lsys.qns.toordereddict()
                envantipt,envod=np.argsort(envpt),Lenv.qns.toordereddict()
                subslice=QuantumNumbers.kron([Lsys.qns,Lenv.qns],signs=[1,-1]).subslice(targets=(zero,))
                qns=QuantumNumbers.mono(zero,count=len(subslice))
                qnpairs=[[(sqn,sqn-oqn) for sqn in Lsys.qns if sqn in envod and sqn-oqn in sysod and sqn-oqn in envod] for oqn in Oa.qns]
            else:
                sysantipt,envantipt,subslice=None,None,slice(None)
                qns=Lsys.qns*Lenv.qns
            Hsys=(Ha*Hasite).transpose([Oa,La.P,Sa.P,La,Sa]).merge(([La.P,Sa.P],Lsys.P,syspt),([La,Sa],Lsys,syspt))
            Henv=(Hbsite*Hb).transpose([Ob,Sb.P,Rb.P,Sb,Rb]).merge(([Sb.P,Rb.P],Lenv.P,envpt),([Sb,Rb],Lenv,envpt))
            if self.mps.mode=='QN':
                vecold=np.zeros(Hsys.shape[1]*Henv.shape[1],dtype=Hsys.dtype)
                vecnew=np.zeros((Hsys.shape[1],Henv.shape[1]),dtype=Hsys.dtype)
                def matvec(v):
                    with self.timers.get('matvec'):
                        vec,result=vecold,vecnew
                        vec[subslice]=v
                        result[...]=0.0
                        vec=vec.reshape((Hsys.shape[1],Henv.shape[1]))
                        for hsys,henv,pairs in zip(Hsys.data,Henv.data,qnpairs):
                            for qn1,qn2 in pairs:
                                result[sysod[qn1],envod[qn1]]+=hsys[sysod[qn1],sysod[qn2]].dot(vec[sysod[qn2],envod[qn2]]).dot(henv.T[envod[qn2],envod[qn1]])
                    return result.reshape(-1)[subslice] 
                matrix=hm.LinearOperator(shape=(len(subslice),len(subslice)),matvec=matvec,dtype=Hsys.dtype)
            else:
                def matvec(v):
                    with self.timers.get('matvec'):
                        v=v.reshape((Hsys.shape[1],Henv.shape[1]))
                        result=np.zeros_like(v)
                        for hsys,henv in zip(Hsys.data,Henv.data):
                            result+=hsys.dot(v).dot(henv.T)
                    return result.reshape(-1)
                matrix=hm.LinearOperator(shape=(Hsys.shape[1]*Henv.shape[1],Hsys.shape[1]*Henv.shape[1]),matvec=matvec,dtype=Hsys.dtype)
        with self.timers.get('Diagonalization'):
            u,s,v=self.mps[self.cut-1],self.mps.Lambda,self.mps[self.cut]
            v0=(u*s*v).merge(([La,Sa],Lsys,syspt),([Sb,Rb],Lenv,envpt)).toarray().reshape(-1)[subslice] if sp and s.norm>RZERO else None
            es,vs=hm.eigsh(matrix,which='SA',v0=v0,k=1)
            energy,Psi=es[0],vs[:,0]
            self.info['Etotal']=energy,'%.6f'
            self.info['Esite']=energy/self.nsite/self.divisor,'%.8f'
            self.info['nmatvec']=matrix.count
            self.info['overlap']=np.inf if v0 is None else np.abs(Psi.conjugate().dot(v0)/norm(v0)/norm(Psi)),'%.6f'
        with self.timers.get('Truncation'):
            Lgs,new=Label('__DMRG_ITERATE_GS__',qns=qns),Ra.replace(qns=None)
            u,s,v,err=partitionedsvd(Tensor(Psi,labels=[Lgs]),Lsys,new,Lenv,nmax=nmax,tol=tol,returnerr=True)
            self.mps[self.cut-1]=u.split((Lsys,[La,Sa],sysantipt))
            self.mps[self.cut]=v.split((Lenv,[Sb,Rb],envantipt))
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
    nsite : int
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

    def __init__(self,lattice,terms,config,degfres,mask=(),dtype=np.complex128,target=0,**karg):
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
        self.initblock(target=target)
        self.cache={}
        self.logging()

    @property
    def nspb(self):
        '''
        The number of site labels per block.
        '''
        config,degfres=deepcopy(self.config),deepcopy(self.degfres)
        config.reset(pids=self.lattice(['__DMRG_NSPB__']).pids)
        degfres.reset(leaves=config.table(mask=self.mask).keys())
        return len(degfres.indices())

    def initblock(self,target=None):
        '''
        Init the block of the DMRG.

        Parameters
        ----------
        target : QuantumNumber, optional
            The target of the block of the DMRG.
        '''
        if len(self.lattice)>0:
            mpo=MPO.fromoperators(self.generator.operators,self.degfres)
            sites,bonds=self.degfres.labels('S'),self.degfres.labels('B')
            if self.degfres.mode=='QN':
                bonds[+0]=Label(bonds[+0],qns=QuantumNumbers.mono(target.zero()),flow=+1)
                bonds[-1]=Label(bonds[-1],qns=QuantumNumbers.mono(target),flow=-1)
            mps=MPS.random(sites,bonds,cut=len(sites)/2,nmax=10,dtype=self.dtype)
        else:
            mpo=MPO()
            mps=MPS(mode=self.degfres.mode)
        self.block=Block(mpo,mps,target=target)

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
        with open('%s/%s_%s.dat'%(self.din,pattern(self.name,self.parameters,self.block.target,self.block.nsite,mode='py'),self.block.mps.nmax),'wb') as fout:
            core=   {
                        'lattice':  self.lattice,
                        'block':    self.block,
                        'cache':    self.cache
                        }
            pk.dump(core,fout,2)

    @staticmethod
    def load(din,pattern,nmax):
        '''
        Use pickle to load the core of the dmrg from existing data files.

        Parameters
        ----------
        din : str
            The directory where the data files are searched.
        pattern : str
            The matching pattern of the data files.
        nmax : int
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
    nmax : int
        The maximum singular values to be kept.
    npresweep : int
        The number of presweeps to make a random mps converged to the target state.
    nsweep : int
        The number of sweeps to make the predicted mps converged to the target state.
    tol : float
        The tolerance of the target state energy.
    '''

    def __init__(self,targets,nmax,npresweep=10,nsweep=4,tol=10**-6,**karg):
        '''
        Constructor.

        Parameters
        ----------
        targets : sequence of QuantumNumber
            The target space at each growth of the DMRG.
        nmax : int
            The maximum number of singular values to be kept.
        npresweep : int, optional
            The number of presweeps to make a random mps converged to the target state.
        nsweep : int, optional
            The number of sweeps to make the predicted mps converged to the target state.
        tol : float, optional
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
        int
            The recover code.
        '''
        for i,target in enumerate(reversed(self.targets)):
            core=DMRG.load(din=engine.din,pattern=pattern(engine.name,engine.parameters,target,(len(self.targets)-i)*engine.nspb*2,mode='re'),nmax=self.nmax)
            if core:
                for key,value in core.iteritems(): setattr(engine,key,value)
                engine.config.reset(pids=engine.lattice.pids)
                engine.degfres.reset(leaves=engine.config.table(mask=engine.mask).keys())
                engine.generator.reset(bonds=engine.lattice.bonds,config=engine.config)
                code=len(self.targets)-i-1
                break
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
    BS : BaseSpace
        The basespace of the DMRG's parameters for the sweeps.
    paths : list of list of '<<' or '>>'
        The paths along which the sweeps are performed.
    forcesweep : logical
        When True, the sweep will be taken at least once even if the mps are recovered from existing data files perfectly.
        When False, no real sweep will be taken if the mps can be perfectly recovered from existing data files.
    '''

    def __init__(self,target,nsite,nmaxs,BS=None,paths=None,forcesweep=False,**karg):
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
        BS : list of dict, optional
            The parameters of the DMRG for each sweep.
        paths : list of list of '<<' or '>>', optional
            The paths along which the sweeps are performed.
        forcesweep : logical, optional
            When True, the sweep will be taken at least once even if the mps are recovered from existing data files perfectly.
            When False, no real sweep will be taken if the mps can be perfectly recovered from existing data files.
        '''
        self.target=target
        self.nsite=nsite
        self.nmaxs=nmaxs
        self.BS=[{}]*len(nmaxs) if BS is None else BS
        self.paths=[None]*len(nmaxs) if paths is None else paths
        assert len(nmaxs)==len(self.BS) and len(nmaxs)==len(self.paths)
        self.forcesweep=forcesweep

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
        parameters=deepcopy(engine.parameters)
        for i,(nmax,paras) in enumerate(reversed(zip(self.nmaxs,self.BS))):
            parameters.update(paras)
            core=DMRG.load(din=engine.din,pattern=pattern(self.name,parameters,self.target,self.nsite,mode='re'),nmax=nmax)
            if core:
                for key,value in core.iteritems(): setattr(engine,key,value)
                engine.parameters=parameters
                engine.config.reset(pids=engine.lattice.pids)
                engine.degfres.reset(leaves=engine.config.table(mask=engine.mask).keys())
                engine.block.mps>>=1 if engine.block.cut==0 else -1 if engine.block.cut==engine.block.nsite else 0
                code=len(self.nmaxs)-1-i
                if self.forcesweep or engine.block.mps.nmax<nmax: code-=1
                break
        else:
            code=None
        return code
