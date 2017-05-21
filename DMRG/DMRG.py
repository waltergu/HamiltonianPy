'''
====================================
Density matrix renormalization group
====================================

DMRG, including:
    * classes: Cylinder, DMRG, TSG, TSS
    * function: pattern, DMRGTSG, DMRGTSS
'''

__all__=['pattern','Cylinder','DMRG','TSG','DMRGTSG','TSS','DMRGTSS']

import os
import re
import numpy as np
import pickle as pk
import itertools as it
import scipy.sparse as sp
import HamiltonianPy.Misc as hm
import matplotlib.pyplot as plt
from numpy.linalg import norm
from HamiltonianPy import *
from HamiltonianPy.TensorNetwork import *
from copy import copy,deepcopy

def pattern(status,target,nsite,mode='re'):
    '''
    Return the pattern of data files for match.

    Parameters
    ----------
    status : Status
        The status of the DMRG.
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
    result='%s_(%s,%s)'%(status,tuple(target) if isinstance(target,QuantumNumber) else None,nsite)
    if mode=='re':
        ss=['(',')','[',']']
        rs=['\(','\)','\[','\]']
        for s,r in zip(ss,rs):
            result=result.replace(s,r)
    return result

class Cylinder(Lattice):
    '''
    The cylinder geometry of a lattice.

    Attributes
    ----------
    block : list of 1d ndarray
        The building block of the cylinder.
    translation : 1d ndarray
        The translation vector of the building block to construct the cylinder.
    '''

    def __init__(self,block,translation,**karg):
        '''
        Constructor.

        Parameters
        ----------
        block : list of 1d ndarray
            The building block of the cylinder.
        translation : 1d ndarray
            The translation vector of the building block to construct the cylinder.
        '''
        super(Cylinder,self).__init__(**karg)
        self.block=block
        self.translation=translation

    def insert(self,A,B,news=None):
        '''
        Insert two blocks into the center of the cylinder.

        Parameters
        ----------
        A,B : any hashable object
            The scopes of the insert block points.
        news : list of any hashable object, optional
            The new scopes for the points of the cylinder before the insertion.
            If None, the old scopes remain unchanged.
        '''
        if len(self)==0:
            aps=[Point(PID(scope=A,site=i),rcoord=rcoord-self.translation/2,icoord=np.zeros_like(rcoord)) for i,rcoord in enumerate(self.block)]
            bps=[Point(PID(scope=B,site=i),rcoord=rcoord+self.translation/2,icoord=np.zeros_like(rcoord)) for i,rcoord in enumerate(self.block)]
            ass,bss=[],[]
        else:
            if news is not None:
                assert len(news)*len(self.block)==len(self)
                for i,scope in enumerate(news):
                    for j in xrange(len(self.block)):
                        self.points[i*len(self.block)+j].pid=self.points[i*len(self.block)+j].pid._replace(scope=scope)
            aps,bps=self.points[:len(self)/2],self.points[len(self)/2:]
            for ap,bp in zip(aps,bps):
                ap.rcoord-=self.translation
                bp.rcoord+=self.translation
            ass=[Point(PID(scope=A,site=i),rcoord=rcoord-self.translation/2,icoord=np.zeros_like(rcoord)) for i,rcoord in enumerate(self.block)]
            bss=[Point(PID(scope=B,site=i),rcoord=rcoord+self.translation/2,icoord=np.zeros_like(rcoord)) for i,rcoord in enumerate(self.block)]
        self.points=aps+ass+bss+bps
        links,mindists=intralinks(
                mode=                   'nb',
                cluster=                np.asarray([p.rcoord for p in self.points]),
                indices=                None,
                vectors=                self.vectors,
                nneighbour=             self.nneighbour,
                max_coordinate_number=  self.max_coordinate_number,
                return_mindists=        True
                )
        self.links=links
        self.mindists=mindists

    def lattice(self,scopes):
        '''
        Construct a cylinder with the assigned scopes.

        Parameters
        ----------
        scopes : list of hashable object
            The scopes of the cylinder.

        Returns
        -------
        Lattice
            The constructed cylinder.
        '''
        points,num=[],len(scopes)
        for i,rcoord in enumerate(tiling(self.block,[self.translation],np.linspace(-(num-1)/2.0,(num-1)/2.0,num) if num>1 else xrange(1))):
            points.append(Point(PID(scope=scopes[i/len(self.block)],site=i%len(self.block)),rcoord=rcoord,icoord=np.zeros_like(rcoord)))
        return Lattice.compose(name=self.name,points=points,vectors=self.vectors,nneighbour=self.nneighbour)

class DMRG(Engine):
    '''
    Density matrix renormalization group method.

    Attributes
    ----------
    mps : MPS
        The matrix product state of the DMRG.
    lattice : Cylinder/Lattice
        The lattice of the DMRG.
    terms : list of Term
        The terms of the DMRG.
    config : IDFConfig
        The configuration of the internal degrees of freedom on the lattice.
    degfres : DegFreTree
        The physical degrees of freedom tree.
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
    _Hs_ : dict
        * entry 'L': list of 3d Tensor
            The contraction of mpo and mps from the left.
        * entry 'R': list of 3d Tensor
            The contraction of mpo and mps from the right.
    timers : Timers
        The timers of the dmrg processes.
    info : Info
        The info of the dmrg processes.
    cache : dict
        * entry 'osvs': 1d ndarray
            The old singular values of the DMRG.
    '''

    def __init__(self,mps,lattice,terms,config,degfres,mask=[],target=None,dtype=np.complex128,**karg):
        '''
        Constructor.

        Parameters
        ----------
        mps : MPS
            The matrix product state of the DMRG.
        lattice : Lattice
            The lattice of the DMRG.
        terms : list of Term
            The terms of the DMRG.
        config : IDFConfig
            The configuration of the internal degrees of freedom on the lattice.
        degfres : DegFreTree
            The physical degrees of freedom tree.
        mask : [] or ['nambu']
            [] for spin systems and ['nambu'] for fermionic systems.
        target : QuantumNumber
            The target space of the DMRG.
        dtype : np.float64,np.complex128, optional
            The data type.
        '''
        self.mps=mps
        self.lattice=lattice
        self.terms=terms
        self.config=config
        self.degfres=degfres
        self.mask=mask
        self.target=target
        self.dtype=dtype
        self.generator=Generator(bonds=lattice.bonds,config=config,terms=terms,dtype=dtype)
        self.status.update(const=self.generator.parameters['const'])
        self.status.update(alter=self.generator.parameters['alter'])
        self.set_operators()
        self.set_mpo()
        self.set_Hs_()
        self.timers=Timers('Preparation','Hamiltonian','Diagonalization','Truncation')
        self.timers.add(parent='Hamiltonian',name='kron')
        self.timers.add(parent='Hamiltonian',name='sum')
        if self.mps.mode=='QN':
            self.timers.add(parent='kron',name='csr')
            self.timers.add(parent='kron',name='fkron')
        self.info=Info('Etotal','Esite','dE/E','nbasis','nslice','nnz','nz','density','overlap','err')
        self.cache={}

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
        return '%s(target=%s,nsite=%s,cut=%s)'%(self.status,self.target,self.mps.nsite,self.mps.cut)

    def update(self,**karg):
        '''
        Update the DMRG with new parameters.
        '''
        self.generators['h'].update(**karg)
        self.status.update(alter=self.generators['h'].parameters['alter'])
        self.set_operators()
        self.set_mpo()
        self.set_Hs_()

    def set_operators(self):
        '''
        Set the operators of the DMRG.
        '''
        self.operators=self.generator.operators
        if self.mask==['nambu']:
            for operator in self.operators.values():
                self.operators+=operator.dagger

    def set_mpo(self):
        '''
        Set the mpo of the DMRG.
        '''
        if len(self.operators)>0:
            self.mpo=OptMPO([OptStr.from_operator(operator,self.degfres) for operator in self.operators.itervalues()],self.degfres).to_mpo()

    def set_HL_(self,pos,job='contract',tol=hm.TOL):
        '''
        Set a certain left block Hamiltonian.

        Parameters
        ----------
        pos : integer
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
        SL/EL : integer, optional
            The start/end position of the left Hamiltonians to be set.
        SR/ER : integer, optional
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
            The infomation string passed to self.log.
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
                rcs=subslice
                qns=QuantumNumbers.mono(self.target.zero(),count=len(subslice))
                self.info['nslice']=len(subslice)
            else:
                sysqns,syspt=np.product([La.qns,Sa.qns]),None
                envqns,envpt=np.product([Sb.qns,Rb.qns]),None
                sysantipt,envantipt=None,None
                subslice=slice(None)
                rcs=None
                qns=sysqns*envqns
                self.info['nslice']=qns
            Lpa,Spa,Spb,Rpb=La.prime,Sa.prime,Sb.prime,Rb.prime
            Lsys,Lenv,new=Label('__DMRG_TWO_SITE_STEP_SYS__',qns=sysqns),Label('__DMRG_TWO_SITE_STEP_ENV__',qns=envqns),Ra.replace(qns=None)
            Lpsys,Lpenv=Lsys.prime,Lenv.prime
            Hsys=contract([Ha,Hasite],engine='tensordot').transpose([Oa,Lpa,Spa,La,Sa]).merge(([Lpa,Spa],Lpsys,syspt),([La,Sa],Lsys,syspt))
            Henv=contract([Hbsite,Hb],engine='tensordot').transpose([Ob,Spb,Rpb,Sb,Rb]).merge(([Spb,Rpb],Lpenv,envpt),([Sb,Rb],Lenv,envpt))
            if rcs is None:
                rcs1,rcs2,slices=None,None,None
            else:
                rcs1,rcs2=np.divide(rcs,Henv.shape[1]),np.mod(rcs,Henv.shape[1])
                slices=np.zeros(Hsys.shape[1]*Henv.shape[1],dtype=np.int64)
                slices[rcs]=xrange(len(rcs))
        with self.timers.get('Hamiltonian'):
            matrix=0
            for hsys,henv in zip(Hsys,Henv):
                with self.timers.get('kron'):
                    temp=hm.kron(hsys,henv,rcs=rcs,rcs1=rcs1,rcs2=rcs2,slices=slices,timers=self.timers)
                with self.timers.get('sum'):
                    matrix+=temp
            self.info['nnz']=matrix.nnz
            self.info['nz']=(len(np.argwhere(np.abs(matrix.data)<tol))*100.0/matrix.nnz) if matrix.nnz>0 else 0,'%1.1f%%'
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
            self.info['overlap']=np.inf if v0 is None else np.abs(Psi.conjugate().dot(v0)/norm(v0)/norm(Psi)),'%.6f'
        with self.timers.get('Truncation'):
            u,s,v,err=Tensor(Psi,labels=[Label('__DMRG_TWO_SITE_STEP__',qns=qns)]).partitioned_svd(Lsys,new,Lenv,nmax=nmax,tol=tol,return_truncation_err=True)
            self.mps[self.mps.cut-1]=u.split((Lsys,[La,Sa],sysantipt))
            self.mps[self.mps.cut]=v.split((Lenv,[Sb,Rb],envantipt))
            self.mps.Lambda=s
            self.set_HL_(self.mps.cut-1,tol=tol)
            self.set_HR_(self.mps.cut,tol=tol)
            self.info['nbasis']=len(s)
            self.info['density']=1.0*self.info['nnz']/self.info['nslice']**2,'%.1e'
            self.info['err']=err,'%.1e'
        self.timers.record()
        self.log<<'timers of the dmrg:\n%s\n'%self.timers.tostr(Timers.ALL)
        self.log<<'info of the dmrg:\n%s\n\n'%self.info
        if piechart: self.timers.graph(parents=Timers.ALL)

    @staticmethod
    def imps_predict(mps,sites,bonds,osvs,target=None,dtype=np.float64):
        '''
        Infinite DMRG state prediction.

        Parameters
        ----------
        mps : MPS
            The mps to be predicted.
        sites/bonds : list of Label
            The new site/bond labels of the mps.
        osvs : 1d ndarray
            The old singular values.
        target : QuantumNumber, optional
            The new target of the mps.
        dtype : np.float64, np.complex128, etc, optional
            The data type of the mps.

        Returns
        -------
        nsvs : 1d ndarray
            The updated singular values for the next imps prediction.
        '''
        if mps.cut is None: mps.cut=0
        assert mps.cut==mps.nsite/2 and mps.nsite%2==0
        obonds=mps.bonds
        diff=0 if target is None else (target if mps.nsite==0 else target-next(iter(obonds[-1].qns)))
        ob,nb=mps.nsite/2+1,(len(bonds)+1)/2
        L,LS,RS,R=bonds[:ob],bonds[ob:nb],bonds[nb:-ob],bonds[-ob:]
        if mps.cut==0:
            L[+0]=L[+0].replace(qns=QuantumNumbers.mono(target.zero(),count=1) if mps.mode=='QN' else 1)
            R[-1]=R[-1].replace(qns=QuantumNumbers.mono(target,count=1) if mps.mode=='QN' else 1)
        for i,(bond,obond) in enumerate(zip(L,obonds[:ob])):
            L[i]=bond.replace(qns=obond.qns)
        for i,(bond,obond) in enumerate(zip(R,obonds[-ob:])):
            R[i]=bond.replace(qns=obond.qns+diff)
        if mps.nsite>0:
            for i,(bond,obond) in enumerate(zip(LS,obonds[ob:nb])):
                LS[i]=bond.replace(qns=obond.qns)
            for i,(bond,obond) in enumerate(zip(RS,obonds[-nb+1:-ob])):
                RS[i]=bond.replace(qns=obond.qns+diff)
        else:
            LS,RS=deepcopy(LS),deepcopy(RS)
        nbonds=L+LS+RS+R
        ns,nms,lms,rms=nb-ob,[],[],[]
        if mps.nsite>0:
            us,vs,nsvs=mps[mps.cut-ns:mps.cut],mps[mps.cut:mps.cut+ns],np.asarray(mps.Lambda)
            for i,(L,S,R) in enumerate(zip(nbonds[ob-1:nb-1],sites[ob-1:nb-1],nbonds[ob:nb])):
                m=Tensor(np.asarray(vs[i].dotarray(axis=MPS.L,array=nsvs) if i==0 else vs[i]),labels=[L,S,R])
                u,s,v=m.svd(row=[L,S],new=Label('__DMRG_INSERT_L_%i__'%i),col=[R],row_signs='++',col_signs='+')
                if i<len(vs)-1:
                    u.relabel(olds=s.labels,news=[R.replace(qns=s.labels[0].qns)])
                    vs[i+1].relabel(olds=[MPS.L],news=[R.replace(qns=s.labels[0].qns)])
                    vs[i+1]=contract([s,v,vs[i+1]],engine='einsum',reserve=s.labels)
                    vs[i+1].relabel(olds=s.labels,news=[R.replace(qns=s.labels[0].qns)])
                else:
                    ml=v.dotarray(axis=0,array=np.asarray(s))
                lms.append(u)
            for i,(L,S,R) in enumerate(reversed(zip(nbonds[nb-1:2*nb-ob-1],sites[nb-1:2*nb-ob-1],nbonds[nb:2*nb-ob]))):
                m=Tensor(np.asarray(us[-1-i].dotarray(axis=MPS.R,array=nsvs) if i==0 else us[-1-i]),labels=[L,S,R])
                u,s,v=m.svd(row=[L],new=Label('__DMRG_INSERT_R_%i__'%i),col=[S,R],row_signs='+',col_signs='-+')
                if i<len(us)-1:
                    v.relabel(olds=s.labels,news=[L.replace(qns=s.labels[0].qns)])
                    us[-i-2].relabel(olds=[MPS.R],news=[L.replace(qns=s.labels[0].qns)])
                    us[-i-2]=contract([us[-i-2],u,s],engine='einsum',reserve=s.labels)
                    us[-i-2].relabel(olds=s.labels,news=[L.replace(qns=s.labels[0].qns)])
                else:
                    mr=u.dotarray(axis=1,array=np.asarray(s))
                rms.insert(0,v)
            u,s,v=contract([ml,Tensor(1.0/osvs,labels=[nbonds[nb-1]]),mr],engine='einsum').svd(row=[0],new=nbonds[nb-1],col=[1])
            lms[-1]=contract([lms[-1],u],engine='tensordot')
            rms[+0]=contract([v,rms[+0]],engine='tensordot')
            nms=lms+rms
            mps.Lambda=s
        else:
            for L,S,R in zip(nbonds[ob-1:nb-1],sites[ob-1:nb-1],nbonds[ob:nb]):
                nms.append(Tensor(np.zeros((1,S.dim,1),dtype=dtype),labels=[L,S,R]))
            for L,S,R in zip(nbonds[nb-1:2*nb-ob-1],sites[nb-1:2*nb-ob-1],nbonds[nb:2*nb-ob]):
                nms.append(Tensor(np.zeros((1,S.dim,1),dtype=dtype),labels=[L,S,R]))
            nsvs=np.array([1.0])
        mps[mps.cut:mps.cut]=nms
        mps.cut=mps.nsite/2
        mps.relabel(sites=sites,bonds=nbonds)
        return nsvs

    @staticmethod
    def impo_generate(mpo,sites,bonds):
        '''
        Infinite DMRG mpo generation.

        Parameters
        ----------
        mpo : MPO
            The mpo to be generated.
        sites/bonds : list of Label
            The site/bond labels of the mpo.
        '''
        ob,nb,obonds=mpo.nsite/2+1,(len(bonds)+1)/2,mpo.bonds
        L=[bond.replace(qns=obond.qns) for bond,obond in zip(bonds[:ob],obonds[:ob])]
        C=[bond.replace(qns=obond.qns) for bond,obond in zip(bonds[ob:-ob],obonds[ob:nb]*2)]
        R=[bond.replace(qns=obond.qns) for bond,obond in zip(bonds[-ob:],obonds[-ob:])]
        mpo[mpo.nsite/2:mpo.nsite/2]=deepcopy(mpo[2*ob-nb-1:nb-1])
        mpo.relabel(sites=sites,bonds=L+C+R)

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
        with open('%s/%s_%s.dat'%(self.din,pattern(self.status,self.target,self.mps.nsite,mode='py'),self.mps.nmax),'wb') as fout:
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
        Use pickle to laod the core of the dmrg from existing data files.

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
    nspb : integer
        The number of sites per block of the dmrg.
    nmax : integer
        The maximum singular values to be kept.
    npresweep : integer
        The number of presweeps to make a random mps converged to the target state.
    nsweep : integer
        The number of sweeps to make the predicted mps converged to the target state.
    terminate : logical
        True for terminate the growing process if converged target state energy has been obtained, False for not.
    tol : float64
        The tolerance of the target state energy.
    '''

    def __init__(self,targets,nspb,nmax,npresweep=10,nsweep=4,terminate=False,tol=10**-6,**karg):
        '''
        Constructor.

        Parameters
        ----------
        targets : sequence of QuantumNumber
            The target space at each growth of the DMRG.
        nspb : integer
            The number of sites per block of the dmrg.
        nmax : integer
            The maximum number of singular values to be kept.
        npresweep : integer, optional
            The number of presweeps to make a random mps converged to the target state.
        nsweep : integer, optional
            The number of sweeps to make the predicted mps converged to the target state.
        terminate : logical, optional
            True for terminate the growing process if converged target state energy has been obtained, False for not.
        tol : float64, optional
            The tolerance of the target state energy.
        '''
        self.targets=targets
        self.nspb=nspb
        self.nmax=nmax
        self.npresweep=npresweep
        self.nsweep=nsweep
        self.terminate=terminate
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
            core=DMRG.coreload(din=engine.din,pattern=pattern(engine.status,target,(len(self.targets)-i)*self.nspb*2,mode='re'),nmax=self.nmax)
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
    This method iterativey update the DMRG by increasing its lattice in the center by 2 blocks at each iteration.
    '''
    engine.log.open()
    num=app.recover(engine)
    scopes=range(len(app.targets)*2)
    for i,target in enumerate(app.targets[num+1:]):
        pos,nold,nnew=i+num+1,engine.mps.nsite,engine.mps.nsite+2*app.nspb
        engine.insert(scopes[pos],scopes[-pos-1],news=scopes[:pos]+scopes[-pos:] if pos>0 else None,target=target)
        assert nnew==engine.mps.nsite
        geold=engine.info['Esite']
        engine.iterate(info='(++)',sp=True if pos>0 else False,nmax=app.nmax,piechart=app.plot)
        for sweep in xrange(app.npresweep if pos==0 else app.nsweep):
            path=it.chain(['++<<']*((nnew-nold-2)/2),['++>>']*(nnew-nold-2),['++<<']*((nnew-nold-2)/2))
            seold=engine.info['Esite']
            engine.sweep(info=' No.%s'%(sweep+1),path=path,nmax=app.nmax,piechart=app.plot)
            senew=engine.info['Esite']
            if norm(seold-senew)/norm(seold+senew)<app.tol: break
        if app.nspb>1 and pos==0 and app.save_data: engine.coredump()
        genew=engine.info['Esite']
        if app.terminate and geold is not None and norm(geold-genew)/norm(geold+genew)<app.tol: break
    if app.plot and app.save_fig: plt.savefig('%s/%s_.png'%(engine.dout,engine.status))
    if app.save_data: engine.coredump()
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
        The basespace of the DMRG's parametes for the sweeps.
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
        status=deepcopy(engine.status)
        for i,(nmax,parameters) in enumerate(reversed(zip(self.nmaxs,self.BS))):
            status.update(alter=parameters)
            core=DMRG.coreload(din=engine.din,pattern=pattern(status,self.target,self.nsite,mode='re'),nmax=nmax)
            if core:
                for key,value in core.iteritems():
                    setattr(engine,key,value)
                engine.status=status
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
    This method iterativey sweep the DMRG with 2 sites updated at each iteration.
    '''
    engine.log.open()
    num=app.recover(engine)
    if num is None:
        if app.status.name in engine.apps: engine.rundependences(app.status.name)
        num=-1
    for i,(nmax,parameters,path) in enumerate(zip(app.nmaxs[num+1:],app.BS[num+1:],app.paths[num+1:])):
        app.status.update(alter=parameters)
        if not (app.status<=engine.status): engine.update(**parameters)
        engine.sweep(info=' No.%s'%(i+1),path=path,nmax=nmax,piechart=app.plot)
        if app.save_data: engine.coredump()
    if app.plot and app.save_fig: plt.savefig('%s/%s_.png'%(engine.dout,engine.status))
    engine.log.close()
