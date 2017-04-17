'''
DMRG, including:
1) classes: Cylinder,DMRG,TSG,TSS
2) function: pattern,DMRGTSG,DMRGTSS
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

def pattern(status,target,layer,nsite,mode='re'):
    '''
    Return the pattern of data files for match.
    Parameters:
        status: Status
            The status of the DMRG.
        target: QuantumNumber
            The target of the DMRG.
        layer: integer
            The layer of the DMRG.
        nsite: integer
            The number of sites of the DMRG.
        mode: 're','py'
    Returns: string
        The pattern.
    '''
    assert mode in ('re','py')
    result='%s_(%s,%s,%s)'%(status,tuple(target) if isinstance(target,QuantumNumber) else None,layer,nsite)
    if mode=='re':
        ss=['(',')','[',']']
        rs=['\(','\)','\[','\]']
        for s,r in zip(ss,rs):
            result=result.replace(s,r)
    return result

class Cylinder(Lattice):
    '''
    The cylinder geometry of a lattice.
    Attribues:
        block: list of 1d ndarray
            The building block of the cylinder.
        translation: 1d ndarray
            The translation vector of the building block to construct the cylinder.
    '''

    def __init__(self,block,translation,**karg):
        '''
        Constructor.
        Parameters:
            block: list of 1d ndarray
                The building block of the cylinder.
            translation: 1d ndarray
                The translation vector of the building block to construct the cylinder.
        '''
        super(Cylinder,self).__init__(**karg)
        self.block=block
        self.translation=translation

    def insert(self,A,B):
        '''
        Insert two blocks into the center of the cylinder.
        Parameters:
            A,B: any hashable object
                The scopes of the insert block points.
        '''
        if len(self)==0:
            aps=[Point(PID(scope=A,site=i),rcoord=rcoord-self.translation/2,icoord=np.zeros_like(rcoord)) for i,rcoord in enumerate(self.block)]
            bps=[Point(PID(scope=B,site=i),rcoord=rcoord+self.translation/2,icoord=np.zeros_like(rcoord)) for i,rcoord in enumerate(self.block)]
        else:
            aps,bps=self.points[:len(self)/2],self.points[len(self)/2:]
            for ap,bp in zip(aps,bps):
                ap.rcoord-=self.translation
                bp.rcoord+=self.translation
            aps.extend(Point(PID(scope=A,site=i),rcoord=rcoord-self.translation/2,icoord=np.zeros_like(rcoord)) for i,rcoord in enumerate(self.block))
            bps.extend(Point(PID(scope=B,site=i),rcoord=rcoord+self.translation/2,icoord=np.zeros_like(rcoord)) for i,rcoord in enumerate(self.block))
        self.points=aps+bps
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
        Parameters:
            scopes: list of hashable object
                The scopes of the cylinder.
        Returns: Lattice
            The constructed cylinder.
        '''
        points,num=[],len(scopes)
        for i,rcoord in enumerate(tiling(self.block,vectors=[self.translation],translations=np.linspace(-(num-1)/2.0,(num-1)/2.0,num))):
            points.append(Point(PID(scope=scopes[i/num],site=i%num),rcoord=rcoord,icoord=np.zeros_like(rcoord)))
        return Lattice.compose(name=self.name,points=points,vectors=self.vectors,nneighbour=self.nneighbour)

class DMRG(Engine):
    '''
   Density matrix renormalization group method.
    Attribues:
        mps: MPS
            The matrix product state of the DMRG.
        lattice: Cylinder/Lattice
            The lattice of the DMRG.
        terms: list of Term
            The terms of the DMRG.
        config: IDFConfig
            The configuration of the internal degrees of freedom on the lattice.
        degfres: DegFreTree
            The physical degrees of freedom tree.
        layer: integer
            The layer on which the DMRG works.
        mask: [] or ['nambu']
            [] for spin systems and ['nambu'] for fermionic systems.
        target: QuantumNumber
            The target space of the DMRG.
        dtype: np.float64, np.complex128
            The data type.
        generator: Generator
            The generator of the Hamiltonian.
        operators: OperatorCollection
            The operators of the Hamiltonian.
        mpo: MPO
            The MPO-formed Hamiltonian.
        _Hs_: dict
            entry 'L': list of 3d Tensor
                The contraction of mpo and mps from the left.
            entry 'R': list of 3d Tensor
                The contraction of mpo and mps from the right.
        timers: Timers
            The timers of the dmrg processes.
        info: Info
            The info of the dmrg processes.
        cache: dict
            entry 'osvs': 1d ndarray
                The old singular values of the DMRG.
    '''

    def __init__(self,mps,lattice,terms,config,degfres,layer=0,mask=[],target=None,dtype=np.complex128,**karg):
        '''
        Constructor.
        Parameters:
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
            layer: integer
                The layer on which the DMRG works.
            mask: [] or ['nambu']
                [] for spin systems and ['nambu'] for fermionic systems.
            target: QuantumNumber
                The target space of the DMRG.
            dtype: np.float64,np.complex128, optional
                The data type.
        '''
        self.mps=mps
        self.lattice=lattice
        self.terms=terms
        self.config=config
        self.degfres=degfres
        self.layer=layer
        self.mask=mask
        self.target=target
        self.dtype=dtype
        self.generator=Generator(bonds=lattice.bonds,config=config,terms=terms,dtype=dtype)
        self.status.update(const=self.generator.parameters['const'])
        self.status.update(alter=self.generator.parameters['alter'])
        self.set_operators_mpo()
        self.set_Hs_()
        self.timers=Timers('Preparation','Hamiltonian','Diagonalization','Truncation')
        self.timers.add(parent='Hamiltonian',name='kron')
        self.timers.add(parent='Hamiltonian',name='sum')
        if self.mps.mode=='QN':
            self.timers.add(parent='kron',name='csr')
            self.timers.add(parent='kron',name='fkron')
        self.info=Info('Etotal','Esite','nbasis','nslice','nnz','nz','density','overlap','err')
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
        return '%s(target=%s,layer=%s,nsite=%s,cut=%s)'%(self.status,self.target,self.layer,self.mps.nsite,self.mps.cut)

    def update(self,**karg):
        '''
        Update the DMRG with new parameters.
        '''
        self.generators['h'].update(**karg)
        self.status.update(alter=self.generators['h'].parameters['alter'])
        self.set_operators_mpo()
        self.set_Hs_()

    def set_operators_mpo(self):
        '''
        Set the generators, operators and optstrs of the DMRG.
        '''
        self.operators=self.generator.operators
        if self.mask==['nambu']:
            for operator in self.operators.values()[:]:
                self.operators+=operator.dagger
        for i,operator in enumerate(self.operators.itervalues()):
            if i==0:
                self.mpo=OptStr.from_operator(operator,self.degfres,self.layer).to_mpo(self.degfres)
            else:
                self.mpo+=OptStr.from_operator(operator,self.degfres,self.layer).to_mpo(self.degfres)
            if i%20==0 or i==len(self.operators)-1:
                self.mpo.compress(nsweep=1,method='dpl',options=dict(tol=hm.TOL))

    def set_HL_(self,pos,tol=hm.TOL):
        '''
        Set a certain left block Hamiltonian.
        Parameters:
            pos: integer
                The position of the left block Hamiltonian.
            tol: np.float64, optional
                The tolerance of the non-zeros.
        '''
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

    def set_HR_(self,pos,tol=hm.TOL):
        '''
        Set a certain right block Hamiltonian.
        Parameters:
            pos: integer
                The position of the right block Hamiltonian.
            tol: np.float64, optional
                The tolerance of the non-zeros.
        '''
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

    def set_Hs_(self,L=None,R=None,tol=hm.TOL):
        '''
        Set the Hamiltonians of blocks.
        Parameters:
            L: integer, optional
                The maximum position of the left Hamiltonians to be set.
            R: integer, optional
                The maximum position of the right Hamiltonians to be set.
            tol: np.float64, optional
                The tolerance of the zeros.
        '''
        self._Hs_={'L':[None]*(self.mps.nsite+1),'R':[None]*(self.mps.nsite+1)}
        if self.mps.cut is not None:
            L=self.mps.cut-1 if L is None else L
            R=self.mps.cut if R is None else R
            for pos in xrange(-1,L+1):
                self.set_HL_(pos,tol=tol)
            for pos in xrange(self.mps.nsite,R-1,-1):
                self.set_HR_(pos,tol=tol)

    def two_site_step(self,sp=True,nmax=200,tol=hm.TOL,piechart=True):
        '''
        The two site dmrg step.
        Parameters:
            sp: logical, optional
                True for state prediction False for not.
            nmax: integer, optional
                The maximum singular values to be kept.
            tol: np.float64, optional
                The tolerance of the singular values.
            piechart: logical, optional
                True for showing the piechart of self.timers while False for not.
        '''
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
                subslice=QuantumNumbers.kron([sysqns,envqns],signs='+-').subslice(targets=(self.target.zeros(),))
                rcs=subslice
                qns=QuantumNumbers.mono(self.target.zeros(),count=len(subslice))
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
            self.info['nz']=len(np.argwhere(np.abs(matrix.data)<tol))*100.0/matrix.nnz,'%1.1f%%'
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
        self.log<<'timers of the dmrg:\n%s\n'%self.timers.tostr(None)
        self.log<<'info of the dmrg:\n%s\n\n'%self.info
        if piechart: self.timers.graph(parents=Timers.ALL)

    def relayer(self,layer,nsweep=1,cut=0,nmax=None,tol=None):
        '''
        Change the layer of the physical degrees of freedom.
        Parameters:
            layer: integer/tuple-of-string
                The new layer.
            nsweep: integer, optional
                The number of sweeps to compress the new mps and mpo.
            cut: integer, optional
                The position of the connecting bond of the new mps.
            nmax: integer, optional
                The maximum number of singular values to be kept.
            tol: np.float64, optional
                The tolerance of the singular values.
        '''
        self.layer=layer if type(layer) in (int,long) else self.degfres.layers.index(layer)
        self.set_operators_mpo()
        self.mpo.compress(nsweep=nsweep,options=dict(method='dpl',tol=hm.TOL))
        self.mps=self.mps.relayer(self.degfres,layer,nmax=nmax,tol=tol)
        self.mps.compress(nsweep=nsweep,cut=cut,nmax=nmax,tol=tol)
        self.set_Hs_(tol=tol)

    def insert(self,A,B,target=None,mps0=None):
        '''
        Insert two blocks of points into the center of the lattice.
        Parameters:
            A,B: any hashable object
                The scopes of the insert block points.
            target: QuantumNumber, optional
                The new target of the DMRG.
            mps0: MPS, optional
                The initial mps.
        '''
        self.lattice.insert(A,B)
        self.config.reset(pids=self.lattice.pids)
        self.degfres.reset(leaves=self.config.table(mask=self.mask).keys())
        self.generator.reset(bonds=self.lattice.bonds,config=self.config)
        self.set_operators_mpo()
        layer=self.degfres.layers[self.layer]
        sites,bonds=self.degfres.labels(layer,'S'),self.degfres.labels(layer,'B')
        if mps0 is None:
            if self.mps.cut is None: self.mps.cut=0
            assert self.mps.cut==self.mps.nsite/2 and self.mps.nsite%2==0
            obonds=self.mps.bonds
            diff=0 if target is None else (target if self.target is None else target-self.target)
            ob,nb=(len(obonds)+1)/2 if self.mps.nsite>0 else 1,(len(bonds)+1)/2
            L,LS,RS,R=bonds[:ob],bonds[ob:nb],bonds[nb:-ob],bonds[-ob:]
            if self.mps.cut==0:
                L[+0]=L[+0].replace(qns=QuantumNumbers.mono(target.zeros(),count=1) if self.mps.mode=='QN' else 1)
                R[-1]=R[-1].replace(qns=QuantumNumbers.mono(target,count=1) if self.mps.mode=='QN' else 1)
            for i,(bond,obond) in enumerate(zip(L,obonds[:ob])):
                L[i]=bond.replace(qns=obond.qns)
            for i,(bond,obond) in enumerate(zip(R,obonds[-ob:])):
                R[i]=bond.replace(qns=obond.qns+diff)
            if self.mps.nsite>0:
                for i,(bond,obond) in enumerate(zip(LS,obonds[ob:nb])):
                    LS[i]=bond.replace(qns=obond.qns)
                for i,(bond,obond) in enumerate(zip(RS,obonds[-nb+1:-ob])):
                    RS[i]=bond.replace(qns=obond.qns+diff)
            else:
                LS,RS=deepcopy(LS),deepcopy(RS)
            nbonds=L+LS+RS+R
            ns,nms,lms,rms=nb-ob,[],[],[]
            if len(self.lattice)/len(self.lattice.block)>2:
                us,vs=self.mps[self.mps.cut-ns:self.mps.cut],self.mps[self.mps.cut:self.mps.cut+ns]
                nsvs,osvs=np.asarray(self.mps.Lambda),self.cache.get('osvs',np.array([1.0]))
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
                self.cache['osvs']=np.asarray(self.mps.Lambda)
                self.mps.Lambda=s
            else:
                for L,S,R in zip(nbonds[ob-1:nb-1],sites[ob-1:nb-1],nbonds[ob:nb]):
                    nms.append(Tensor(np.zeros((1,S.dim,1),dtype=self.dtype),labels=[L,S,R]))
                for L,S,R in zip(nbonds[nb-1:2*nb-ob-1],sites[nb-1:2*nb-ob-1],nbonds[nb:2*nb-ob]):
                    nms.append(Tensor(np.zeros((1,S.dim,1),dtype=self.dtype),labels=[L,S,R]))
            self.mps[self.mps.cut:self.mps.cut]=nms
            self.mps.relabel(sites=sites,bonds=nbonds)
            self.mps.cut=self.mps.nsite/2
        else:
            assert self.mps.nsite==0 and mps0.nsite==len(self.mpo) and mps0.cut==mps0.nsite/2
            self.mps=mps0
            self.mps.relabel(sites=sites,bonds=[bond.replace(qns=obond.qns) for bond,obond in zip(bonds,mps0.bonds)])
        self.set_Hs_(L=self.mps.cut-2,R=self.mps.cut+1)
        self.target=target

    def coredump(self):
        '''
        Use pickle to dump the core of the dmrg.
        '''
        with open('%s/%s_%s.dat'%(self.din,pattern(self.status,self.target,self.layer,self.mps.nsite,mode='py'),self.mps.nmax),'wb') as fout:
            core=   {
                        'lattice':      self.lattice,
                        'layer':        self.layer,
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
        Parameters:
            din: string
                The directory where the data files are searched.
            pattern: string
                The matching pattern of the data files.
            nmax: integer
                The maximum number of singular values kept in the mps.
        Returns: dict
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
    Attribues:
        scopes: list of hashable objects
            The scopes of the blocks to be added two-by-two into the lattice of the DMRG.
        targets: sequence of QuantumNumber
            The target space at each growth of the DMRG.
        mps0: MPS
            The initial mps.
        layer: integer
            The layer of the DMRG's mps.
        ns: integer
            The number of sites per block of the dmrg.
        nmax: integer
            The maximum singular values to be kept.
        tol: float64
            The tolerance of the singular values.
        heatbaths: list of integer
            The maximum bond dimensions for the heat bath processes.
        sweeps: list of integer
            The maximum bond dimensions for the sweep processes.
    '''

    def __init__(self,scopes,targets,mps0=None,layer=0,ns=1,nmax=200,tol=hm.TOL,heatbaths=None,sweeps=None,**karg):
        '''
        Constructor.
        Parameters:
            scopes: list of hashable objects
                The scopes of the blocks to be added two-by-two into the lattice of the DMRG.
            targets: sequence of QuantumNumber
                The target space at each growth of the DMRG.
            mps0: MPS, optional
                The initial mps.
            layer: integer, optional
                The layer of the DMRG's mps.
            ns: integer, optional
                The number of sites per block of the dmrg.
            nmax: integer, optional
                The maximum number of singular values to be kept.
            tol: float64, optional
                The tolerance of the singular values.
            heatbaths: list of integer, optional
                The maximum bond dimensions for the heat bath processes.
            sweeps: list of integer, optional
                The maximum bond dimensions for the sweep processes.
        '''
        assert len(scopes)==len(targets)*2
        self.scopes=scopes
        self.targets=targets
        self.mps0=mps0
        self.layer=layer
        self.ns=ns
        self.nmax=nmax
        self.tol=tol
        self.heatbaths=[nmax*(i+1)/15 for i in xrange(15)] if heatbaths is None else heatbaths
        self.sweeps=[nmax]*4 if sweeps is None else sweeps

    def recover(self,engine):
        '''
        Recover the core of a dmrg engine.
        Parameters:
            engine: DMRG
                The dmrg engine whose core is to be recovered.
        Returns: integer
            The recover code.
        '''
        for i,target in enumerate(reversed(self.targets)):
            core=DMRG.coreload(din=engine.din,pattern=pattern(engine.status,target,self.layer,(len(self.targets)-i)*self.ns*2,mode='re'),nmax=self.nmax)
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
    if engine.layer!=app.layer: engine.layer=app.layer
    for pos,target in enumerate(app.targets[num+1:]):
        nold=engine.mps.nsite
        engine.insert(app.scopes[pos+num+1],app.scopes[-pos-num-2],target=target,mps0=app.mps0 if num+pos<0 else None)
        nnew=engine.mps.nsite
        engine.log<<'%s(++)\n%s\n'%(engine.state,engine.graph)
        engine.two_site_step(sp=True if num+pos>=0 else False,nmax=app.nmax,tol=app.tol,piechart=app.plot)
        for i,nmax in enumerate(app.heatbaths if num+pos==-1 else app.sweeps):
            for move in it.chain(['++<<']*((nnew-nold-2)/2),['++>>']*(nnew-nold-2),['++<<']*((nnew-nold-2)/2)):
                engine.mps<<=1 if move=='++<<' else -1
                engine.log<<'%s No.%s(%s)\n%s\n'%(engine.state,i+1,move,engine.graph)
                engine.two_site_step(sp=True,nmax=nmax,tol=app.tol,piechart=app.plot)
    if app.plot and app.save_fig: plt.savefig('%s/%s_.png'%(engine.dout,engine.status))
    engine.log.close()
    if app.save_data: engine.coredump()

class TSS(App):
    '''
    Two site sweep of a DMRG.
    Attribues:
        target: QuantumNumber
            The target of the DMRG's mps.
        layer: integer
            The layer of the DMRG's mps.
        nsite: integer
            The length of the DMRG's mps.
        protocal: 0,1
            Before sweeps, the core of the dmrg will be recovered from exsiting data files.
            When protocal==0, no real sweep will be taken when the recovered mps perfectly matches the recovering rule;
            When protocal==1, the sweep will be taken at least once even if the recovered mps perfectly matches the recovering rule.
        BS: BaseSpace
            The basespace of the DMRG's parametes for the sweeps.
        nmaxs: list of integer
            The maximum numbers of singular values to be kept for the sweeps.
        paths: list of list of '<<' or '>>'
            The paths along which the sweeps are performed.
        tol: np.float64
            The tolerance of the singular values.
    '''

    def __init__(self,target,layer,nsite,nmaxs,protocal=0,BS=None,paths=None,tol=hm.TOL,**karg):
        '''
        Constructor.
        Parameters:
            target: QuantumNumber
                The target of the DMRG's mps.
            layer: integer
                The layer of the DMRG's mps.
            nsite: integer
                The length of the DMRG's mps.
            nmaxs: list of integer
                The maximum numbers of singular values to be kept for each sweep.
            protocal: 0, 1
                The protocal of the app to carry out the sweep.
            BS: list of dict, optional
                The parameters of the DMRG for each sweep.
            paths: list of list of '<<' or '>>', optional
                The paths along which the sweeps are performed.
            tol: np.float64, optional
                The tolerance of the singular values.
        '''
        self.target=target
        self.layer=layer
        self.nsite=nsite
        self.nmaxs=nmaxs
        self.protocal=protocal
        self.BS=[{}]*len(nmaxs) if BS is None else BS
        self.paths=[None]*len(nmaxs) if paths is None else paths
        assert len(nmaxs)==len(self.BS) and len(nmaxs)==len(self.paths)
        self.tol=tol

    def recover(self,engine):
        '''
        Recover the core of a dmrg engine.
        Parameters:
            engine: DMRG
                The dmrg engine whose core is to be recovered.
        Returns: integer
            The recover code.
        '''
        status=deepcopy(engine.status)
        for i,(nmax,parameters) in enumerate(reversed(zip(self.nmaxs,self.BS))):
            status.update(alter=parameters)
            core=DMRG.coreload(din=engine.din,pattern=pattern(status,self.target,self.layer,self.nsite,mode='re'),nmax=nmax)
            if core:
                for key,value in core.iteritems():
                    setattr(engine,key,value)
                engine.status=status
                engine.config.reset(pids=engine.lattice.pids)
                engine.degfres.reset(leaves=engine.config.table(mask=engine.mask).keys())
                engine.mps>>=1 if engine.mps.cut==0 else (-1 if engine.mps.cut==engine.mps.nsite else 0)
                code=len(self.nmaxs)-1-i
                if self.protocal==1 or engine.mps.nmax<nmax: code-=1
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
        if engine.layer!=app.layer: engine.relayer(layer=app.layer,cut=app.nsite/2,tol=app.tol)
        num=-1
    for i,(nmax,parameters,path) in enumerate(zip(app.nmaxs[num+1:],app.BS[num+1:],app.paths[num+1:])):
        app.status.update(alter=parameters)
        if not (app.status<=engine.status): engine.update(**parameters)
        for move in it.chain(['<<']*(engine.mps.cut-1),['>>']*(engine.mps.nsite-2),['<<']*(engine.mps.nsite-engine.mps.cut-1)) if path is None else path:
            engine.mps<<=1 if move=='<<' else -1
            engine.log<<'%s No.%s(%s)\n%s\n'%(engine.state,i+1,move,engine.graph)
            engine.two_site_step(sp=True,nmax=nmax,tol=app.tol,piechart=app.plot)
        if app.save_data: engine.coredump()
    if app.plot and app.save_fig: plt.savefig('%s/%s_.png'%(engine.dout,engine.status))
    engine.log.close()
