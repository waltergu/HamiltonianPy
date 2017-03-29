'''
DMRG, including:
1) classes: DMRG,TSG,TSS
2) function: pattern,DMRGTSG,DMRGTSS
'''

__all__=['pattern','DMRG','TSG','DMRGTSG','TSS','DMRGTSS']

import os
import re
import numpy as np
import pickle as pk
import scipy.sparse as sp
import HamiltonianPy.Misc as hm
import matplotlib.pyplot as plt
from numpy.linalg import norm
from HamiltonianPy import *
from HamiltonianPy.TensorNetwork import *
from copy import copy,deepcopy

def pattern(status,target,layer,mode='re'):
    '''
    Return the pattern of data files for match.
    Parameters:
        status: Status
            The status of the DMRG.
        target: QuantumNumber
            The target of the DMRG.
        layer: integer
            The layer of the DMRG.
        mode: 're','py'
    Returns: string
        The pattern.
    '''
    assert mode in ('re','py')
    pattern='%s_(%s,%s)'%(status,tuple(target) if isinstance(target,QuantumNumber) else None,layer)
    if mode=='re':
        ss=['(',')','[',']']
        rs=['\(','\)','\[','\]']
        for s,r in zip(ss,rs):
            pattern=pattern.replace(s,r)
    return pattern

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

    def __init__(self,name,mps,lattice,terms,config,degfres,layer=0,mask=[],target=None,dtype=np.complex128,**karg):
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
        self.timers.add(parent='kron',name='csr')
        self.timers.add(parent='kron',name='fkron')
        self.info=Info(['energy','nbasis','subslice','nnz','nz','density','overlap','err'])
        self.cache={}

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
                self.mpo.compress(nsweep=1,options=dict(method='dpl',tol=hm.TOL))

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

    def two_site_step(self,job='sweep',nmax=200,tol=hm.TOL):
        '''
        The two site dmrg step.
        Parameters:
            job: 'sweep','grow'
                'sweep' for two site sweep and 'grow' for two site grow.
            nmax: integer
                The maximum singular values to be kept.
            tol: float64
                The tolerance of the singular values.
        '''
        assert job in ('sweep','grow')
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
                self.info['subslice']=len(subslice)
            else:
                sysqns,syspt=np.product([La.qns,Sa.qns]),None
                envqns,envpt=np.product([Sb.qns,Rb.qns]),None
                sysantipt,envantipt=None,None
                subslice=slice(None)
                rcs=None
                qns=sysqns*envqns
                self.info['subslice']=qns
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
            self.info['nz']='%1.1f%%'%(len(np.argwhere(np.abs(matrix.data)<tol))*100.0/matrix.nnz)
        with self.timers.get('Diagonalization'):
            if job=='sweep':
                u,s,v=self.mps[self.mps.cut-1],self.mps.Lambda,self.mps[self.mps.cut]
                v0=np.asarray(contract([u,s,v],engine='einsum').merge(([La,Sa],Lsys,syspt),([Sb,Rb],Lenv,envpt))).reshape(-1)[subslice]
            else:
                assert self.mps.cut==self.mps.nsite/2
                v0=None
                if self.mps.nsite>=6:
                    u,nsvs,osvs,v=self.mps[self.mps.cut-2],np.asarray(self.mps.Lambda),self.cache['osvs'],self.mps[self.mps.cut+1]
                    ml=np.asarray(v.dotarray(axis=MPS.L,array=nsvs).merge(([MPS.L,MPS.S],Lsys,syspt)))
                    mr=np.asarray(u.dotarray(axis=MPS.R,array=nsvs).merge(([MPS.S,MPS.R],Lenv,envpt)))
                    v0=np.einsum('ij,j,jk->ik',ml,1.0/osvs,mr).reshape(-1)[subslice]
            es,vs=hm.eigsh(matrix,which='SA',v0=v0,k=1)
            energy,Psi=es[0],vs[:,0]
            self.info['energy']=energy/self.mps.nsite
            self.info['overlap']=None if v0 is None else '%1.8f'%np.abs(Psi.conjugate().dot(v0)/norm(v0)/norm(Psi))
        with self.timers.get('Truncation'):
            u,s,v,err=Tensor(Psi,labels=[Label('__DMRG_TWO_SITE_STEP__',qns=qns)]).partitioned_svd(Lsys,new,Lenv,nmax=nmax,tol=tol,return_truncation_err=True)
            self.mps[self.mps.cut-1]=u.split((Lsys,[La,Sa],sysantipt))
            self.mps[self.mps.cut]=v.split((Lenv,[Sb,Rb],envantipt))
            self.cache['osvs']=np.asarray(self.mps.Lambda)
            self.mps.Lambda=s
            self.set_HL_(self.mps.cut-1,tol=tol)
            self.set_HR_(self.mps.cut,tol=tol)
            self.info['nbasis']=len(s)
            self.info['density']='%.2e'%(1.0*self.info['nnz']/self.info['subslice']**2)
            self.info['err']='%.2e'%err

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

    def coredump(self):
        '''
        Use pickle to dump the core of the dmrg.
        '''
        with open('%s/%s_%s.dat'%(self.din,pattern(self.status,self.target,self.layer,mode='py'),self.mps.status),'wb') as fout:
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
    def coreload(din,pattern,nsite,nmax):
        '''
        Use pickle to laod the core of the dmrg from existing data files.
        Parameters:
            din: string
                The directory where the data files are searched.
            pattern: string
                The matching pattern of the data files.
            nsite: integer
                The length of the mps.
            nmax: integer
                The maximum number of singular values kept in the mps.
        Returns: dict
            The loaded core of the dmrg.
        '''
        candidates={}
        names=[name for name in os.listdir(din) if re.match(pattern,name)]
        for name in names:
            split=name.split('_')
            cnsite,cnmax=int(split[-2]),int(split[-1][0:-4])
            if cnsite==nsite and cnmax<=nmax:candidates[name]=(cnsite,cnmax)
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

    def __init__(self,block,vector,scopes,targets,nmax=200,tol=hm.TOL,**karg):
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

    def iterlattices(self):
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

    def lattices(self):
        '''
        Return a list over the sequence of the SuperLattices of the DMRG with blocks added two-by-two into the center.
        '''
        return list(self.iterlattices())

def DMRGTSG(engine,app):
    '''
    This method iterativey update the DMRG by increasing its lattice in the center by 2 blocks at each iteration.
    '''
    engine.log.open()
    engine.layer=0
    for i,target in enumerate(reversed(app.targets)):
        core=DMRG.coreload(din=engine.din,pattern=pattern(engine.status,target,engine.layer,mode='re'),nsite=(len(app.targets)-i)*2,nmax=app.nmax)
        if core:
            for key,value in core.iteritems():
                setattr(engine,key,value)
            engine.config.reset(pids=engine.lattice)
            engine.degfres.reset(leaves=engine.config.table(mask=engine.mask).keys())
            engine.generator.reset(bonds=engine.lattice.bonds,config=engine.config)
            num=len(app.targets)-i-1
            break
        else:
            num=-1
    for lattice,target in zip(app.lattices()[num+1:],app.targets[num+1:]):
        engine.lattice=lattice
        engine.config.reset(pids=engine.lattice)
        engine.degfres.reset(leaves=engine.config.table(mask=engine.mask).keys())
        engine.generator.reset(bonds=engine.lattice.bonds,config=engine.config)
        engine.set_operators_mpo()
        if engine.mps.cut is None: engine.mps.cut=0
        assert engine.mps.cut==engine.mps.nsite/2
        layer=engine.degfres.layers[engine.layer]
        sites,bonds=engine.degfres.labels(layer,'S'),engine.degfres.labels(layer,'B')
        obonds,nbonds,diff=engine.mps.bonds,[],0 if target is None else (target if engine.target is None else target-engine.target)
        for i,bond in enumerate(bonds):
            if i==0:
                nbonds.append(bond.replace(qns=QuantumNumbers.mono(target.zeros(),count=1)))
            elif i==len(bonds)-1:
                nbonds.append(bond.replace(qns=QuantumNumbers.mono(target,count=1)))
            elif i<engine.mps.cut+1:
                nbonds.append(bond.replace(qns=obonds[i].qns))
            elif i>engine.mps.cut+1:
                nbonds.append(bond.replace(qns=obonds[i-2].qns+diff))
            else:
                nbonds.append(bond.replace(qns=None))
        engine.mps.insert(engine.mps.cut,Tensor([[[0.0]]],labels=[nbonds[engine.mps.cut-1],sites[engine.mps.cut-1],nbonds[engine.mps.cut]]))
        engine.mps.insert(engine.mps.cut+1,Tensor([[[0.0]]],labels=[nbonds[engine.mps.cut],sites[engine.mps.cut],nbonds[engine.mps.cut+1]]))
        engine.mps.relabel(sites=sites,bonds=nbonds)
        engine.mps.cut+=1
        engine._Hs_["L"].extend([None,None])
        engine._Hs_["R"].extend([None,None])
        engine.set_Hs_(L=engine.mps.cut-2,R=engine.mps.cut+1,tol=app.tol)
        engine.target=target
        engine.log<<'%s(target=%s,layer=%s,nsite=%s,cut=%s)(++)\n'%(engine.status,engine.target,engine.layer,engine.mps.nsite,engine.mps.cut)
        engine.log<<engine.graph<<'\n'
        engine.log<<'-'.join(str(bond.dim) for bond in engine.mpo.bonds)<<'\n'
        engine.two_site_step(job='grow',nmax=app.nmax,tol=app.tol)
        engine.timers.record()
        engine.log<<'timers of the dmrg:\n'<<engine.timers.tostr(None)<<'\n'
        engine.log<<'info of the dmrg:\n'<<engine.info<<'\n\n'
        if app.plot: engine.timers.graph(parents=Timers.ALL)
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

def DMRGTSS(engine,app):
    '''
    This method iterativey sweep the DMRG with 2 sites updated at each iteration.
    '''
    engine.log.open()
    status=deepcopy(engine.status)
    for i,(nmax,parameters) in enumerate(reversed(zip(app.nmaxs,app.BS))):
        status.update(alter=parameters)
        core=DMRG.coreload(din=engine.din,pattern=pattern(status,app.target,app.layer,mode='re'),nsite=app.nsite,nmax=nmax)
        if core:
            for key,value in core.iteritems():
                setattr(engine,key,value)
            engine.status=status
            engine.config.reset(pids=engine.lattice)
            engine.degfres.reset(leaves=engine.config.table(mask=engine.mask).keys())
            num=len(app.nmaxs)-1-i
            if app.protocal==1 or engine.mps.status['nmax']<nmax: num-=1
            break
    else:
        engine.rundependences(app.status.name)
        num=-1
    if engine.layer!=app.layer: engine.relayer(layer=app.layer,cut=engine.mps.nsite/2,tol=app.tol)
    for i,(nmax,parameters,path) in enumerate(zip(app.nmaxs[num+1:],app.BS[num+1:],app.paths[num+1:])):
        suffix='st'if i==0 else ('nd' if i==1 else ('rd' if i==2 else 'th'))
        app.status.update(alter=parameters)
        if not (app.status<=engine.status): engine.update(**parameters)
        if path is None:
            if engine.mps.cut==0:
                engine.mps>>=1
            elif engine.mps.cut==engine.mps.nsite:
                engine.mps<<=1
            path=['<<']*(engine.mps.cut-1)+['>>']*(engine.mps.nsite-2)+['<<']*(engine.mps.nsite-engine.mps.cut-1)
        for move in path:
            if move=='<<':
                engine.mps<<=1
            else:
                engine.mps>>=1
            engine.log<<'%s(target=%s,layer=%s,nsite=%s,cut=%s) %s%s(%s)\n'%(engine.status,engine.target,engine.layer,app.nsite,engine.mps.cut,i+1,suffix,move)
            engine.log<<engine.graph<<'\n'
            engine.log<<'-'.join(str(bond.dim) for bond in engine.mpo.bonds)<<'\n'
            engine.two_site_step(job='sweep',nmax=nmax,tol=app.tol)
            engine.timers.record()
            engine.log<<'timers of the dmrg:\n%s\n'%engine.timers.tostr(None)
            engine.log<<'info of the dmrg:\n%s\n\n'%engine.info
            if app.plot: engine.timers.graph(parents=Timers.ALL)
        if app.save_data: engine.coredump()
    if app.plot and app.save_fig: plt.savefig('%s/%s_.png'%(engine.dout,engine.status))
    engine.log.close()
