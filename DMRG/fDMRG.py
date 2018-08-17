'''
===========================================
Finite density matrix renormalization group
===========================================

fDMRG, including:
    * classes: fDMRG
    * function: fDMRGTSG, fDMRGTSS, fDMRGQP
'''

__all__=['fDMRG','fDMRGTSG','fDMRGTSS','fDMRGQP']

import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from numpy.linalg import norm
from HamiltonianPy import *
from HamiltonianPy.TensorNetwork import *
from .DMRG import *

class fDMRG(DMRG):
    '''
    Finite density matrix renormalization group.
    '''

    def __init__(self,tsg=None,tss=None,lattice=None,terms=None,config=None,degfres=None,mask=(),ttype='D',dtype=np.complex128,target=0,**karg):
        '''
        Constructor.

        Parameters
        ----------
        tsg : TSG
            Two-site grow app.
        tss : TSS
            Two-site sweep app.
        lattice,terms,config,degfres,mask,ttype,dtype,target :
            See DMRG.__init__ for details.
        '''
        super(fDMRG,self).__init__(lattice,terms,config,degfres,mask,ttype,dtype,target)
        if isinstance(lattice,Cylinder):
            assert isinstance(tsg,TSG)
            self.preload(tsg)
        assert isinstance(tss,TSS)
        self.preload(tss)

    @property
    def TSG(self):
        '''
        Two-site grow app.
        '''
        return self.apps[self.preloads[0]] if isinstance(self.lattice,Cylinder) else None

    @property
    def TSS(self):
        '''
        Two-site grow app.
        '''
        return self.apps[self.preloads[1 if isinstance(self.lattice,Cylinder) else 0]]

    def update(self,**karg):
        '''
        Update the fDMRG with new parameters.
        '''
        super(fDMRG,self).update(**karg)
        if len(karg)>0:
            self.block.reset(mpo=MPO.fromoperators(self.generator.operators,self.degfres,ttype=self.block.ttype))

    def resetgenerator(self):
        '''
        Reset the generator of the engine.
        '''
        self.config.reset(pids=self.lattice.pids)
        self.degfres.reset(leaves=list(self.config.table(mask=self.mask).keys()))
        self.generator.reset(bonds=self.lattice.bonds,config=self.config)

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
        self.resetgenerator()
        niter=len(self.lattice)//len(self.lattice.block)//2
        sites,obonds,sbonds=self.degfres.labels('S'),self.degfres.labels('O'),self.degfres.labels('B')
        if self.block.ttype=='S': sites=[site.replace(qns=site.qns.sorted()) for site in sites]
        qn=target-self.block.target if isinstance(target,QuantumNumber) else 0
        if niter>self.lattice.nneighbour+self.DTRP:
            osvs=self.cache['osvs']
            self.cache['osvs']=self.block.mps.Lambda.data
            self.block.grow(sites,obonds,sbonds,osvs,qn)
        else:
            mpo=MPO.fromoperators(self.generator.operators,self.degfres,ttype=self.block.ttype)
            osvs=self.cache.get('osvs',np.array([1.0]))
            self.cache['osvs']=self.block.mps.Lambda.data if niter>1 else np.array([1.0])
            mps=self.block.mps.impsgrowth(sites,sbonds,osvs,qn,ttype=self.block.ttype)
            self.block.reset(mpo=mpo,mps=mps,target=target)

def fDMRGTSG(engine,app):
    '''
    This method iterative update the fDMRG by increasing its lattice in the center by 2 blocks at each iteration.

    Parameters
    ---------
    engine : fDMRG
    app : TSG
    '''
    engine.log.open()
    niter=app.recover(engine)
    scopes,nspb=list(range(app.maxiter*2)),engine.nspb
    def TSGSWEEP(nsweep):
        assert engine.block.cut==engine.block.nsite//2
        nold,nnew=engine.block.nsite-2*nspb,engine.block.nsite
        path=list(it.chain(['++<<']*((nnew-nold-2)//2),['++>>']*(nnew-nold-2),['++<<']*((nnew-nold-2)//2)))
        for sweep in range(nsweep):
            seold=engine.block.info['Esite']
            engine.sweep(info='No.%s'%(sweep+1),path=path,nmax=app.nmax,piechart=app.plot)
            senew=engine.block.info['Esite']
            if norm(seold-senew)/norm(seold+senew)<app.tol: break
    for i in range(niter+1,app.maxiter):
        pos=i+niter+1
        engine.insert(scopes[pos],scopes[-pos-1],news=scopes[:pos]+scopes[-pos:] if pos>0 else None,target=app.target(pos))
        engine.block.iterate(engine.log,info='%s_%s(++)'%(engine,engine.block),sp=True if pos>0 else False,nmax=app.nmax,piechart=app.plot)
        TSGSWEEP(app.npresweep if pos==0 else app.nsweep)
        if nspb>1 and app.maxiter>1 and pos==0 and app.savedata: engine.dump()
    if niter==app.maxiter-1 and app.nmax>engine.block.mps.nmax: TSGSWEEP(app.nsweep)
    if app.plot and app.savefig:
        plt.savefig('%s/%s_%s_%s.png'%(engine.log.dir,engine,engine.block.target,app.name))
        plt.close()
    if app.savedata: engine.dump()
    engine.log.close()

def fDMRGTSS(engine,app):
    '''
    This method iterative sweep the fDMRG with 2 sites updated at each iteration.

    Parameters
    ----------
    engine : fDMRG
    app : TSS
    '''
    engine.log.open()
    niter=app.recover(engine)
    if niter is None:
        if app.name in engine.apps: engine.rundependences(app.name)
        niter=-1
    for i,(nmax,path) in enumerate(zip(app.nmaxs[niter+1:],app.paths[niter+1:])):
        engine.sweep(info='No.%s'%(i+1),path=path,nmax=nmax,piechart=app.plot)
        if app.savedata: engine.dump()
    if app.plot and app.savefig:
        plt.savefig('%s/%s_%s_%s.png'%(engine.log.dir,engine,engine.block.target,app.name))
        plt.close()
    engine.log.close()

def fDMRGQP(engine,app):
    '''
    This function calculate the pumped charge during an adiabatic process.

    Parameters
    ----------
    engine : fDMRG
    app : QP
    '''
    count=0
    def pumpedcharge(parameters):
        t1=time.time()
        engine.update(**parameters)
        engine.rundependences(app.name,mask={engine.TSG.name} if count>0 and engine.TSG is not None else None)
        def averagedcharge(mps,statistics):
            ps=mps.Lambda.data**2
            qnindex=mps.Lambda.labels[0].qns.type.names.index(app.qnname)
            qns=mps.Lambda.labels[0].qns.expansion()[:,qnindex]
            if statistics=='f': qns*=(-1)**(qns-1)
            return ps.dot(qns)/ps.sum()
        result=averagedcharge(engine.block.mps,'f' if tuple(engine.mask)==('nambu',) else 'b')-getattr(engine.block.target,app.qnname)/2
        t2=time.time()
        engine.log<<'::<parameters>:: %s\n'%(', '.join('%s=%s'%(key,decimaltostr(value)) for key,value in engine.parameters.items()))
        engine.log<<'::<informtation>:: pumped charge=%.6f, time=%.4es\n\n'%(result,t2-t1)
        return result
    result=np.zeros((app.path.rank(0),2))
    for parameters in app.path('+'):
        result[count,0]=list(parameters.values())[0] if len(parameters)==1 else count
        result[count,1]=pumpedcharge(parameters)
        count+=1
    name='%s_%s'%(engine.tostr(mask=app.path.tags),app.name)
    if app.savedata: np.savetxt('%s/%s.dat'%(engine.dout,name),result)
    if app.plot: app.figure('L',result,'%s/%s'%(engine.dout,name))
    if app.returndata: return result
