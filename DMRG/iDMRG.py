'''
=============================================
Infinite density matrix renormalization group
=============================================

iDMRG, including:
    * classes: iDMRG
    * function: iDMRGTSG, iDMRGQP
'''

__all__=['iDMRG','iDMRGTSG','iDMRGQP']

from .DMRG import *
import time
import warnings
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from numpy.linalg import norm
from HamiltonianPy import *
from HamiltonianPy.TensorNetwork import *
from .DMRG import *

class iDMRG(DMRG):
    '''
    Infinite density matrix renormalization group.

    Attributes
    ----------
    niter : int
        The number of iterations.
    '''
    CORE=('niter','lattice','block','cache')

    def __init__(self,tsg,lattice,terms,config,degfres,mask=(),ttype='D',dtype=np.complex128,target=0,**karg):
        '''
        Constructor.

        Parameters
        ----------
        tsg : TSG
            The two-site grow app.
        lattice,terms,config,degfres,mask,ttype,dtype,target :
            See DMRG.__init__ for details.
        '''
        super(iDMRG,self).__init__(lattice,terms,config,degfres,mask,ttype,dtype,target)
        assert isinstance(tsg,TSG)
        self.preload(tsg)
        self.niter=-1

    @property
    def TSG(self):
        '''
        Two-site grow app.
        '''
        return self.apps[self.preloads[0]]

    @property
    def mpo(self):
        '''
        MPO shifted by the last ground state energy.
        '''
        if self.niter<0:
            mpo=MPO()
        elif self.niter<self.lattice.nneighbour+self.DTRP:
            self.cache['optstrs']=[OptStr.fromoperator(operator,self.degfres) for operator in self.generator.operators]
            if self.niter==0:
                self.cache['shiftedsites']=[]
            else:
                sites=self.degfres.labels('S')
                nsite,nspb=len(sites),self.nspb
                sites=sites[nsite//2-min(2,self.niter)*nspb:nsite//2+min(2,self.niter)*nspb]
                if self.block.ttype=='S': sites=[site.replace(qns=site.qns.sorted()) for site in sites]
                if self.niter>1:
                    for i,site in enumerate(it.chain(sites[:nspb],sites[-nspb:])):
                        self.cache['shiftedsites'][-nspb*2+i][0].site=site
                    sites=sites[nspb:3*nspb]
                self.cache['shiftedsites'].extend(OptStr([Opt.identity(site,self.dtype)*(-self.block.info['Esite'])]) for site in sites)
            mpo=OptMPO(self.cache['optstrs']+self.cache['shiftedsites'],self.degfres).tompo(ttype=self.block.ttype)
        else:
            self.cache['optstrs']=[OptStr.fromoperator(operator,self.degfres) for operator in self.generator.operators]
            sites=self.block.mps.sites
            self.cache['shiftedsites']=[OptStr([Opt.identity(site,self.dtype)*(-self.block.info['Esite'])]) for site in sites]
            mpo=OptMPO(self.cache['optstrs']+self.cache['shiftedsites'],self.degfres).tompo(ttype=self.block.ttype)
            mpo=mpo[len(mpo)//2-len(sites)//2:len(mpo)//2+len(sites)//2]
            lold,rold=self.block.mpo[0].labels[MPO.L],self.block.mpo[-1].labels[MPO.R]
            lnew,rnew=mpo[0].labels[MPO.L],mpo[-1].labels[MPO.R]
            assert lnew.equivalent(lold) and rnew.equivalent(rold)
        return mpo

    def update(self,**karg):
        '''
        Update the iDMRG with new parameters.
        '''
        super(iDMRG,self).update(**karg)
        if len(karg)>0 and len(self.generator.operators)>0:
            self.block.reset(mpo=self.mpo,LEND=self.block.lcontracts[0],REND=self.block.rcontracts[-1])

    def resetgenerator(self):
        '''
        Reset the generator of the engine.
        '''
        self.config.reset(pids=self.lattice.pids)
        self.degfres.reset(leaves=self.config.table(mask=self.mask,maps={'scope':iDMRG.scopemap}))
        self.generator.reset(bonds=self.lattice.bonds,config=self.config)

    @staticmethod
    def scopemap(scope):
        '''
        '''
        if isinstance(scope,str):
            assert scope in {'A','B'}
            return (1,) if scope=='A' else (2,)
        else:
            assert isinstance(scope,tuple) and len(scope)==2
            return (0,scope[1]) if scope[0]=='L' else (2,-scope[1])

    def iterate(self,target=None):
        '''
        Iterate the block of the DMRG.

        Parameters
        ----------
        target : QuantumNumber, optional
            The target of the block of the DMRG.
        '''
        self.niter+=1
        osvs=self.cache.get('osvs',np.array([1.0]))
        if self.niter>=self.lattice.nneighbour+self.DTRP:
            self.cache['osvs']=self.block.mps.Lambda.data
            self.block.mpo=self.mpo
            sites=self.block.mps.sites
            obonds=[bond.identifier for bond in self.block.mpo.bonds]
            sbonds=[bond.identifier for bond in self.block.mps.bonds]
            qn=target-self.block.target if isinstance(target,QuantumNumber) else 0
            self.block.predict(sites,obonds,sbonds,osvs,qn)
        else:
            A,B=(('L',0),('R',0)) if self.niter==0 else ('A','B')
            ls=[('L',i) for i in range(self.niter)]
            rs=[('R',i) for i in reversed(range(self.niter))]
            self.lattice.insert(A,B,news=ls+rs)
            self.resetgenerator()
            mpo=self.mpo
            sites,bonds=self.degfres.labels('S'),self.degfres.labels('B')
            if self.block.ttype=='S': sites=[site.replace(qns=site.qns.sorted()) for site in sites]
            osvs=self.cache.get('osvs',np.array([1.0]))
            qn=target-self.block.target if isinstance(target,QuantumNumber) else 0
            self.cache['osvs']=self.block.mps.Lambda.data if self.niter>0 else np.array([1.0])
            mps=self.block.mps.impsgrowth(sites,bonds,osvs,qn,ttype=self.block.ttype)
            self.block.reset(mpo=mpo,mps=mps,target=target)
            if self.niter+1==self.lattice.nneighbour+self.DTRP:
                nsite,nspb=self.block.nsite,self.nspb
                self.block=self.block[nsite//2-nspb:nsite//2+nspb]

def iDMRGTSG(engine,app):
    '''
    This method iterative update the iDMRG (two-site update).

    Parameters
    ----------
    engine : iDMRG
    app : TSG
    '''
    niter=app.recover(engine)
    if niter<0:
        engine.log.open()
        nspb=engine.nspb
        def TSGSWEEP(nsweep,ngrowth):
            assert engine.block.cut==engine.block.nsite/2
            path=list(it.chain(['<<']*(nspb-1),['>>']*(nspb*2-2),['<<']*(nspb-1)))
            for sweep in range(nsweep):
                seold=engine.block.info['Esite']
                engine.sweep(info='No.%s-%s'%(ngrowth+1,sweep+1),path=path,nmax=app.nmax,divisor=2*nspb,piechart=app.plot)
                senew=engine.block.info['Esite']
                if norm(seold-senew)/norm(seold+senew)<app.tol: break
        for i in range(app.maxiter):
            seold=engine.block.info['Esite']
            engine.iterate(target=app.target(getattr(engine,'niter',i-1)+1))
            engine.block.iterate(engine.log,info='%s_%s(%s++)'%(engine,engine.block,i),sp=i>0 or engine.niter>0,divisor=2*nspb,nmax=app.nmax,piechart=app.plot)
            TSGSWEEP(app.npresweep if engine.niter==0 else app.nsweep,i)
            senew=engine.block.info['Esite']
            if i>=app.miniter-1 and seold is not None and norm(seold-senew)/norm(seold+senew)<10*app.tol: break
        else:
            warnings.warn('iDMRGTSG warning: not converged energy after %s iterations.'%app.maxiter)
        if app.plot and app.savefig:
            plt.savefig('%s/%s_%s_%s.png'%(engine.log.dir,engine,engine.block.target,app.name))
            plt.close()
        if app.savedata: engine.dump()
        engine.log.close()

def iDMRGQP(engine,app):
    '''
    This function calculate the pumped charge during an adiabatic process.

    Parameters
    ----------
    engine : iDMRG
    app : QP
    '''
    def pumpedcharge(parameters):
        t1=time.time()
        engine.update(**parameters)
        engine.rundependences(app.name)
        def averagedcharge(mps):
            ps=mps.Lambda.data**2
            qnindex=mps.Lambda.labels[0].qns.type.names.index(app.qnname)
            qns=mps.Lambda.labels[0].qns.expansion()[:,qnindex]
            return ps.dot(qns)/ps.sum()
        result=averagedcharge(engine.block.mps)-getattr(engine.block.target,app.qnname)/2
        t2=time.time()
        engine.log<<'::<parameters>:: %s\n'%(', '.join('%s=%s'%(key,decimaltostr(value)) for key,value in engine.parameters.items()))
        engine.log<<'::<informtation>:: pumped charge=%.6f, time=%.4es\n\n'%(result,t2-t1)
        return result
    result=np.zeros((app.path.rank(0),2))
    for i,parameters in enumerate(app.path('+')):
        result[i,0]=list(parameters.values())[0] if len(parameters)==1 else i
        result[i,1]=pumpedcharge(parameters)
    name='%s_%s'%(engine.tostr(mask=app.path.tags),app.name)
    if app.savedata: np.savetxt('%s/%s.dat'%(engine.dout,name),result)
    if app.plot: app.figure('L',result,'%s/%s'%(engine.dout,name))
    if app.returndata: return result
