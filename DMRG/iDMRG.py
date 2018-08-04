'''
=============================================
Infinite density matrix renormalization group
=============================================

iDMRG, including:
    * classes: iDMRG, QP
    * function: iDMRGTSG, iDMRGQP, iDMRGGSE
'''

__all__=['iDMRG','iDMRGTSG','iDMRGGSE','QP','iDMRGQP']

from .DMRG import *
import time
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

    def __init__(self,tsg,lattice,terms,config,degfres,mask=(),dtype=np.complex128,target=0,**karg):
        '''
        Constructor.

        Parameters
        ----------
        tsg : TSG
            The two-site grow app.
        lattice,terms,config,degfres,mask,dtype,target :
            See DMRG.__init__ for details.
        '''
        super(iDMRG,self).__init__(lattice,terms,config,degfres,mask,dtype,target)
        assert isinstance(tsg,TSG)
        self.preload(tsg)

    @property
    def TSG(self):
        '''
        Two-site grow app.
        '''
        return self.apps[self.preloads[0]]

    def update(self,**karg):
        '''
        Update the iDMRG with new parameters.
        '''
        super(iDMRG,self).update(**karg)
        if len(karg)>0:
            operators=self.generator.operators
            if len(operators)>0:
                mpo=MPO.fromoperators(operators,self.degfres)
                if getattr(self,'niter',0)>=self.lattice.nneighbour+self.DTRP:
                    nsite,nspb=len(mpo),self.nspb
                    mpo=mpo[nsite/2-nspb:nsite/2+nspb]
                self.block.reset(mpo=mpo,LEND=self.block.lcontracts[0],REND=self.block.rcontracts[-1])

    def iterate(self,target=None):
        '''
        Iterate the block of the DMRG.

        Parameters
        ----------
        target : QuantumNumber, optional
            The target of the block of the DMRG.
        '''
        self.niter=getattr(self,'niter',-1)+1
        osvs=self.cache.get('osvs',np.array([1.0]))
        if self.niter>=self.lattice.nneighbour+self.DTRP:
            self.cache['osvs']=self.block.mps.Lambda.data
            sites=self.block.mps.sites
            obonds=[bond.identifier for bond in self.block.mpo.bonds]
            sbonds=[bond.identifier for bond in self.block.mps.bonds]
            qn=target-self.block.target if isinstance(target,QuantumNumber) else 0
            self.block.predict(sites,obonds,sbonds,osvs,qn)
            self.block.divisor=self.niter+1
        else:
            ls=['iDMRG_K%s'%i for i in xrange(self.niter)]
            rs=['iDMRG_S%s'%i for i in xrange(self.niter)]
            self.lattice.insert('iDMRG_L','iDMRG_R',news=ls+rs)
            self.config.reset(pids=self.lattice.pids)
            self.degfres.reset(leaves=self.config.table(mask=self.mask).keys())
            self.generator.reset(bonds=self.lattice.bonds,config=self.config)
            mpo=MPO.fromoperators(self.generator.operators,self.degfres)
            sites,bonds=self.degfres.labels('S'),self.degfres.labels('B')
            osvs=self.cache.get('osvs',np.array([1.0]))
            qn=target-self.block.target if isinstance(target,QuantumNumber) else 0
            self.cache['osvs']=self.block.mps.Lambda.data if self.niter>0 else np.array([1.0])
            mps=self.block.mps.impsgrowth(sites,bonds,osvs,qn)
            self.block.reset(mpo=mpo,mps=mps,target=target,divisor=1)
            if self.niter+1==self.lattice.nneighbour+self.DTRP:
                nsite,nspb=self.block.nsite,self.nspb
                self.block=self.block[nsite/2-nspb:nsite/2+nspb]
                self.block.divisor=self.niter+1

def iDMRGTSG(engine,app):
    '''
    This method iterative update the iDMRG (two-site update).
    '''
    niter=app.recover(engine,'iDMRG')
    if niter<0:
        engine.log.open()
        nspb=engine.nspb
        def TSGSWEEP(nsweep):
            assert engine.block.cut==engine.block.nsite/2
            path=list(it.chain(['<<']*(nspb-1),['>>']*(nspb*2-2),['<<']*(nspb-1)))
            for sweep in xrange(nsweep):
                seold=engine.block.info['Esite']
                engine.sweep(info='No.%s'%(sweep+1),path=path,nmax=app.nmax,piechart=app.plot)
                senew=engine.block.info['Esite']
                if norm(seold-senew)/norm(seold+senew)<app.tol: break
        for i in xrange(app.maxiter):
            engine.iterate(target=app.target(getattr(engine,'niter',i-1)+1))
            engine.block.iterate(engine.log,info='%s_%s(++)'%(engine,engine.block),sp=i>0,nmax=app.nmax,piechart=app.plot)
            TSGSWEEP(app.npresweep if i==0 else app.nsweep)
        if app.plot and app.savefig:
            plt.savefig('%s/%s_%s_%s.png'%(engine.log.dir,engine,engine.block.target,app.name))
            plt.close()
        if app.savedata: engine.dump()
        engine.log.close()

def iDMRGGSE(engine,app):
    '''
    This method calculates the ground state energy.
    '''
    result=np.zeros((app.path.rank(0),2))
    for i,parameters in enumerate(app.path('+')):
        engine.update(**parameters)
        engine.log<<'::<parameters>:: %s\n'%(', '.join('%s=%s'%(key,decimaltostr(value)) for key,value in engine.parameters.iteritems()))
        engine.rundependences(app.name)
        result[i,0]=parameters.values()[0] if len(parameters)==1 else i
        result[i,1]=engine.block.info['Esite']
    name='%s_%s'%(engine.tostr(mask=app.path.tags),app.name)
    if app.savedata: np.savetxt('%s/%s.dat'%(engine.dout,name),result)
    if app.plot: app.figure('L',result,'%s/%s'%(engine.dout,name))
    if app.returndata: return result

class QP(App):
    '''
    Quantum pump of an adiabatic process.

    Attributes
    ----------
    path : BaseSpace
        The path of the varing parameters during the quantum pump.
    qnname : str
        The name of the pumped quantum number.
    nprediction : int
        Number of predictions before the sweep.
    '''

    def __init__(self,path,qnname,nprediction=0,**karg):
        '''
        Constructor.

        Parameters
        ----------
        path : BaseSpace
            The path of the varing parameters during the quantum pump.
        qnname : str
            The name of the pumped quantum number.
        nprediction : int, optional
            Number of predictions before the sweep.
        '''
        self.path=path
        self.qnname=qnname
        self.nprediction=nprediction

def iDMRGQP(engine,app):
    '''
    This function calculate the pumped charge during an adiabatic process.
    '''
    count=0
    def pumpedcharge(parameters):
        t1=time.time()
        engine.update(**parameters)
        if count>0:
            for _ in range(app.nprediction):
                engine.iterate(engine.TSG.target(engine.niter+1))
        engine.rundependences(app.name)
        ps,qns=engine.block.mps.Lambda.data**2,engine.block.mps.Lambda.labels[0].qns
        diff=getattr(engine.block.target,app.qnname)/2
        result=(ps.dot(qns.expansion()[:,qns.type.names.index(app.qnname)])-diff)/ps.sum()
        t2=time.time()
        engine.log<<'::<parameters>:: %s\n'%(', '.join('%s=%s'%(key,decimaltostr(value)) for key,value in engine.parameters.iteritems()))
        engine.log<<'::<informtation>:: pumped charge=%.6f, time=%.4es\n\n'%(result,t2-t1)
        return result
    result=np.zeros((app.path.rank(0),2))
    for parameters in app.path('+'):
        result[count,0]=parameters.values()[0] if len(parameters)==1 else count
        result[count,1]=pumpedcharge(parameters)
        count+=1
    name='%s_%s'%(engine.tostr(mask=app.path.tags),app.name)
    if app.savedata: np.savetxt('%s/%s.dat'%(engine.dout,name),result)
    if app.plot: app.figure('L',result,'%s/%s'%(engine.dout,name))
    if app.returndata: return result