'''
=============================================
Infinite density matrix renormalization group
=============================================

iDMRG, including:
    * classes: iDMRG
    * function: iDMRGTSG
'''

__all__=['iDMRG','iDMRGTSG']

from .DMRG import *
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

    def iterate(self,target=None):
        '''
        Init the block of the DMRG.

        Parameters
        ----------
        target : QuantumNumber, optional
            The target of the block of the DMRG.
        '''
        self.niter=getattr(self,'niter',0)+1
        osvs=self.cache.get('osvs',np.array([1.0]))
        if self.niter>self.lattice.nneighbour+self.DTRP:
            self.cache['osvs']=self.block.mps.Lambda.data
            sites=self.block.mps.sites
            obonds=[bond.identifier for bond in self.block.mpo.bonds]
            sbonds=[bond.identifier for bond in self.block.mps.bonds]
            qn=target-self.block.target if isinstance(target,QuantumNumber) else 0
            self.block.predict(sites,obonds,sbonds,osvs,qn)
            self.block.divisor=self.niter
        else:
            ls=['iDMRG_K%s'%i for i in xrange(self.niter-1)]
            rs=['iDMRG_S%s'%i for i in xrange(self.niter-1)]
            self.lattice.insert('iDMRG_L','iDMRG_R',news=ls+rs)
            self.config.reset(pids=self.lattice.pids)
            self.degfres.reset(leaves=self.config.table(mask=self.mask).keys())
            self.generator.reset(bonds=self.lattice.bonds,config=self.config)
            mpo=MPO.fromoperators(self.generator.operators,self.degfres)
            sites,bonds=self.degfres.labels('S'),self.degfres.labels('B')
            osvs=self.cache.get('osvs',np.array([1.0]))
            qn=target-self.block.target if isinstance(target,QuantumNumber) else 0
            self.cache['osvs']=self.block.mps.Lambda.data if self.niter>1 else np.array([1.0])
            mps=self.block.mps.impsgrowth(sites,bonds,osvs,qn)
            self.block.reset(mpo=mpo,mps=mps,target=target,divisor=1)
            if self.niter==self.lattice.nneighbour+self.DTRP:
                nsite,nspb=self.block.nsite,self.nspb
                self.block=self.block[nsite/2-nspb:nsite/2+nspb]
                self.block.divisor=self.niter

def iDMRGTSG(engine,app):
    '''
    This method iterative update the iDMRG (two-site update).
    '''
    engine.log.open()
    num=app.recover(engine)
    nspb=engine.nspb
    def TSGSWEEP(nsweep):
        assert engine.block.cut==engine.block.nsite/2
        path=list(it.chain(['<<']*(nspb-1),['>>']*(nspb*2-2),['<<']*(nspb-1)))
        for sweep in xrange(nsweep):
            seold=engine.block.info['Esite']
            engine.sweep(info='No.%s'%(sweep+1),path=path,nmax=app.nmax,piechart=app.plot)
            senew=engine.block.info['Esite']
            if norm(seold-senew)/norm(seold+senew)<app.tol: break
    for i,target in enumerate(app.targets[num+1:]):
        pos=i+num+1
        engine.iterate(target=target)
        engine.block.iterate(engine.log,info='%s_%s(++)'%(engine,engine.block),sp=True if pos>0 else False,nmax=app.nmax,piechart=app.plot)
        TSGSWEEP(app.npresweep if pos==0 else app.nsweep)
    if num==len(app.targets)-1 and app.nmax>engine.block.mps.nmax: TSGSWEEP(app.nsweep)
    if app.plot and app.savefig:
        plt.savefig('%s/%s_%s_%s.png'%(engine.log.dir,engine,repr(engine.block.target),app.name))
        plt.close()
    if app.savedata: engine.dump()
    engine.log.close()