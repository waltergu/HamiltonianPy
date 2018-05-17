'''
===========================================
Finite density matrix renormalization group
===========================================

fDMRG, including:
    * classes: fDMRG
    * function: fDMRGTSG, fDMRGTSS
'''

__all__=['fDMRG','fDMRGTSG','fDMRGTSS']

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
        niter=len(self.lattice)/len(self.lattice.block)/2
        sites,obonds,sbonds=self.degfres.labels('S'),self.degfres.labels('O'),self.degfres.labels('B')
        qn=target-self.block.target if isinstance(target,QuantumNumber) else 0
        if niter>self.lattice.nneighbour+2:
            osvs=self.cache['osvs']
            self.cache['osvs']=self.block.mps.Lambda.data
            self.block.grow(sites,obonds,sbonds,osvs,qn,dtype=self.dtype)
        else:
            mpo=OptMPO([OptStr.from_operator(operator,self.degfres) for operator in self.generator.operators.itervalues()],self.degfres).to_mpo()
            ob,nb=self.block.nsite/2+1,(len(sbonds)+1)/2
            osvs=self.cache.get('osvs',np.array([1.0]))
            if niter>1 or nb-ob==1:self.cache['osvs']=self.block.mps.Lambda.data if self.block.nsite>0 else np.array([1.0])
            mps=self.block.mps.impsgrowth(sites,sbonds,osvs,qn,dtype=self.dtype)
            self.block.reset(mpo=mpo,mps=mps,target=target)

def fDMRGTSG(engine,app):
    '''
    This method iterative update the fDMRG by increasing its lattice in the center by 2 blocks at each iteration.
    '''
    engine.log.open()
    num=app.recover(engine)
    scopes,nspb=range(len(app.targets)*2),engine.nspb
    def TSGSWEEP(nsweep):
        assert engine.block.cut==engine.block.nsite/2
        nold,nnew=engine.block.nsite-2*nspb,engine.block.nsite
        path=list(it.chain(['++<<']*((nnew-nold-2)/2),['++>>']*(nnew-nold-2),['++<<']*((nnew-nold-2)/2)))
        for sweep in xrange(nsweep):
            seold=engine.block.info['Esite']
            engine.sweep(info='No.%s'%(sweep+1),path=path,nmax=app.nmax,piechart=app.plot)
            senew=engine.block.info['Esite']
            if norm(seold-senew)/norm(seold+senew)<app.tol: break
    for i,target in enumerate(app.targets[num+1:]):
        pos=i+num+1
        engine.insert(scopes[pos],scopes[-pos-1],news=scopes[:pos]+scopes[-pos:] if pos>0 else None,target=target)
        engine.block.iterate(engine.log,info='%s_%s(++)'%(engine,engine.block),sp=True if pos>0 else False,nmax=app.nmax,piechart=app.plot)
        TSGSWEEP(app.npresweep if pos==0 else app.nsweep)
        if nspb>1 and len(app.targets)>1 and pos==0 and app.savedata: engine.dump()
    if num==len(app.targets)-1 and app.nmax>engine.block.mps.nmax: TSGSWEEP(app.nsweep)
    if app.plot and app.savefig:
        plt.savefig('%s/%s_%s_%s.png'%(engine.log.dir,engine,repr(engine.block.target),app.name))
        plt.close()
    if app.savedata: engine.dump()
    engine.log.close()

def fDMRGTSS(engine,app):
    '''
    This method iterative sweep the fDMRG with 2 sites updated at each iteration.
    '''
    engine.log.open()
    num=app.recover(engine)
    if num is None:
        if app.name in engine.apps: engine.rundependences(app.name)
        num=-1
    for i,(nmax,parameters,path) in enumerate(zip(app.nmaxs[num+1:],app.BS[num+1:],app.paths[num+1:])):
        app.parameters.update(parameters)
        if not (app.parameters.match(engine.parameters)): engine.update(**parameters)
        engine.sweep(info='No.%s'%(i+1),path=path,nmax=nmax,piechart=app.plot)
        if app.savedata: engine.dump()
    if app.plot and app.savefig:
        plt.savefig('%s/%s_%s_%s.png'%(engine.log.dir,engine,repr(engine.block.target),app.name))
        plt.close()
    engine.log.close()
