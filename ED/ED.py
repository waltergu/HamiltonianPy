'''
=====================
Exact diagonalization
=====================

Base class for exact diagonalization, including:
    * classes: ED, EL, BGF, GF, EIGS
    * functions: EDEIGS, EDEL, EDGFP, EDGF, EDDOS
'''

__all__=['ED','EIGS','EDEIGS','EL','EDEL','BGF','GF','EDGFP','EDGF','EDDOS']

import numpy as np
import pickle as pk
import scipy.linalg as sl
import HamiltonianPy as HP
import HamiltonianPy.Misc as HM
import matplotlib.pyplot as plt
import os,sys,time
from mpi4py import MPI

class ED(HP.Engine):
    '''
    Base class for exact diagonalization.

    Attributes
    ----------
    sector : any hashable object
        The sector of the system.
    sectors : iterable
        The sectors of the system.
    lattice : Lattice
        The lattice of the system.
    config : IDFConfig
        The configuration of the internal degrees of freedom on the lattice.
    terms : list of Term
        The terms of the system.
    dtype : np.float32, np.float64, np.complex64, np.complex128
        The data type of the matrix representation of the Hamiltonian.
    generator : Generator
        The generator of the Hamiltonian.
    operators : Operators
        The operators of the Hamiltonian.
    timers : Timers
        The timers of the ED processes.

    Supported methods:
        ========     ================================
        METHODS      DESCRIPTION
        ========     ================================
        `EDEIGS`     calculate the eigen systems
        `EDEL`       calculates the energy spectrum
        `EDGF`       calculates the Green's function
        `EDDOS`      calculates the density of states
        ========     ================================
    '''

    def __new__(cls,*arg,**karg):
        '''
        Constructor.
        '''
        result=HP.Engine.__new__(cls,*arg,**karg)
        result.timers=HP.Timers('Matrix','ES')
        return result

    def update(self,**karg):
        '''
        Update the engine.
        '''
        if len(karg)>0:
            super(ED,self).update(**karg)
            self.generator.update(**self.data(karg))
            self.operators=self.generator.operators

    def matrix(self,sector,reset=True):
        '''
        The matrix representation of the Hamiltonian.

        Parameters
        ----------
        sector : any hashable object
            The sector of the matrix representation of the Hamiltonian.
        reset : logical, optional
            True for resetting the matrix cache and False for not.

        Returns
        -------
        csr_matrix
            The matrix representation of the Hamiltonian.
        '''
        raise NotImplementedError("%s matrix err: not implemented."%self.__class__.__name__)

    def eigs(self,sector,v0=None,k=1,return_eigenvectors=False,reset_matrix=True,reset_timers=True,show_evs=True):
        '''
        Lowest k eigenvalues and optionally, the corresponding eigenvectors.

        Parameters
        ----------
        sector : any hashable object
            The sector of the eigensystem.
        v0 : 1d ndarray, optional
            The starting vector.
        k : integer, optional
            The number of eigenvalues to be computed.
        return_eigenvectors : logical, optional
            True for returning the eigenvectors and False for not.
        reset_matrix : logical, optional
            True for resetting the matrix cache and False for not.
        reset_timers : logical, optional
            True for resetting the timers and False for not.
        show_evs : logical, optional
            True for showing the calculated eigenvalues and False for not.

        Returns
        -------
        sectors : list of any hashable object
            The sectors of the k eigenvalues
        es : 1d ndarray
            Array of k eigenvalues.
        vs : list of 1d ndarray, optional
            List of k eigenvectors.
        '''
        self.log<<'::<Parameters>:: %s\n'%(', '.join('%s=%s'%(key,HP.decimaltostr(value,n=10)) for key,value in self.parameters.iteritems()))
        if reset_timers: self.timers.reset()
        if sector is None and len(self.sectors)>1:
            cols=['nopt','dim','nnz','Mt(s)','Et(s)']+(['E%s'%i for i in xrange(k-1,-1,-1)] if show_evs else [])
            widths=[14,4,8,10,10,10]+([13]*k if show_evs else [])
            info=HP.Sheet(corner='sector',rows=self.sectors,cols=cols,widths=widths)
            self.log<<'%s\n%s\n%s\n'%(info.frame(),info.coltagstostr(corneron=True),info.division())
            sectors,es,vs=[],[],[]
            for i,sector in enumerate(self.sectors):
                with self.timers.get('Matrix'): matrix=self.matrix(sector,reset=reset_matrix)
                V0=None if v0 is None or matrix.shape[0]!=v0.shape[0] else v0
                with self.timers.get('ES'): eigs=HM.eigsh(matrix,v0=V0,k=min(k,matrix.shape[0]),which='SA',return_eigenvectors=return_eigenvectors)
                self.timers.record()
                sectors.extend([sector]*min(k,matrix.shape[0]))
                es.extend(eigs[0] if return_eigenvectors else eigs)
                if return_eigenvectors: vs.extend(eigs[1].T)
                info[(sector,'nopt')]=len(self.operators)
                info[(sector,'dim')]=matrix.shape[0]
                info[(sector,'nnz')]=matrix.nnz
                info[(sector,'Mt(s)')]=self.timers['Matrix'].records[-1],'%.4e'
                info[(sector,'Et(s)')]=self.timers['ES'].records[-1],'%.4e'
                for j in xrange(k-1,-1,-1): info[(sector,'E%s'%j)]=(es[-1-j],'%.8f') if j<matrix.shape[0] else ''
                self.log<<'%s\n'%info.rowtostr(row=sector)
            indices=np.argsort(es)[:k]
            sectors=[sectors[index] for index in indices]
            es=np.asarray(es)[indices]
            if return_eigenvectors: vs=[vs[index] for index in indices]
            self.log<<'%s\n'%info.frame()
        else:
            if sector is None: sector=next(iter(self.sectors))
            with self.timers.get('Matrix'): matrix=self.matrix(sector,reset=reset_matrix)
            self.log<<'::<Information>:: sector=%s, nopt=%s, dim=%s, nnz=%s, '%(sector,len(self.operators),matrix.shape[0],matrix.nnz)
            V0=None if v0 is None or matrix.shape[0]!=v0.shape[0] else v0
            with self.timers.get('ES'): eigs=HM.eigsh(matrix,v0=V0,k=k,which='SA',return_eigenvectors=return_eigenvectors)
            self.timers.record()
            sectors=[sector]*k
            es=eigs[0] if return_eigenvectors else eigs
            if return_eigenvectors: vs=list(eigs[1].T)
            self.log<<'Mt=%.4es, Et=%.4es'%(self.timers['Matrix'].records[-1],self.timers['ES'].records[-1])
            self.log<<(', evs=%s\n'%(' '.join('%.8f'%e for e in es)) if show_evs else '\n')
        if return_eigenvectors:
            return sectors,es,vs
        else:
            return sectors,es

class EIGS(HP.App):
    '''
    The eigen system.

    Attributes
    ----------
    sector : any hashable object
        The sector of the eigensystem.
    ne : integer
        The number of lowest eigen values to compute.
    evon : logical
        True for calculating the eigenvectors and False for not.
    '''

    def __init__(self,sector=None,ne=1,evon=True,**karg):
        '''
        Constructor.

        Parameters
        ----------
        sector : any hashable object, optional
            The sector of the eigensystem.
        ne : integer, optional
            The number of lowest eigen values to compute.
        evon : logical, optional
            True for calculating the eigenvectors and False for not.
        '''
        super(EIGS,self).__init__(**karg)
        self.sector=sector
        self.ne=ne
        self.evon=evon

def EDEIGS(engine,app):
    '''
    This method calculates the lowest eigenvalues and optionally the corresponding eigenvectors of the engine.
    '''
    eigs=engine.eigs(sector=app.sector,k=app.ne,return_eigenvectors=app.evon,reset_matrix=True,reset_timers=True)
    engine.log<<'::<Time>:: matrix=%.4es, gse=%.4es\n'%(engine.timers.time('Matrix'),engine.timers.time('ES'))
    engine.log<<HP.Sheet(
                    corner=     'Energy',
                    rows=       ['Es','Et'],
                    cols=       ['Level %s'%(i+1) for i in xrange(app.ne)],
                    contents=   np.array([eigs[1],eigs[1]/len(engine.lattice)])
                    )<<'\n'
    if app.returndata: return eigs

class EL(HP.EB):
    '''
    Energy level.

    Attributes
    ----------
    sector : any hashable object
        The sector of the energy levels.
    nder : integer
        The order of derivatives to be computed.
    ns : integer
        The number of energy levels.
    '''

    def __init__(self,sector=None,nder=0,ns=6,**karg):
        '''
        Constructor.

        Parameters
        ----------
        sector : any hashable object, optional
            The sector of the energy levels.
        nder : integer, optional
            The order of derivatives to be computed.
        ns : integer, optional
            The number of energy levels.
        '''
        super(EL,self).__init__(**karg)
        self.sector=sector
        self.nder=nder
        self.ns=ns

def EDEL(engine,app):
    '''
    This method calculates the energy levels of the Hamiltonian.
    '''
    name='%s_%s'%(engine.tostr(mask=app.path.tags),app.name)
    result=np.zeros((app.path.rank(0),app.ns*(app.nder+1)+1))
    result[:,0]=app.path.mesh(0) if len(app.path.tags)==1 and app.path.mesh(0).ndim==1 else np.array(xrange(app.path.rank(0)))
    for i,paras in enumerate(app.path('+')):
        engine.update(**paras)
        result[i,1:app.ns+1]=engine.eigs(sector=app.sector,k=app.ns,return_eigenvectors=False,reset_matrix=True if i==0 else False,reset_timers=True if i==0 else False)[1]
        engine.log<<'%s\n\n'%engine.timers.tostr(HP.Timers.ALL)
        if app.plot: engine.timers.graph(parents=HP.Timers.ALL)
    else:
        if app.plot:
            engine.timers.cleancache()
            if app.savefig: plt.savefig('%s/%s_TIMERS.png'%(engine.log.dir,name))
            plt.close()
    if app.nder>0:
        for i in xrange(app.ns): result.T[[j*app.ns+i+1 for j in xrange(1,app.nder+1)]]=HM.derivatives(result[:,0],result[:,i+1],ders=range(1,app.nder+1))
    if app.savedata: np.savetxt('%s/%s.dat'%(engine.dout,name),result)
    if app.plot:
        ns=app.ns
        options={'legend':[('%s der of '%HP.ordinal(k/ns-1) if k/ns>0 else '')+'$E_{%s}$'%(k%ns) for k in xrange(result.shape[1]-1)],'legendloc':'lower right'} if ns<=10 else {}
        app.figure('L',result,'%s/%s'%(engine.dout,name),**options)
    if app.returndata: return result

class BGF(object):
    '''
    A block of a zero-temperature Green's function.

    Attributes
    ----------
    method : 'S' or 'B'
        'S' for simple Lanczos method and 'B' for block Lanczos method.
    indices : 1d ndarray
        The block's indices in the Green's function.
    controllers : dict
        The controllers to set the cached data of the block.
    data : dict
        The cached data to calculate the block valus of the Green's function.
    '''

    def __init__(self,method,indices,sign,matrix,operators):
        '''
        Constructor.

        Parameters
        ----------
        indices : 1d ndarray
            The block's indices in the Green's function.
        method : 'S' or 'B'
            'S' for simple Lanczos method and 'B' for block Lanczos method.
        sign : +1,-1
            The corresponding sign of the block.
        matrix : csr_matrix
            The corresponding matrix of the block.
        operators : list of csr_matrix
            The matrix representations of the corresponding operators of the block.
        '''
        assert method in ('S','B') and np.abs(sign)==1
        self.method=method
        self.indices=indices
        self.controllers={'sign':sign,'matrix':matrix,'operators':operators}
        self.data={}

    def prepare(self,groundstate,nstep):
        '''
        Prepare the lanczos representation of the block.

        Parameters
        ----------
        groundstate : 1d ndarray
            The ground state.
        nstep : int
            The number of iterations over the whole starting states.
        '''
        matrix,operators=self.controllers['matrix'],self.controllers['operators']
        if self.method=='S':
            self.controllers['vecs'],self.controllers['lczs']=[],[]
            for operator in operators:
                v0=operator.dot(groundstate)
                self.controllers['vecs'].append(v0.conjugate())
                self.controllers['lczs'].append(HM.Lanczos(matrix,[v0],maxiter=nstep,keepstate=False))
            self.controllers['vecs']=np.asarray(self.controllers['vecs'])
            self.controllers['Qs']=np.zeros((len(operators),len(operators),nstep),dtype=matrix.dtype)
        else:
            self.controllers['lanczos']=HM.Lanczos(matrix,v0=[operator.dot(groundstate) for operator in operators],maxiter=nstep*len(operators),keepstate=False)

    def iter(self,log=None,np=None):
        '''
        The iteration of the Lanczos.

        Parameters
        ----------
        log : Log, optional
            The log file to record the iteration information.
        np : int, optional
            The number of subprocess to perform the iteration.
        '''
        t0=time.time()
        if self.method=='S' and (np is None or np<=0):
            vecs,Qs=self.controllers['vecs'],self.controllers['Qs']
            for i,lanczos in enumerate(self.controllers['lczs']):
                ts=time.time()
                while lanczos.niter<lanczos.maxiter and not lanczos.stop:
                    lanczos.iter()
                    if lanczos.niter>0: Qs[i,:,lanczos.niter-1]=vecs.dot(lanczos.vectors[lanczos.niter-1])
                te=time.time()
                if log: log<<'%s%s%s'%('\b'*30 if i>0 else '',('%s/%s(%.2es/%.3es)'%(i+1,len(Qs),te-ts,te-t0)).center(30),'\b'*30 if i==len(Qs)-1 else '')
        elif self.method=='B':
            lanczos=self.controllers['lanczos']
            for i in xrange(lanczos.maxiter):
                ts=time.time()
                lanczos.iter()
                te=time.time()
                if log: log<<'%s%s%s'%('\b'*30 if i>0 else '',('%s/%s(%.2es/%.3es)'%(i+1,lanczos.maxiter,te-ts,te-t0)).center(30),'\b'*30 if i==lanczos.maxiter-1 else '')
        elif self.method=='S' and np is not None:
            path,Qs=os.path.dirname(os.path.realpath(__file__)),self.controllers['Qs']
            datas=[[self.controllers['vecs'],[],[]] for i in xrange(np)]
            for i,lanczos in enumerate(self.controllers['lczs']):
                datas[i%np][1].append(lanczos)
                datas[i%np][2].append(i)
            comm=MPI.COMM_SELF.Spawn(sys.executable,args=['%s/edbgf.py'%path],maxprocs=np)
            for i,data in enumerate(datas):
                comm.send(data,dest=i,tag=0)
            info,ic,nc=MPI.Status(),0,0
            while nc<np:
                data=comm.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=info)
                if info.Get_tag()==0:
                    for index,(_T_,P,niter),Q in data:
                        lanczos=self.controllers['lczs'][index]
                        lanczos._T_,lanczos.P,lanczos.niter=_T_,P,niter
                        Qs[index,:,:]=Q
                    nc+=1
                else:
                    ic,(index,t)=ic+1,data
                    if log: log<<'%s%s%s'%('\b'*30 if ic>1 else '',('%s/%s(%.2es/%.3es)'%(ic,len(Qs),t,time.time()-t0)).center(30),'\b'*30 if ic==len(Qs) else '')
            comm.Disconnect()
        else:
            raise ValueError('BGF iter error: not supported.')

    def set(self,gse):
        '''
        Set the Lambda matrix, Q matrix and QT matrix of the block.

        Parameters
        ----------
        gse : number
            The groundstate energy.
        '''
        sign=self.controllers['sign']
        if self.method=='S':
            lczs,Qs=self.controllers['lczs'],self.controllers['Qs']
            self.data['niters']=np.zeros(Qs.shape[0],dtype=np.int32)
            self.data['Lambdas']=np.zeros((Qs.shape[0],Qs.shape[2]),dtype=np.float64)
            self.data['Qs']=np.zeros(Qs.shape,dtype=Qs.dtype)
            self.data['QTs']=np.zeros((Qs.shape[0],Qs.shape[2]),dtype=Qs.dtype)
            for i,(lcz,Q) in enumerate(zip(lczs,Qs)):
                if lcz.niter>0:
                    E,V=sl.eigh(lcz.T,eigvals_only=False)
                    self.data['niters'][i]=lcz.niter
                    self.data['Lambdas'][i,0:lcz.niter]=sign*(E-gse)
                    self.data['Qs'][i,:,0:lcz.niter]=Q[:,0:lcz.niter].dot(V)
                    self.data['QTs'][i,0:lcz.niter]=lcz.P[0,0]*V[0,:].conjugate()
        else:
            lanczos=self.controllers['lanczos']
            E,V=sl.eigh(lanczos.T,eigvals_only=False)
            self.data['Lambda']=sign*(E-gse)
            self.data['Q']=lanczos.P[:min(lanczos.nv0,lanczos.niter),:].T.conjugate().dot(V[:min(lanczos.nv0,lanczos.niter),:])
            self.data['QT']=HM.dagger(self.data['Q'])

    def clear(self):
        '''
        Clear the controllers of the block.
        '''
        delattr(self,'controllers')

    def gf(self,omega):
        '''
        The values of the block of the Green's function.

        Parameters
        ----------
        omega : number
            The frequency.

        Returns
        -------
        2d ndarray
            The values of the block.
        '''
        if self.method=='S':
            niters,Lambdas,Qs,QTs=self.data['niters'],self.data['Lambdas'],self.data['Qs'],self.data['QTs']
            result=np.zeros((Lambdas.shape[0],Lambdas.shape[0]),dtype=np.complex128)
            for i in xrange(Lambdas.shape[0]):
                result[i,:]=(Qs[i,:,0:niters[i]]/(omega-Lambdas[i,0:niters[i]])[np.newaxis,:]).dot(QTs[i,0:niters[i]])
            return result
        else:
            return (self.data['Q']/(omega-self.data['Lambda'])[np.newaxis,:]).dot(self.data['QT'])

class GF(HP.GF):
    '''
    Zero-temperature Green's function.

    Attributes
    ----------
    generate : callable
        The function that generates the blocks of the Green's function.
    compose : callable
        The function that composes the Green's function from its blocks.
    v0 : 1d ndarray
        The initial guess of the groundstate.
    nstep : int
        The number of steps for the Lanczos iteration.
    gse : number
        The groundstate energy.
    blocks : list of BGF
        The blocks of the Green's function.
    '''

    def __init__(self,generate,compose,v0=None,nstep=200,method='S',**karg):
        '''
        Constructor.

        Parameters
        ----------
        generate : callable
            The function that generates the blocks of the Green's function.
        compose : callable
            The function that composes the Green's function from its blocks.
        v0 : 1d ndarray, optional
            The initial guess of the groundstate.
        nstep : int, optional
            The number of steps for the Lanczos iteration.
        method : 'S' or 'B', optional
            'S' for simple Lanczos method and 'B' for block Lanczos method.
        '''
        super(GF,self).__init__(**karg)
        self.generate=generate
        self.compose=compose
        self.v0=v0
        self.nstep=nstep
        self.method=method

def EDGFP(engine,app):
    '''
    This method prepares the GF.
    '''
    if os.path.isfile('%s/%s_coeff.dat'%(engine.din,engine.tostr(ndecimal=14))):
        engine.log<<'::<Parameters>:: %s\n'%(', '.join('%s=%s'%(key,HP.decimaltostr(value,n=10)) for key,value in engine.parameters.iteritems()))
        with open('%s/%s_coeff.dat'%(engine.din,engine.tostr(ndecimal=14)),'rb') as fin:
            app.gse=pk.load(fin)
            app.blocks=pk.load(fin)
        return
    sectors,es,vs=engine.eigs(sector=None,v0=app.v0,k=1,return_eigenvectors=True,reset_matrix=True,reset_timers=True)
    engine.sector,app.gse,app.v0=sectors[0],es[0],vs[0]
    if len(engine.sectors)>1: engine.log<<'::<Information>:: sector=%s, gse=%.8f.\n'%(engine.sector,app.gse)
    app.blocks,nb,blocks=[],next(iter(app.generate(engine,app.operators,'NB'))),app.generate(engine,app.operators,app.method)
    timers=HP.Timers('Preparation','Iteration','Diagonalization',root='Total')
    info=HP.Sheet(corner='GF Block',cols=['Preparation','Iteration','Diagonalization','Total'],rows=['# %s'%(i+1) for i in xrange(nb)]+['Summary'],widths=[8,11,11,15,11])
    engine.log<<'%s\n%s\n%s\n'%(info.frame(),info.coltagstostr(corneron=True),info.division())
    for i in xrange(nb):
        with timers.get('Total') as gftimer:
            bnum='# %s'%(i+1)
            engine.log<<'%s|'%info.tagtostr(bnum)
            with timers.get('Preparation') as timer:
                block=next(iter(blocks))
                block.prepare(app.v0,app.nstep)
                timer.record()
                info[(bnum,'Preparation')]=timer.records[-1],'%.5e'
                engine.log<<'%s|'%info.entrytostr((bnum,'Preparation'))
            with timers.get('Iteration') as timer:
                block.iter(engine.log,np=app.np)
                timer.record()
                info[(bnum,'Iteration')]=timer.records[-1],'%.5e'
                engine.log<<'%s|'%info.entrytostr((bnum,'Iteration'))
            with timers.get('Diagonalization') as timer:
                block.set(app.gse)
                block.clear()
                app.blocks.append(block)
                timer.record()
                info[(bnum,'Diagonalization')]=timer.records[-1],'%.5e'
                engine.log<<'%s|'%info.entrytostr((bnum,'Diagonalization'))
            gftimer.record()
            info[(bnum,'Total')]=gftimer.records[-1],'%.5e'
            engine.log<<'%s\n'%info.entrytostr((bnum,'Total'))
    for key in ['Preparation','Iteration','Diagonalization','Total']:
        info[('Summary',key)]=timers.time(key),'%.5e'
    engine.log<<'%s\n%s\n'%(info.rowtostr('Summary'),info.frame())
    if app.savedata:
        with open('%s/%s_coeff.dat'%(engine.din,engine.tostr(ndecimal=14)),'wb') as fout:
            pk.dump(app.gse,fout,2)
            pk.dump(app.blocks,fout,2)

def EDGF(engine,app):
    '''
    This method calculate the GF.
    '''
    gf=engine.records[app.name]
    if app.omega is not None:
        gf=app.compose(app.blocks,app.omega)
        engine.records[app.name]=gf
    return gf

def EDDOS(engine,app):
    '''
    This method calculates the DOS.
    '''
    engine.rundependences(app.name)
    erange=np.linspace(app.emin,app.emax,num=app.ne)
    gf=engine.apps[app.dependences[0]]
    gf_mesh=np.zeros((app.ne,gf.nopt,gf.nopt),dtype=gf.dtype)
    for i,omega in enumerate(erange+app.mu+1j*app.eta):
        gf.omega=omega
        gf_mesh[i,:,:]=gf.run(engine,gf)
    result=np.zeros((app.ne,2))
    result[:,0]=erange
    result[:,1]=-2*np.trace(gf_mesh,axis1=1,axis2=2).imag
    name='%s_%s'%(engine,app.name)
    if app.savedata: np.savetxt('%s/%s.dat'%(engine.dout,name),result)
    if app.plot: app.figure('L',result,'%s/%s'%(engine.dout,name))
    if app.returndata: return result
