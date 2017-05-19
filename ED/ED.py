'''
====================
Exat diagonalization
====================

Exact diagonalization for fermionic systems, including:
    * classes: ED, EL, GF
    * functions: EDEL, EDGFP, EDGF, EDDOS
'''

__all__=['ED','EL','EDEL','GF','EDGFP','EDGF','EDDOS']

from ..Misc import Lanczos,derivatives
from scipy.linalg import eigh,norm,solve_banded,solveh_banded
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from copy import deepcopy
import numpy as np
import pickle as pk
import HamiltonianPy as HP
import matplotlib.pyplot as plt
import os.path,sys,time

class ED(HP.Engine):
    '''
    Exact diagonalization for an electron system, based on the sparse matrix representation of the Hamiltonian on the occupation number basis.

    Attributes
    ----------
    basis : FBasis
        The occupation number basis of the system.
    lattice : Lattice
        The lattice of the system.
    config : IDFConfig
        The configuration of the internal degrees of freedom on the lattice.
    terms : list of Term
        The terms of the system.
    dtype : np.float32, np.float64, np.complex64, np.complex128
        The data type of the matrix representation of the Hamiltonian.
    generator : Generator
        The generator for the Hamiltonian.
    operators : Operators
        The 'half' of the operators for the Hamiltonian.
    matrix : csr_matrix
        The sparse matrix representation of the cluster Hamiltonian.


    Supported methods:
        =======     ===============================================
        METHODS     DESCRIPTION
        =======     ===============================================
        `EDEL`      calculates the energy spectrum
        `EDGF`      calculates the single-particle Green's function
        `EDDOS`     calculates the density of states
        =======     ===============================================
    '''

    def __init__(self,basis=None,lattice=None,config=None,terms=None,dtype=np.complex128,**karg):
        '''
        Constructor.

        Parameters
        ----------
        basis : FBasis, optional
            The occupation number basis of the system.
        lattice : Lattice, optional
            The lattice of the system.
        config : IDFConfig, optional
            The configuration of the internal degrees of freedom on the lattice.
        terms : list of Term, optional
            The terms of the system.
        dtype : np.float32, np.float64, np.complex64, np.complex128
            The data type of the matrix representation of the Hamiltonian.
        '''
        self.basis=basis
        self.lattice=lattice
        self.config=config
        self.terms=terms
        self.dtype=dtype
        self.generator=HP.Generator(bonds=lattice.bonds,config=config,table=config.table(mask=['nambu']),terms=terms,dtype=dtype)
        self.status.update(const=self.generator.parameters['const'])
        self.status.update(alter=self.generator.parameters['alter'])
        self.operators=self.generator.operators

    def update(self,**karg):
        '''
        Update the engine.
        '''
        self.generator.update(**karg)
        self.operators=self.generator.operators
        self.status.update(alter=self.generator.parameters['alter'])

    def set_matrix(self):
        '''
        Set the csr_matrix representation of the Hamiltonian.
        '''
        self.matrix=csr_matrix((self.basis.nbasis,self.basis.nbasis),dtype=self.dtype)
        for operator in self.operators.itervalues():
            self.matrix+=HP.foptrep(operator,self.basis,transpose=False)
        self.matrix+=self.matrix.T.conjugate()
        self.matrix=self.matrix.T

    def __replace_basis__(self,nambu,spin):
        '''
        Return a new ED instance with the basis replaced.

        Parameters
        ----------
        nambu : CREATION or ANNIHILATION
            CREATION for adding one electron and ANNIHILATION for subtracting one electron.
        spin : 0 or 1
            0 for spin down and 1 for spin up.

        Returns
        -------
        ED
            The new ED instance with the wanted new basis.
        '''
        if self.basis.mode=='FG':
            return self
        elif self.basis.mode=='FP':
            result=deepcopy(self)
            if nambu==HP.CREATION:
                result.basis=HP.FBasis((self.basis.nstate,self.basis.nparticle+1))
            else:
                result.basis=HP.FBasis((self.basis.nstate,self.basis.nparticle-1))
            result.matrix=csr_matrix((result.basis.nbasis,result.basis.nbasis),dtype=self.dtype)
            result.set_matrix()
            return result
        else:
            result=deepcopy(self)
            if nambu==HP.CREATION and spin==0:
                result.basis=HP.FBasis(up=(self.basis.nstate[0],self.basis.nparticle[0]),down=(self.basis.nstate[1],self.basis.nparticle[1]+1))
            elif nambu==HP.ANNIHILATION and spin==0:
                result.basis=HP.FBasis(up=(self.basis.nstate[0],self.basis.nparticle[0]),down=(self.basis.nstate[1],self.basis.nparticle[1]-1))
            elif nambu==HP.CREATION and spin==1:
                result.basis=HP.FBasis(up=(self.basis.nstate[0],self.basis.nparticle[0]+1),down=(self.basis.nstate[1],self.basis.nparticle[1]))
            else:
                result.basis=HP.FBasis(up=(self.basis.nstate[0],self.basis.nparticle[0]-1),down=(self.basis.nstate[1],self.basis.nparticle[1]))
            result.matrix=csr_matrix((result.basis.nbasis,result.basis.nbasis),dtype=self.dtype)
            result.set_matrix()
            return result

    def eig(self,k=1,return_eigenvectors=False):
        '''
        Lowest k eigenvalues and optionally, the corresponding eigenvectors.

        Parameters
        ----------
        k : integer, optional
            The number of eigenvalues to be computed.
        return_eigenvectors : logical, optional
            True for returning the eigenvectors and False for not.
        '''
        self.set_matrix()
        return eigsh(self.matrix,k=k,which='SA',return_eigenvectors=return_eigenvectors)

class EL(HP.EB):
    '''
    Energy level.

    Attributes
    ----------
    nder : integer
        The order of derivatives to be computed.
    ns : integer
        The number of energy levels.
    '''
    
    def __init__(self,nder=0,ns=6,**karg):
        '''
        Constructor.

        Parameters
        ----------
        nder : integer, optional
            The order of derivatives to be computed.
        ns : integer, optional
            The number of energy levels.
        '''
        super(EL,self).__init__(**karg)
        self.nder=nder
        self.ns=ns

def EDEL(engine,app):
    '''
    This method calculates the energy levels of the Hamiltonian.
    '''
    result=np.zeros((app.path.rank.values()[0],app.ns*(app.nder+1)+1))
    if len(app.path.rank)==1 and len(app.path.mesh.values()[0].shape)==1:
        result[:,0]=app.path.mesh.values()[0]
    else:
        result[:,0]=array(xrange(app.path.rank.values()[0]))
    for i,paras in enumerate(app.path('+')):
        engine.update(**paras)
        engine.set_matrix()
        result[i,1:app.ns+1]=eigsh(engine.matrix,k=app.ns,which='SA',return_eigenvectors=False)
    suffix='_%s'%(app.status.name)
    if app.nder>0:
        for i in xrange(app.ns):
            result.T[[j*app.ns+i+1 for j in xrange(1,app.nder+1)]]=derivatives(result[:,0],result[:,i+1],ders=range(1,app.nder+1))
    if app.save_data:
        np.savetxt('%s/%s_%s.dat'%(engine.dout,engine.status.const,app.status.name),result)
    if app.plot:
        plt.title('%s_%s'%(engine.status.const,app.status.name))
        prefixs={i:'1st' if i==1 else ('2nd' if i==2 else ('3rd' if i==3 else '%sth'%i)) for i in xrange(app.nder+1)}
        for k in xrange(1,result.shape[1]):
            i,j=divmod(k-1,app.ns)
            plt.plot(result[:,0],result[:,k],label=('%s der of '%prefixs[i] if i>0 else '')+'E%s'%(j))
        plt.legend(shadow=True,fancybox=True,loc='lower right')
        if app.show and app.suspend: plt.show()
        if app.show and not app.suspend: plt.pause(app.SUSPEND_TIME)
        if app.save_fig: plt.savefig('%s/%s_%s.png'%(engine.dout,engine.status.const,app.status.name))
        plt.close()

class GF(HP.GF):
    '''
    The single-particle zero-temperature Green's function.

    Attributes
    ----------
    mask : ['nambu'] or []
        * When ['nambu'], the anomalous Green's functions are not computed;
        * When [], the anomalous Green's functions are also computed.
    nspin : 1 or 2
        * When 1, sometimes(engine.basis.mode=='FS') only spin-down parts of the Green's function is computed;
        * When 2, both spin-up and spin-down parts of the Green's function is computed.
    v0 : 1D ndarray
        The initial guess of the groundstate.
    nstep : integer
        The max number of steps for the Lanczos iteration.
    method : 'user', 'python' or 'dense'
        It specifies the method the engine uses to compute the ground state.
        'user' for HamiltonianPy.Math.Lanczos.eig, 'python' for scipy.sparse.linalg.eigsh, and 'dense' for scipy.linalg.eigh.
    vtype : 'rd' or 'sy'
        When v0 is None, it specifies the type of the default initial guess of the groundstate.
        It only makes sense when method is 'user'. Then 'rd' means random ones and 'sy' for symmetric ones.
    tol : float
        The tolerance used to terminate the iteration.
    gse : float64
        The ground state energy of the system.
    coeff,hs : 4d ndarray
        The auxiliary data for the computing of GF.
    '''

    def __init__(self,mask=['nambu'],nspin=2,v0=None,nstep=200,method='python',vtype='rd',tol=0,**karg):
        '''
        Constructor.

        Parameters
        ----------
        mask : ['nambu'] or [], optional
            The flag to tell whether or not to compute the anomalous Green's function.
        nspin : 1 or 2, optional
            The flag to tell whether or not to compute only spin-down component.
        v0 : 1D ndarray, optional
            The initial guess of the groundstate.
        nstep : integer, optional
            The max number of steps for the Lanczos iteration.
        method : 'user', 'python' or 'dense', optional
            The method the engine uses to compute the ground state.
        vtype : 'rd' or 'sy', optional
            When v0 is None, it specifies the type of the default initial guess of the groundstate.
        tol : float, optional
            The tol used to terminate the iteration.
        '''
        super(GF,self).__init__(**karg)
        self.mask=mask
        self.nspin=nspin
        self.v0=v0
        self.nstep=nstep
        self.method=method
        self.vtype=vtype
        self.tol=tol

    @staticmethod
    def select(nspin):
        '''
        Return a select function based on nspin.
        '''
        return lambda index: True if nspin==2 or index.spin==0 else False

def EDGFP(engine,app):
    '''
    This method prepares the GF.
    '''
    engine.log<<'::<Parameters>:: %s\n'%(', '.join('%s=%s'%(name,value) for name,value in engine.status.data.iteritems()))
    if engine.basis.mode in ('FG','FP'): assert app.nspin==2
    if os.path.isfile('%s/%s_coeff.dat'%(engine.din,engine.status)):
        with open('%s/%s_coeff.dat'%(engine.din,engine.status),'rb') as fin:
            app.gse=pk.load(fin)
            app.coeff=pk.load(fin)
            app.hs=pk.load(fin)
        return
    timers=HP.Timers('Matrix','GSE','GF')
    timers.add(parent='GF',name='Preparation')
    timers.add(parent='GF',name='Iteration')
    app.coeff=np.zeros((2,app.nopt,app.nopt,app.nstep),dtype=app.dtype)
    app.hs=np.zeros((2,app.nopt,2,app.nstep),dtype=app.dtype)
    with timers.get('Matrix'):
        engine.set_matrix()
    engine.log<<'::<Information>:: %s=%s, %s=%s, %s=%s, '%('nopts',len(engine.operators),'shape',engine.matrix.shape,'nnz',engine.matrix.nnz)
    with timers.get('GSE'):
        if app.method in ('user',):
            app.gse,app.v0=Lanczos(engine.matrix,v0=app.v0,vtype=app.vtype,zero=app.tol).eig(job='v',precision=app.tol)
        elif app.method in ('python',):
            es,vs=eigsh(engine.matrix,k=1,which='SA',v0=app.v0,tol=app.tol)
            app.gse,app.v0=es[0],vs[:,0]
        elif app.method in ('dense',):
            w,v=eigh(engine.matrix.todense())
            app.gse,app.v0=w[0],v[:,0]
        else:
            raise ValueError('GF preparation error: mehtod(%s) not supported.'%(app.method))
    engine.log<<'%s=%.6f\n'%('GSE',app.gse)
    with timers.get('GF'):
        if engine.basis.mode in ('FS','FP'): engine.matrix=None
        engine.log<<'%s\n'%('~'*56)
        engine.log<<'%s|%s|%s|%s\n%s\n'%('Time(seconds)'.center(13),'Preparation'.center(13),'Iteration'.center(13),'Total'.center(13),'-'*56)
        for h in xrange(2):
            t0=time.time()
            with timers.get('Preparation'):
                engine.log<<'%s|'%(':Electron:' if h==0 else ':Hole:').center(13)
                states,norms,lczs=[],[],[]
                for i,opt in enumerate(app.operators):
                    if i==0:
                        ed=engine.__replace_basis__(nambu=1-h,spin=0)
                    if i==app.nopt/2 and app.nspin==2 and engine.basis.mode=='FS':
                        ed=engine.__replace_basis__(nambu=1-h,spin=1)
                    mat=HP.foptrep(opt.dagger if h==0 else opt,basis=[engine.basis,ed.basis],transpose=True)
                    state=mat.dot(app.v0)
                    states.append(state)
                    temp=norm(state)
                    norms.append(temp)
                    lczs.append(Lanczos(ed.matrix,v0=state/temp,check_normalization=False))
                    engine.log<<'%s%s'%('\b'*21 if i>0 else '',('%s/%s(%1.5es)'%(i,app.nopt,time.time()-t0)).center(21))
                engine.log<<'%s%s|'%('\b'*21,('%1.5e'%(time.time()-t0)).center(13))
            t1=time.time()
            with timers.get('Iteration'):
                for k in xrange(app.nstep):
                    for i,(temp,lcz) in enumerate(zip(norms,lczs)):
                        if not lcz.cut:
                            for j,state in enumerate(states):
                                if engine.basis.mode in ('FP','FG') or app.nspin==1 or (i<app.nopt/2 and j<app.nopt/2) or (i>=app.nopt/2 and j>=app.nopt/2):
                                    app.coeff[h,i,j,k]=np.vdot(state,lcz.new)*temp
                            lcz.iter()
                    engine.log<<'%s%s'%(('\b'*21 if k>0 else ''),('%s/%s(%1.5es)'%(k,app.nstep,time.time()-t1)).center(21))
                engine.log<<'%s%s|'%('\b'*21,('%1.5e'%(time.time()-t1)).center(13))
                for i,lcz in enumerate(lczs):
                    app.hs[h,i,0,0:len(lcz.a)]=np.array(lcz.a)
                    app.hs[h,i,1,0:len(lcz.b)]=np.array(lcz.b)
            engine.log<<('%1.5e'%(time.time()-t0)).center(13)<<'\n'
        tp,ti,tg=('%1.5e'%timers.time('Preparation')).center(13),('%1.5e'%timers.time('Iteration')).center(13),('%1.5e'%timers.time('GF')).center(13)
        engine.log<<'%s|%s|%s|%s\n%s\n'%('Summary'.center(13),tp,ti,tg,'~'*56)
    timers.record()
    engine.log<<'Summary of the gf preparation:\n%s\n'%timers.tostr(None,form='s')
    if app.save_data:
        with open('%s/%s_coeff.dat'%(engine.din,engine.status),'wb') as fout:
            pk.dump(app.gse,fout,2)
            pk.dump(app.coeff,fout,2)
            pk.dump(app.hs,fout,2)

def EDGF(engine,app):
    '''
    This method calculate the GF.
    '''
    if app.omega is not None:
        app.gf[...]=0.0
        buff=np.zeros((3,app.nstep),dtype=app.dtype)
        b=np.zeros(app.nstep,dtype=app.dtype)
        for h in xrange(2):
            for i in xrange(app.nopt):
                b[...]=0
                b[0]=1
                buff[0,1:]=app.hs[h,i,1,0:app.nstep-1]*(-1)**(h+1)
                buff[1,:]=app.omega-(app.hs[h,i,0,:]-app.gse)*(-1)**h
                buff[2,:]=app.hs[h,i,1,:]*(-1)**(h+1)
                temp=solve_banded((1,1),buff,b,overwrite_ab=True,overwrite_b=True,check_finite=False)
                if h==0:
                    app.gf[:,i]+=np.dot(app.coeff[h,i,:,:],temp)
                else:
                    app.gf[i,:]+=np.dot(app.coeff[h,i,:,:],temp)
    return app.gf

def EDDOS(engine,app):
    '''
    This method calculates the DOS.
    '''
    engine.rundependences(app.status.name)
    erange=np.linspace(app.emin,app.emax,num=app.ne)
    gf=app.dependences[0]
    gf_mesh=np.zeros((app.ne,)+gf.gf.shape,dtype=gf.dtype)
    for i,omega in enumerate(erange+app.mu+1j*app.eta):
        gf.omega=omega
        gf_mesh[i,:,:]=gf.run(engine,gf)
    result=np.zeros((app.ne,2))
    result[:,0]=erange
    result[:,1]=-2*np.trace(gf_mesh,axis1=1,axis2=2).imag
    if app.save_data:
        np.savetxt('%s/%s_%s.dat'%(engine.dout,engine.status,app.status.name),result)
    if app.plot:
        plt.title('%s_%s'%(engine.status,app.status.name))
        plt.plot(result[:,0],result[:,1])
        if app.show and app.suspend: plt.show()
        if app.show and not app.suspend: plt.pause(app.SUSPEND_TIME)
        if app.save_fig: plt.savefig('%s/%s_%s.png'%(engine.dout,engine.status,app.status.name))
        plt.close()
