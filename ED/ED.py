'''
Exat diagonalization, including:
1) classes: ED, EL, GF
2) functions: EDEL, EDGFP, EDGF, EDDOS
'''

__all__=['ED','EL','EDEL','GF','EDGFP','EDGF','EDDOS']

from ..Math import Lanczos
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
    Attributes:
        basis: BasisF
            The occupation number basis of the system.
        filling: float
            The filling factor.
        mu: float
            The chemical potential.
        lattice: Lattice
            The lattice of the system.
        config: IDFConfig
            The configuration of the internal degrees of freedom on the lattice.
        terms: list of Term
            The terms of the system.
        dtype: np.float64, np.complex128
            The data type of the matrix representation of the Hamiltonian.
        generators: dict of Generator
            It has only one entries:
            1) 'h': Generator
                The generator for the Hamiltonian.
        operators: dict of OperatorCollection
            It has only one entry:
            1) 'h': OperatorCollection
                The 'half' of the operators for the Hamiltonian.
        matrix: csr_matrix
            The sparse matrix representation of the cluster Hamiltonian.
    Supported methods include:
        1) EDEL: calculates the energy spectrum.
        2) EDGF: calculates the single-particle Green's function.
        3) EDDOS: calculates the density of states.
    '''

    def __init__(self,basis=None,filling=None,mu=None,lattice=None,config=None,terms=None,dtype=np.complex128,**karg):
        '''
        Constructor.
        Parameters:
            basis: BasisF, optional
                The occupation number basis of the system.
            filling: float, optional
                The filling factor.
            mu: float, optional
                The chemical potential.
            lattice: Lattice, optional
                The lattice of the system.
            config: IDFConfig, optional
                The configuration of the internal degrees of freedom on the lattice.
            terms: list of Term, optional
                The terms of the system.
            dtype: np.float64, np.complex128
                The data type of the matrix representation of the Hamiltonian.
        '''
        self.basis=basis
        if basis.mode=='FG':
            self.filling=filling
            self.mu=[term for term in terms if term.id=='mu'][0].value
            self.status.update(alter={'mu':self.mu})
        else:
            self.filling=1.0*basis.nparticle.sum()/basis.nstate.sum()
            self.mu=mu
            self.status.update(const={'filling':self.filling})
        self.lattice=lattice
        self.config=config
        self.terms=terms
        self.dtype=dtype
        self.generators={'h':HP.Generator(bonds=lattice.bonds,config=config,table=config.table(mask=['nambu']),terms=terms,dtype=dtype)}
        self.status.update(const=self.generators['h'].parameters['const'])
        self.status.update(alter=self.generators['h'].parameters['alter'])
        self.operators={'h':self.generators['h'].operators}

    def update(self,**karg):
        '''
        Update the alterable operators.
        '''
        self.generators['h'].update(**karg)
        self.operators['h']=self.generators['h'].operators
        self.status.update(alter=self.generators['h'].parameters['alter'])

    def set_matrix(self):
        '''
        Set the csr_matrix representation of the Hamiltonian.
        '''
        self.matrix=csr_matrix((self.basis.nbasis,self.basis.nbasis),dtype=self.dtype)
        for operator in self.operators['h'].itervalues():
            self.matrix+=HP.f_opt_rep(operator,self.basis,transpose=False)
        self.matrix+=self.matrix.T.conjugate()
        self.matrix=self.matrix.T

    def __replace_basis__(self,nambu,spin):
        '''
        Return a new ED instance with the basis replaced.
        Parameters:
            nambu: CREATION or ANNIHILATION
                CREATION for adding one electron and ANNIHILATION for subtracting one electron.
            spin: 0 or 1
                0 for spin down and 1 for spin up.
        Returns: ED
            The new ED instance with the wanted new basis.
        '''
        if self.basis.mode=='FG':
            return self
        elif self.basis.mode=='FP':
            result=deepcopy(self)
            if nambu==HP.CREATION:
                result.basis=HP.BasisF((self.basis.nstate,self.basis.nparticle+1))
            else:
                result.basis=HP.BasisF((self.basis.nstate,self.basis.nparticle-1))
            result.matrix=csr_matrix((result.basis.nbasis,result.basis.nbasis),dtype=np.complex128)
            result.set_matrix()
            return result
        else:
            result=deepcopy(self)
            if nambu==HP.CREATION and spin==0:
                result.basis=HP.BasisF(up=(self.basis.nstate[0],self.basis.nparticle[0]),down=(self.basis.nstate[1],self.basis.nparticle[1]+1))
            elif nambu==HP.ANNIHILATION and spin==0:
                result.basis=HP.BasisF(up=(self.basis.nstate[0],self.basis.nparticle[0]),down=(self.basis.nstate[1],self.basis.nparticle[1]-1))
            elif nambu==HP.CREATION and spin==1:
                result.basis=HP.BasisF(up=(self.basis.nstate[0],self.basis.nparticle[0]+1),down=(self.basis.nstate[1],self.basis.nparticle[1]))
            else:
                result.basis=HP.BasisF(up=(self.basis.nstate[0],self.basis.nparticle[0]-1),down=(self.basis.nstate[1],self.basis.nparticle[1]))
            result.matrix=csr_matrix((result.basis.nbasis,result.basis.nbasis),dtype=self.dtype)
            result.set_matrix()
            return result

class EL(HP.EB):
    '''
    Energy level.
    Attribues:
        ns: integer
            The number of energy levels.
    '''
    
    def __init__(self,ns=6,**karg):
        '''
        Constructor.
        Parameters:
            ns: integer, optional
                The number of energy levels.
        '''
        super(EL,self).__init__(**karg)
        self.ns=ns

def EDEL(engine,app):
    '''
    This method calculates the energy levels of the Hamiltonian.
    '''
    result=np.zeros((app.path.rank.values()[0],app.ns+1))
    if len(app.path.rank)==1 and len(app.path.mesh.values()[0].shape)==1:
        result[:,0]=app.path.mesh.values()[0]
    else:
        result[:,0]=array(xrange(app.path.rank.values()[0]))
    for i,paras in enumerate(app.path('+')):
        engine.update(**paras)
        engine.set_matrix()
        result[i,1:]=eigsh(engine.matrix,k=app.ns,which='SA',return_eigenvectors=False)
    suffix='_%s'%(app.status.name)
    if app.save_data:
        np.savetxt('%s/%s_%s.dat'%(engine.dout,engine.status.const,app.status.name),result)
    if app.plot:
        plt.title('%s_%s'%(engine.status.const,app.status.name))
        plt.plot(result[:,0],result[:,1:])
        if app.show:
            plt.show()
        else:
            plt.savefig('%s/%s_%s.png'%(engine.dout,engine.status.const,app.status.name))
        plt.close()

class GF(HP.GF):
    '''
    The single-particle zero-temperature Green's function.
    Attribues:
        mask: ['nambu'] or []
            When ['nambu'], the anomalous Green's functions are not computed;
            When [], the anomalous Green's functions are also computed.
        nspin: 1 or 2
            When 1, sometimes(engine.basis.mode=='FS') only spin-down parts of the Green's function is computed;
            When 2, both spin-up and spin-down parts of the Green's function is computed.
        v0: 1D ndarray
            The initial guess of the groundstate.
        nstep: integer
            The max number of steps for the Lanczos iteration.
        method: 'user', 'python' or 'dense'
            It specifies the method the engine uses to compute the ground state.
            'user' for HamiltonianPy.Math.Lanczos.eig, 'python' for scipy.sparse.linalg.eigsh, and 'dense' for scipy.linalg.eigh.
        vtype: 'rd' or 'sy'
            When v0 is None, it specifies the type of the default initial guess of the groundstate.
            It only makes sense when method is 'user'. Then 'rd' means random ones and 'sy' for symmetric ones.
        tol: float
            The tolerance used to terminate the iteration.
        gse: float64
            The ground state energy of the system.
        coeff,hs: 4d ndarray
            The auxiliary data for the computing of GF.
    '''

    def __init__(self,mask=['nambu'],nspin=2,v0=None,nstep=200,method='python',vtype='rd',tol=0,**karg):
        '''
        Constructor.
        Parameters:
            mask: ['nambu'] or [], optional
                The flag to tell whether or not to compute the anomalous Green's function.
            nspin: 1 or 2, optional
                The flag to tell whether or not to compute only spin-down component.
            v0: 1D ndarray, optional
                The initial guess of the groundstate.
            nstep: integer, optional
                The max number of steps for the Lanczos iteration.
            method: 'user', 'python' or 'dense', optional
                The method the engine uses to compute the ground state.
            vtype: 'rd' or 'sy', optional
                When v0 is None, it specifies the type of the default initial guess of the groundstate.
            tol: float, optional
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

def EDGFP(engine,app):
    '''
    This method prepares the GF.
    '''
    # set the single particle operators and initialize gf.
    if engine.basis.mode in ('FG','FP'): assert app.nspin==2
    if len(app.operators)==0:
        table=engine.config.table(mask=[])
        if app.nspin==1: table=table.subset(select=lambda index: True if index.spin==0 else False)
        app.operators=GF.fsp_operators(table,engine.lattice)
        app.gf=np.zeros((app.nopt,app.nopt),dtype=np.complex128)
    # if the auxiliary data has been calculated before, recover it.
    if os.path.isfile('%s/%s_coeff.dat'%(engine.din,engine.status)):
        with open('%s/%s_coeff.dat'%(engine.din,engine.status),'rb') as fin:
            app.gse=pk.load(fin)
            app.coeff=pk.load(fin)
            app.hs=pk.load(fin)
        return
    # if the auxiliary data hasn't been calculated before, calculate it.
    app.gse=0.0
    app.coeff=np.zeros((2,app.nopt,app.nopt,app.nstep),dtype=np.complex128)
    app.hs=np.zeros((2,app.nopt,2,app.nstep),dtype=np.complex128)
    # set the matrix of engine.
    t0=time.time()
    engine.set_matrix()
    t1=time.time()
    print "GF preparation: matrix%s containing %s operators and %s non-zeros set in %fs."%(engine.matrix.shape,len(engine.operators['h']),engine.matrix.nnz,t1-t0)
    # get the ground state energy and ground state.
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
    t2=time.time()
    print 'GF preparation: GSE(=%f) calculated in %fs'%(app.gse,t2-t1)
    if engine.basis.mode in ('FS','FP'): engine.matrix=None
    # calculate the auxiliary data.
    print '-'*57
    print '%s%s%s%s'%('Time(seconds)'.center(13),' Preparation'.center(15),'Iteration'.center(15),'Total'.center(15))
    for h in xrange(2):
        t0=time.time()
        print ('Electron part' if h==0 else 'Hole part').center(13),
        sys.stdout.flush()
        t1=time.time()
        states,norms,lczs=[],[],[]
        for i,opt in enumerate(app.operators):
            if i==0:
                ed=engine.__replace_basis__(nambu=1-h,spin=0)
            if i==app.nopt/2 and app.nspin==2 and engine.basis.mode=='FS':
                ed=engine.__replace_basis__(nambu=1-h,spin=1)
            mat=HP.f_opt_rep(opt.dagger if h==0 else opt,basis=[engine.basis,ed.basis],transpose=True)
            state=mat.dot(app.v0)
            states.append(state)
            temp=norm(state)
            norms.append(temp)
            lczs.append(Lanczos(ed.matrix,v0=state/temp,check_normalization=False))
            print ('\b'*26 if i>0 else '')+('%s/%s(%es)'%(i,app.nopt,time.time()-t1)).center(25),
            sys.stdout.flush()
        t2=time.time()
        print '\b'*26+('%e'%(t2-t1)).center(14),
        sys.stdout.flush()
        for k in xrange(app.nstep):
            for i,(temp,lcz) in enumerate(zip(norms,lczs)):
                if not lcz.cut:
                    for j,state in enumerate(states):
                        if engine.basis.mode in ('FP','FG') or app.nspin==1 or (i<app.nopt/2 and j<app.nopt/2) or (i>=app.nopt/2 and j>=app.nopt/2):
                            app.coeff[h,i,j,k]=np.vdot(state,lcz.new)*temp
                    lcz.iter()
            print ('\b'*26 if k>0 else '')+('%s/%s(%es)'%(k,app.nstep,time.time()-t2)).center(25),
            sys.stdout.flush()
        t3=time.time()
        print '\b'*26+('%e'%(t3-t2)).center(14),
        sys.stdout.flush()
        for i,lcz in enumerate(lczs):
            app.hs[h,i,0,0:len(lcz.a)]=np.array(lcz.a)
            app.hs[h,i,1,0:len(lcz.b)]=np.array(lcz.b)
        t4=time.time()
        print ('%e'%(t4-t0)).center(14)
    print '-'*57
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
        buff=np.zeros((3,app.nstep),dtype=np.complex128)
        b=np.zeros(app.nstep,dtype=np.complex128)
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
    gf_mesh=np.zeros((app.ne,)+gf.gf.shape,dtype=np.complex128)
    for i,omega in enumerate(erange+engine.mu+1j*app.eta):
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
        if app.show:
            plt.show()
        else:
            plt.savefig('%s/%s_%s.png'%(engine.dout,engine.status,app.status.name))
        plt.close()
