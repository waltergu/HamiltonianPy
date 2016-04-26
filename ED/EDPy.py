'''
Exat diagonalization, including:
1) classes: ED
2) functions: EDGFC, EDGF, EDDOS, EDEB
'''

__all__=['ED','EDGFC','EDGF','EDDOS','EDEB']

from numpy import *
from ..Math import *
from ..Basics import *
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import norm,solve_banded,solveh_banded
from copy import deepcopy
import matplotlib.pyplot as plt
import os.path,sys,time

class ED(Engine):
    '''
    This provides the methods to get the sparse matrix representation on the occupation number basis of an electron system. 
    Attributes:
        ensemble: string
            The ensemble the system uses, 'c' for canonical ensemble and 'g' for grand canonical ensemble.
        filling: float
            The filling factor.
        mu: float
            The chemical potential.
            It makes sense only when ensemble is 'c'.
            It must be zero for grand canonical ensemble since the chemical potential has already been included in the Hamiltonian in this case.
        basis: BasisF
            The occupation number basis of the system.
            When ensemble is 'c', basis.basis_type must be 'FS' or 'FP' and when ensemble is 'g', basis.basis_type must be 'FG'.
        nspin: integer
            It makes sense only when basis.basis_type is 'FS'.
            It should be 1 or 2.
            When it is set to be 1, only spin-down parts of the Green's function is computed and when it is set to be 2, both spin-up and spin-down parts of the Green's function is computed.
        lattice: Lattice
            The lattice the system uses.
        config: Configuration
            The configuration of the degrees of freedom on the lattice.
        terms: list of Term
            The terms of the system.
            The weiss terms are not included in this list.
        nambu: logical
            A flag to tag whether the anomalous Green's function are computed.
        generators: dict of Generator
            It has only one entries:
            1) 'h': Generator
                The generator for the Hamiltonian including Weiss terms.
        operators: dict of OperatorCollection
            It has two entries:
            1) 'h': OperatorCollection
                The 'half' of the operators for the Hamiltonian, including Weiss terms.
            3) 'sp': OperatorCollection
                The single-particle operators in the lattice.
                When nspin is 1 and basis.basis_type is 'es', only spin-down single particle operators are included.
        matrix: csr_matrix
            The sparse matrix representation of the cluster Hamiltonian.
        cache: dict
            The cache during the process of calculation.
    Supported methods include:
        1) EDEB: calculates the energy spectrum.
        2) EDDOS: calculates the density of states.
        3) EDGFC: calculates the coefficients of single-particle Green's function.
        4) EDGF: calculates the single-particle Green's function.
    '''

    def __init__(self,ensemble='c',filling=0.5,mu=0,basis=None,nspin=1,lattice=None,config=None,terms=None,nambu=False,**karg):
        '''
        Constructor.
        '''
        self.ensemble=ensemble
        self.filling=filling
        self.mu=mu
        if self.ensemble.lower()=='c':
            self.name.update(const={'filling':self.filling})
        elif self.ensemble.lower()=='g':
            self.name.update(alter={'mu':self.mu})
        self.basis=basis
        self.nspin=nspin if basis.basis_type=='FS' else 2
        self.lattice=lattice
        self.config=config
        self.terms=terms
        self.nambu=nambu
        self.generators={}
        self.generators['h']=Generator(bonds=lattice.bonds,table=config.table(nambu=False),config=config,terms=terms)
        self.name.update(const=self.generators['h'].parameters['const'])
        self.name.update(alter=self.generators['h'].parameters['alter'])
        self.operators={}
        self.set_operators()
        self.cache={}

    def set_operators(self):
        '''
        Prepare self.operators.
        '''
        self.set_operators_hamiltonian()
        self.set_operators_single_particle()

    def set_operators_hamiltonian(self):
        self.operators['h']=self.generators['h'].operators

    def set_operators_single_particle(self):
        self.operators['sp']=OperatorCollection()
        table=self.config.table(nambu=self.nambu) if self.nspin==2 else subset(self.config.table(nambu=self.nambu),mask=lambda index: True if index.spin==0 else False)
        for index,sequence in table.iteritems():
            self.operators['sp']+=F_Linear(1,indices=[index],rcoords=[self.lattice.points[PID(scope=index.scope,site=index.site)].rcoord],icoords=[self.lattice.points[PID(scope=index.scope,site=index.site)].icoord],seqs=[sequence])

    def update(self,**karg):
        '''
        Update the alterable operators.
        '''
        for generator in self.generators.itervalues():
            generator.update(**karg)
        self.name.update(alter=self.generators['h'].parameters['alter'])
        self.set_operators_hamiltonian()

    def set_matrix(self):
        '''
        Set the csr_matrix representation of the Hamiltonian.
        '''
        self.matrix=csr_matrix((self.basis.nbasis,self.basis.nbasis),dtype=complex128)
        for operator in self.operators['h'].values():
            self.matrix+=opt_rep(operator,self.basis,transpose=False)
        self.matrix+=conjugate(transpose(self.matrix))
        self.matrix=self.matrix.T

    def gf(self,omega=None):
        '''
        Return the single particle Green's function of the system.
        '''
        if 'GF' not in self.apps:
            self.addapps(app=GF((len(self.operators['sp']),len(self.operators['sp'])),run=EDGF))
        if omega is not None:
            self.apps['GF'].omega=omega
            self.runapps('GF')
        return self.apps['GF'].gf

    def gf_mesh(self,omegas):
        '''
        Return the mesh of the single particle Green's functions of the system.
        '''
        if 'gf_mesh' in self.cache:
            return self.cache['gf_mesh']
        else:
            result=zeros((omegas.shape[0],len(self.operators['sp']),len(self.operators['sp'])),dtype=complex128)
            for i,omega in enumerate(omegas):
                result[i,:,:]=self.gf(omega)
            self.cache['gf_mesh']=result
            return result

def EDGFC(engine,app):
    nopt=len(engine.operators['sp'])
    if os.path.isfile(engine.din+'/'+engine.name.full+'_coeff.dat'):
        with open(engine.din+'/'+engine.name.full+'_coeff.dat','rb') as fin:
            app.gse=fromfile(fin,dtype=float64,count=1)
            app.coeff=fromfile(fin,dtype=complex128,count=2*nopt**2*app.nstep)
            app.hs=fromfile(fin,dtype=complex128)
        if len(app.coeff)==2*nopt**2*app.nstep and len(app.hs)==2*nopt*2*app.nstep:
            app.coeff=app.coeff.reshape((2,nopt,nopt,app.nstep))
            app.hs=app.hs.reshape((2,nopt,2,app.nstep))
            return
    app.coeff=zeros((2,nopt,nopt,app.nstep),dtype=complex128)
    app.hs=zeros((2,nopt,2,app.nstep),dtype=complex128)
    t0=time.time()
    engine.set_matrix()
    t1=time.time()
    print "EDGFC: matrix(%s*%s) containing %s operators and %s non-zeros set in %fs."%(engine.matrix.shape[0],engine.matrix.shape[1],len(engine.operators['h']),engine.matrix.nnz,t1-t0)
    if app.method in ('user',):
        app.gse,gs=Lanczos(engine.matrix,vtype=app.vtype,zero=app.error).eig(job='v',precision=app.error)    
    else:
        app.gse,gs=eigsh(engine.matrix,k=1,which='SA',return_eigenvectors=True,tol=app.error)
    t2=time.time()    
    print 'EDGFC: GSE(=%f) calculated in %fs'%(app.gse,t2-t1)
    if engine.basis.basis_type.lower() in ('fs','fp'): engine.matrix=None
    print '-'*57
    print '{0:13}{1:14}{2:14}{3:14}'.format('Time(seconds)','   Preparation','    Iteration','       Total')
    for h in xrange(2):
        t0=time.time()
        print '{0:13}'.format('Electron part' if h==0 else '  Hole part'),
        sys.stdout.flush()
        ed=ed_eh(engine,nambu=1-h,spin=0)
        states,norms,lczs=[],[],[]
        t1=time.time()
        for i,opt in enumerate(sorted(engine.operators['sp'].values(),key=lambda operator: operator.seqs[0])):
            mat=opt_rep(opt.dagger if h==0 else opt,[engine.basis,ed.basis],transpose=True)
            state=mat.dot(gs)
            states.append(state)
            temp=norm(state)
            norms.append(temp)
            lczs.append(Lanczos(ed.matrix,state/temp))
            print ('\b'*26 if i>0 else '')+'{0:25}'.format('  %s/%s(%es)'%(i,nopt,time.time()-t1)),
            sys.stdout.flush()
        t2=time.time()
        print '\b'*26+"{0:e}".format(t2-t1).center(14),
        sys.stdout.flush()
        for k in xrange(app.nstep):
            for i,(temp,lcz) in enumerate(zip(norms,lczs)):
                if not lcz.cut:
                    for j,state in enumerate(states):
                        if h==0:
                            app.coeff[h,j,i,k]=vdot(state,lcz.new)*temp
                        else:
                            app.coeff[h,i,j,k]=vdot(state,lcz.new)*temp
                    lcz.iter()
            print ('\b'*26 if k>0 else '')+'{0:25}'.format('  %s/%s(%es)'%(k,app.nstep,time.time()-t2)),
            sys.stdout.flush()
        t3=time.time()
        print '\b'*26+"{0:e}".format(t3-t2).center(14),
        sys.stdout.flush()
        for i,lcz in enumerate(lczs):
            app.hs[h,i,0,0:len(lcz.a)]=array(lcz.a)
            app.hs[h,i,1,0:len(lcz.b)]=array(lcz.b)
        t4=time.time()
        print "{0:e}".format(t4-t0).center(14)
    print '-'*57
    if app.save_data:
        with open(engine.din+'/'+engine.name.full+'_coeff.dat','wb') as fout:
            array(app.gse).tofile(fout)
            app.coeff.tofile(fout)
            app.hs.tofile(fout)

def ed_eh(self,nambu,spin):
    if self.basis.basis_type.lower()=='fg':
        return self
    elif self.basis.basis_type.lower()=='fp':
        result=deepcopy(self)
        if nambu==CREATION:
            result.basis=BasisF((self.basis.nstate,self.basis.nparticle+1))
        else:
            result.basis=BasisF((self.basis.nstate,self.basis.nparticle-1))
        result.matrix=csr_matrix((result.basis.nbasis,result.basis.nbasis),dtype=complex128)
        result.set_matrix()
        return result
    else:
        result=deepcopy(self)
        if nambu==CREATION and spin==0:
            result.basis=BasisF(up=(self.basis.nstate[0],self.basis.nparticle[0]),down=(self.basis.nstate[1],self.basis.nparticle[1]+1))
        elif nambu==ANNIHILATION and spin==0:
            result.basis=BasisF(up=(self.basis.nstate[0],self.basis.nparticle[0]),down=(self.basis.nstate[1],self.basis.nparticle[1]-1))
        elif nambu==CREATION and spin==1:
            result.basis=BasisF(up=(self.basis.nstate[0],self.basis.nparticle[0]+1),down=(self.basis.nstate[1],self.basis.nparticle[1]))
        else:
            result.basis=BasisF(up=(self.basis.nstate[0],self.basis.nparticle[0]-1),down=(self.basis.nstate[1],self.basis.nparticle[1]))
        result.matrix=csr_matrix((result.basis.nbasis,result.basis.nbasis),dtype=complex128)
        result.set_matrix()
        return result

def EDGF(engine,app):
    nmatrix=engine.apps['GFC'].nstep
    gse=engine.apps['GFC'].gse
    coeff=engine.apps['GFC'].coeff
    hs=engine.apps['GFC'].hs
    nopt=len(engine.operators['sp'])
    app.gf[...]=0.0
    buff=zeros((3,nmatrix),dtype=complex128)
    b=zeros(nmatrix,dtype=complex128)
    for h in xrange(2):
        for i in xrange(nopt):
            b[...]=0;b[0]=1
            buff[0,1:]=hs[h,i,1,0:nmatrix-1]*(-1)**(h+1)
            buff[1,:]=app.omega-(hs[h,i,0,:]-gse)*(-1)**h
            buff[2,:]=hs[h,i,1,:]*(-1)**(h+1)
            temp=solve_banded((1,1),buff,b,overwrite_ab=True,overwrite_b=True,check_finite=False)
            app.gf[i,:]+=dot(coeff[h,i,:,:],temp)

def EDDOS(engine,app):
    engine.cache.pop('gf_mesh',None)
    erange=linspace(app.emin,app.emax,num=app.ne)
    result=zeros((app.ne,2))
    result[:,0]=erange
    result[:,1]=-2*imag(trace(engine.gf_mesh(erange[:]+engine.mu+1j*app.eta),axis1=1,axis2=2))
    if app.save_data:
        savetxt(engine.dout+'/'+engine.name.full+'_DOS.dat',result)
    if app.plot:
        plt.title(engine.name.full+'_DOS')
        plt.plot(result[:,0],result[:,1])
        if app.show:
            plt.show()
        else:
            plt.savefig(engine.dout+'/'+engine.name.full+'_DOS.png')
        plt.close()

def EDEB(engine,app):
    result=zeros((app.path.rank.values()[0],app.ns+1))
    if len(app.path.rank)==1 and len(app.path.mesh.values()[0].shape)==1:
        result[:,0]=app.path.mesh.values()[0]
    else:
        result[:,0]=array(xrange(app.path.rank.values()[0]))
    for i,paras in enumerate(app.path('+')):
        engine.update(**paras)
        engine.set_matrix()
        result[i,1:]=eigsh(engine.matrix,k=app.ns,which='SA',return_eigenvectors=False)
    if app.save_data:
        savetxt(engine.dout+'/'+engine.name.const+'_EB.dat',result)
    if app.plot:
        plt.title(engine.name.const+'_EB')
        plt.plot(result[:,0],result[:,1:])
        if app.show:
            plt.show()
        else:
            plt.savefig(engine.dout+'/'+engine.name.const+'_EB.png')
        plt.close()
