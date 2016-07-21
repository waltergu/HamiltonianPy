'''
Variational cluster approach, including:
1) classes: VCA
2) functions: VCAEB, VCAFF, VCACP, VCAFS, VCADOS, VCAGP, VCAGPM, VCACN, VCATEB, VCAOP
'''

__all__=['VCA','VCAEB','VCAFF','VCACP','VCAFS','VCADOS','VCAGP','VCAGPM','VCACN','VCATEB','VCAOP']

from numpy import *
from VCA_Fortran import *
from ..Basics import *
from ..ED import *
from ..Math import berry_curvature
from numpy.linalg import det,inv
from scipy.linalg import eigh
from scipy import interpolate
from scipy.sparse import csr_matrix
from scipy.integrate import quad
from scipy.optimize import minimize,newton,brenth,brentq,broyden1,broyden2
from copy import deepcopy
import matplotlib.pyplot as plt

class VCA(ED):
    '''
    This class implements the algorithm of the variational cluster approach of an electron system.
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
        cell: Lattice
            The unit cell of the system.
        celfig: Configuration
            The configuration of the degrees of freedom on the unit cell.
        lattice: Lattice
            The cluster the system uses.
        config: Configuration
            The configuration of the degrees of freedom on the lattice.
        terms: list of Term
            The terms of the system.
            The weiss terms are not included in this list.
        weiss: list of Term
            The Weiss terms of the system.
        nambu: logical
            A flag to tag whether the anomalous Green's function are computed.
        generators: dict of Generator
            It has four entries:
            1) 'h': Generator
                The generator for the cluster Hamiltonian, not including Weiss terms.
            2) 'h_w': Generator
                The generator for the cluster Hamiltonian of Weiss terms.
            3) 'pt_h': Generator
                The generator for the perturbation coming from the inter-cluster single-particle terms.
            4) 'pt_w': Generator
                The generator for the perturbation cominig from the Weiss terms.
        operators: dict of OperatorCollection
            It has five entries:
            1) 'h': OperatorCollection
                The 'half' of the operators for the cluster Hamiltonian, including Weiss terms.
            2) 'pt_h': OperatorCollection
                The 'half' of the operators for the perturbation. Not including Weiss terms.
            3) 'pt_w': OperatorCollection
                The 'half' of the operators for the perturbation of Weiss terms.
            4) 'sp': OperatorCollection
                The single-particle operators in the cluster.
                When nspin is 1 and basis.basis_type is 'FS', only spin-down single particle operators are included.
            5) 'csp': OperatorCollection
                The single-particle operators in the unit cell.
                When nspin is 1 and basis.basis_type is 'FS', only spin-down single particle operators are included.
        clmap: dict
            This dict is used to restore the translation symmetry broken by the explicit tiling of the original lattice.
            It has two entries:
            1) 'seqs': 2D ndarray of integers
            2) 'coords': 3D ndarray of floats
        matrix: csr_matrix
            The sparse matrix representation of the cluster Hamiltonian.
        cache: dict
            The cache during the process of calculation, usually to store some meshes.
    Supported methods include:
        1)  VCAEB: calculates the energy bands.
        2)  VCADOS: calculates the density of states.
        3)  VCACP: calculates the chemical potential, behaves bad.
        4)  VCAFF: calculates the filling factor.
        5)  VCACN: calculates the Chern number and Berry curvature.
        6)  VCAGP: calculates the grand potential.
        7)  VCAGPM: minimizes the grand potential.
        8)  VCATEB: calculates the topological Hamiltonian's spectrum.
        9)  VCAOP: calculates the order parameter.
        10) VCAFS: calculates the Fermi surface.
    '''

    def __init__(self,ensemble='c',filling=0.5,mu=0,basis=None,nspin=1,cell=None,celfig=None,lattice=None,config=None,terms=None,weiss=None,nambu=False,**karg):
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
        self.cell=cell
        self.celfig=celfig if celfig is not None else Configuration({key:config[key] for key in cell.points.keys()},priority=config.priority)
        self.lattice=lattice
        self.config=config
        self.terms=terms
        self.weiss=weiss
        self.nambu=nambu
        self.generators={}
        self.generators['h']=Generator(
                    bonds=      [bond for bond in lattice.bonds if bond.is_intra_cell()],
                    config=     config,
                    table=      config.table(nambu=False),
                    terms=      terms
                    )
        self.generators['h_w']=Generator(
                    bonds=      [bond for bond in lattice.bonds],
                    config=     config,
                    table=      config.table(nambu=False),
                    terms=      weiss
                    )
        self.generators['pt_h']=Generator(
                    bonds=      [bond for bond in lattice.bonds if not bond.is_intra_cell()],
                    config=     config,
                    table=      config.table(nambu=nambu) if self.nspin==2 else subset(config.table(nambu=nambu),mask=lambda index: True if index.spin==0 else False),
                    terms=      [term for term in terms if isinstance(term,Quadratic)],
                    )
        self.generators['pt_w']=Generator(
                    bonds=      [bond for bond in lattice.bonds],
                    config=     config,
                    table=      config.table(nambu=nambu) if self.nspin==2 else subset(config.table(nambu=nambu),mask=lambda index: True if index.spin==0 else False),
                    terms=      None if weiss is None else [term*(-1) for term in weiss],
                    )
        self.name.update(const=self.generators['h'].parameters['const'])
        self.name.update(alter=self.generators['h'].parameters['alter'])
        self.name.update(const=self.generators['h_w'].parameters['const'])
        self.name.update(alter=self.generators['h_w'].parameters['alter'])
        self.operators={}
        self.set_operators()
        self.clmap={}
        self.set_clmap()
        self.cache={}

    def set_operators(self):
        '''
        Prepare self.operators.
        '''
        self.set_operators_hamiltonian()
        self.set_operators_perturbation()
        self.set_operators_single_particle()
        self.set_operators_cell_single_particle()

    def set_operators_hamiltonian(self):
        self.operators['h']=self.generators['h'].operators+self.generators['h_w'].operators

    def set_operators_perturbation(self):
        table=self.generators['pt_h'].table
        self.operators['pt_h']=OperatorCollection()
        for opt in self.generators['pt_h'].operators.values():
            if opt.indices[1] in table: self.operators['pt_h']+=opt
        table=self.generators['pt_w'].table 
        self.operators['pt_w']=OperatorCollection()
        for opt in self.generators['pt_w'].operators.values():
            if opt.indices[1] in table: self.operators['pt_w']+=opt

    def set_operators_cell_single_particle(self):
        self.operators['csp']=OperatorCollection()
        temp=self.celfig.table(nambu=self.nambu)
        table=temp if self.nspin==2 else subset(temp,mask=lambda index: True if index.spin==0 else False)
        for index,seq in table.iteritems():
            pid=PID(scope=index.scope,site=index.site)
            self.operators['csp']+=F_Linear(1,indices=[index],rcoords=[self.cell.points[pid].rcoord],icoords=[self.cell.points[pid].icoord],seqs=[seq])

    def update(self,**karg):
        '''
        Update the alterable operators, such as the weiss terms.
        '''
        for generator in self.generators.itervalues():
            generator.update(**karg)
        self.name.update(alter=self.generators['h'].parameters['alter'])
        self.name.update(alter=self.generators['h_w'].parameters['alter'])
        self.set_operators_hamiltonian()
        self.set_operators_perturbation()

    def set_clmap(self):
        '''
        Prepare self.clmap.
        '''
        nsp,ncsp,ndim=len(self.operators['sp']),len(self.operators['csp']),len(self.operators['csp'].values()[0].rcoords[0])
        buff=[]
        for i in xrange(ncsp):
            buff.append(OperatorCollection())
        optsl=sorted(self.operators['sp'].values(),key=lambda operator: operator.seqs[0])
        optsc=sorted(self.operators['csp'].values(),key=lambda operator: operator.seqs[0])
        for optl in optsl:
            for i,optc in enumerate(optsc):
                if optl.indices[0].orbital==optc.indices[0].orbital and optl.indices[0].spin==optc.indices[0].spin and optl.indices[0].nambu==optc.indices[0].nambu and has_integer_solution(optl.rcoords[0]-optc.rcoords[0],self.cell.vectors):
                    buff[i]+=optl
                    break
        self.clmap['seqs'],self.clmap['coords']=zeros((ncsp,nsp/ncsp),dtype=int64),zeros((ncsp,nsp/ncsp,ndim),dtype=float64)
        for i in xrange(ncsp):
            for j,optj in enumerate(sorted(buff[i].values(),key=lambda operator: operator.seqs[0])):
                self.clmap['seqs'][i,j],self.clmap['coords'][i,j,:]=optj.seqs[0]+1,optj.rcoords[0]

    def pt(self,k):
        '''
        Returns the matrix form of the perturbation.
        '''
        ngf=len(self.operators['sp'])
        result=zeros((ngf,ngf),dtype=complex128)
        for opt in self.operators['pt_h'].values():
            result[opt.seqs]+=opt.value*(1 if len(k)==0 else exp(-1j*inner(k,opt.icoords[0])))
        for opt in self.operators['pt_w'].values():
            result[opt.seqs]+=opt.value
        return result+conjugate(result.T)

    def pt_mesh(self,kmesh):
        '''
        Returns the mesh of the perturbation.
        '''
        if 'pt_mesh' in self.cache:
            return self.cache['pt_mesh']
        else:
            result=zeros((kmesh.shape[0],len(self.operators['sp']),len(self.operators['sp'])),dtype=complex128)
            for i,k in enumerate(kmesh):
                result[i,:,:]=self.pt(k)
            self.cache['pt_mesh']=result
            return result

    def gf_mix(self,omega=None,k=[]):
        '''
        Returns the Green's function in the mixed representation.
        '''
        ngf,gf=len(self.operators['sp']),self.gf(omega)
        return dot(gf,inv(identity(ngf,dtype=complex128)-dot(self.pt(k),gf)))

    def gf_mix_kmesh(self,omega,kmesh):
        '''
        Returns the mesh of the Green's functions in the mixed representation.
        '''
        ngf,gf=len(self.operators['sp']),self.gf(omega)
        return einsum('jk,ikl->ijl',gf,inv(identity(ngf,dtype=complex128)-dot(self.pt_mesh(kmesh),gf)))

    def gf_vca(self,omega=None,k=[]):
        '''
        Returns the single particle Green's function of the system.
        '''
        ngf,ngf_vca,gf=len(self.operators['sp']),len(self.operators['csp']),self.gf(omega)
        return gf_contract(k=k,gf_buff=dot(gf,inv(identity(ngf,dtype=complex128)-dot(self.pt(k),gf))),seqs=self.clmap['seqs'],coords=self.clmap['coords'])/(ngf/ngf_vca)

    def gf_vca_kmesh(self,omega,kmesh):
        '''
        Returns the mesh of the single particle Green's functions of the system.
        '''
        ngf,ngf_vca,gf=len(self.operators['sp']),len(self.operators['csp']),self.gf(omega)
        buff=einsum('jk,ikl->ijl',gf,inv(identity(ngf,dtype=complex128)-dot(self.pt_mesh(kmesh),gf)))
        result=zeros((kmesh.shape[0],ngf_vca,ngf_vca),dtype=complex128)
        for n,k in enumerate(kmesh):
            result[n,:,:]=gf_contract(k=k,gf_buff=buff[n,:,:],seqs=self.clmap['seqs'],coords=self.clmap['coords'])
        return result/(ngf/ngf_vca)

def has_integer_solution(coords,vectors):
    nvectors=len(vectors)
    ndim=len(vectors[0])
    a=zeros((3,3))
    for i in xrange(nvectors):
        a[0:ndim,i]=vectors[i]
    if nvectors==2:
        if ndim==2:
            buff=zeros(3)
            buff[2]=cross(vectors[0],vectors[1])
        else:
            buff=cross(vectors[0],vectors[1])
        a[:,2]=buff
    if nvectors==1:
        buff1=a[:,0]
        for i in xrange(3):
            buff2=zeros(3)
            buff2[i]=pi
            if not is_parallel(buff1,buff2): break
        buff3=cross(buff1,buff2)
        a[:,1]=buff2
        a[:,2]=buff3
    b=zeros(3)
    b[0:len(coords)]=coords
    x=dot(inv(a),b)
    if max(abs(x-around(x)))<RZERO:
        return True
    else:
        return False

def VCAEB(engine,app):
    engine.rundependence(app.id)
    engine.cache.pop('pt_mesh',None)
    erange=linspace(app.emin,app.emax,app.ne)
    result=zeros((app.path.rank['k'],app.ne))
    for i,omega in enumerate(erange):
        result[:,i]=-2*imag((trace(engine.gf_vca_kmesh(omega+engine.mu+app.eta*1j,app.path.mesh['k']),axis1=1,axis2=2)))
    suffix='_'+(app.__class__.__name__ if id(app)==app.id else str(app.id))
    if app.save_data:
        buff=zeros((app.path.rank['k']*app.ne,3))
        for k in xrange(buff.shape[0]):
            i,j=divmod(k,app.path.rank['k'])
            buff[k,0]=j
            buff[k,1]=erange[i]
            buff[k,2]=result[j,i]
        savetxt(engine.dout+'/'+engine.name.full+suffix+'.dat',buff)
    if app.plot:
        krange=array(xrange(app.path.rank['k']))
        plt.title(engine.name.full+suffix)
        plt.colorbar(plt.pcolormesh(tensordot(krange,ones(app.ne),axes=0),tensordot(ones(app.path.rank['k']),erange,axes=0),result))
        if app.show:
            plt.show()
        else:
            plt.savefig(engine.dout+'/'+engine.name.full+suffix+'.png')
        plt.close()

def VCAFF(engine,app):
    engine.rundependence(app.id)
    engine.cache.pop('pt_mesh',None)
    nk,nmatrix=app.BZ.rank['k'],len(engine.operators['sp'])
    fx=lambda omega: (sum(trace(engine.gf_mix_kmesh(omega=engine.mu+1j*omega,kmesh=app.BZ.mesh['k']),axis1=1,axis2=2)-nmatrix/(1j*omega-engine.mu-app.p))).real
    app.filling=quad(fx,0,float(inf))[0]/nk/nmatrix/pi
    engine.filling=app.filling
    print 'Filling factor:',app.filling

def VCACP(engine,app):
    engine.cache.pop('pt_mesh',None)
    nk,nmatrix=app.BZ.rank['k'],len(engine.operators['sp'])
    fx=lambda omega,mu: (sum(trace(engine.gf_mix_kmesh(omega=mu+1j*omega,kmesh=app.BZ.mesh['k']),axis1=1,axis2=2)-nmatrix/(1j*omega-mu-app.p))).real
    gx=lambda mu: quad(fx,0,float(inf),args=(mu))[0]/nk/nmatrix/pi-engine.filling
    app.mu=broyden2(gx,engine.mu,verbose=True,reduction_method='svd',maxiter=20,x_tol=app.error)
    engine.mu=app.mu
    print 'mu,error:',engine.mu,gx(engine.mu)

def VCAFS(engine,app):
    engine.rundependence(app.id)
    engine.cache.pop('pt_mesh',None)
    result=-2*imag((trace(engine.gf_vca_kmesh(engine.mu+app.eta*1j,app.BZ.mesh['k']),axis1=1,axis2=2)))
    suffix='_'+(app.__class__.__name__ if id(app)==app.id else str(app.id))
    if app.save_data:
        savetxt(engine.dout+'/'+engine.name.full+suffix+'.dat',append(app.BZ.mesh['k'],result.reshape((app.BZ.rank['k'],1)),axis=1))
    if app.plot:
        plt.title(engine.name.full+suffix)
        plt.axis('equal')
        N=int(round(sqrt(app.BZ.rank['k'])))
        plt.colorbar(plt.pcolormesh(app.BZ.mesh['k'][:,0].reshape((N,N)),app.BZ.mesh['k'][:,1].reshape(N,N),result.reshape(N,N)))
        if app.show:
            plt.show()
        else:
            plt.savefig(engine.dout+'/'+engine.name.full+suffix+'.png')
        plt.close()

def VCADOS(engine,app):
    engine.rundependence(app.id)
    engine.cache.pop('pt_mesh',None)
    erange=linspace(app.emin,app.emax,app.ne)
    result=zeros((app.ne,2))
    for i,omega in enumerate(erange):
        result[i,0]=omega
        result[i,1]=-2*imag(sum((trace(engine.gf_vca_kmesh(omega+engine.mu+app.eta*1j,app.BZ.mesh['k']),axis1=1,axis2=2))))
    suffix='_'+(app.__class__.__name__ if id(app)==app.id else str(app.id))
    if app.save_data:
        savetxt(engine.dout+'/'+engine.name.full+suffix+'.dat',result)
    if app.plot:
        plt.title(engine.name.full+suffix)
        plt.plot(result[:,0],result[:,1])
        if app.show:
            plt.show()
        else:
            plt.savefig(engine.dout+'/'+engine.name.full+suffix+'.png')
        plt.close()

def VCAGP(engine,app):
    engine.rundependence(app.id)
    engine.cache.pop('pt_mesh',None)
    ngf=len(engine.operators['sp'])
    app.gp=0
    fx=lambda omega: sum(log(abs(det(eye(ngf)-dot(engine.pt_mesh(app.BZ.mesh['k']),engine.gf(omega=omega*1j+engine.mu))))))
    app.gp=quad(fx,0,float(inf))[0]
    app.gp=(engine.apps['GFC'].gse-2/engine.nspin*app.gp/(pi*app.BZ.rank['k']))/engine.clmap['seqs'].shape[1]
    app.gp=app.gp+real(sum(trace(engine.pt_mesh(app.BZ.mesh['k']),axis1=1,axis2=2))/app.BZ.rank['k']/engine.clmap['seqs'].shape[1])
    app.gp=app.gp-engine.mu*engine.filling*len(engine.operators['csp'])*2/engine.nspin
    app.gp=app.gp/len(engine.cell.points)
    print 'gp(%s): %s'%(', '.join(['%s:%s'%(key,value) for key,value in engine.name.parameters.items()]),app.gp)

def VCAGPM(engine,app):
    def gp(values,keys):
        engine.cache.pop('pt_mesh',None)
        engine.update(**{key:value for key,value in zip(keys,values)})
        engine.rundependence(app.id)
        print
        return engine.apps['GP'].gp
    suffix='_'+(app.__class__.__name__ if id(app)==app.id else str(app.id))
    if isinstance(app.BS,BaseSpace):
        nbs=len(app.BS.mesh.keys())
        result=zeros((product(app.BS.rank.values()),nbs+1),dtype=float64)
        for i,paras in enumerate(app.BS('*')):
            print paras
            result[i,0:nbs]=array(paras.values())
            result[i,nbs]=gp(paras.values(),paras.keys())
        app.gpm=amin(result[:,nbs])
        index=argmin(result[:,nbs])
        app.bsm={key:value for key,value in zip(paras.keys(),result[index,0:nbs])}
        print 'Minimum value(%s) at point %s'%(app.gpm,app.bsm)
        if app.save_data:
            savetxt(engine.dout+'/'+engine.name.const+suffix+'.dat',result)
        if app.plot:
            if len(app.BS.mesh.keys())==1:
                plt.title(engine.name.const+suffix)
                X=linspace(result[:,0].min(),result[:,0].max(),300)
                for i in xrange(1,result.shape[1]):
                    tck=interpolate.splrep(result[:,0],result[:,i],k=3)
                    Y=interpolate.splev(X,tck,der=0)
                    plt.plot(X,Y)
                plt.plot(result[:,0],result[:,1],'r.')
                if app.show:
                    plt.show()
                else:
                    plt.savefig(engine.dout+'/'+engine.name.const+suffix+'.png')
                plt.close()
    else:
        temp=minimize(gp,app.BS.values(),args=(app.BS.keys()),method=app.method,options=app.options)
        app.bsm,app.gpm={key:value for key,value in zip(app.BS.keys(),temp.x)},temp.fun
        print 'Minimum value(%s) at point %s'%(app.gpm,app.bsm)
        if app.save_data:
            result=array([app.bsm.values()+[app.gpm]])
            if app.fout is None:
                savetxt(engine.dout+'/'+engine.name.const+suffix+'.dat',result)
            else:
                import os
                if os.path.isfile(app.fout):
                    with open(app.fout,'a') as fout:
                        fout.write(' '.join(['%.18e'%data for data in result[0,:]]))
                        fout.write('\n')
                else:
                    savetxt(app.fout,result)

def VCACN(engine,app):
    engine.rundependence(app.id)
    engine.gf(omega=engine.mu)
    H=lambda kx,ky: -inv(engine.gf_vca(k=[kx,ky]))
    app.bc=zeros(app.BZ.rank['k'])
    for i,paras in enumerate(app.BZ()):
        app.bc[i]=berry_curvature(H,paras['k'][0],paras['k'][1],0,d=app.d)
    print 'Chern number(mu):',app.cn,'(',engine.mu,')'
    if app.save_data or app.plot:
        buff=zeros((app.BZ.rank['k'],3))
        buff[:,0:2]=app.BZ.mesh['k']
        buff[:,2]=app.bc
    suffix='_'+(app.__class__.__name__ if id(app)==app.id else str(app.id))
    if app.save_data:
        savetxt(engine.dout+'/'+engine.name.full+suffix+'.dat',buff)
    if app.plot:
        nk=int(round(sqrt(app.BZ.rank['k'])))
        plt.title(engine.name.full+suffix)
        plt.axis('equal')
        plt.colorbar(plt.pcolormesh(buff[:,0].reshape((nk,nk)),buff[:,1].reshape((nk,nk)),buff[:,2].reshape((nk,nk))))
        if app.show:
            plt.show()
        else:
            plt.savefig(engine.dout+'/'+engine.name.full+suffix+'.png')
        plt.close()

def VCATEB(engine,app):
    engine.rundependence(app.id)
    engine.gf(omega=engine.mu)
    H=lambda kx,ky: -inv(engine.gf_vca(k=[kx,ky]))
    result=zeros((app.path.rank['k'],len(engine.operators['csp'])+1))
    for i,paras in enumerate(app.path()):
        result[i,0]=i
        result[i,1:]=eigh(H(paras['k'][0],paras['k'][1]),eigvals_only=True)
    suffix='_'+(app.__class__.__name__ if id(app)==app.id else str(app.id))
    if app.save_data:
        savetxt(engine.dout+'/'+engine.name.full+suffix+'.dat',result)
    if app.plot:
        plt.title(engine.name.full+suffix)
        plt.plot(result[:,0],result[:,1:])
        if app.show:
            plt.show()
        else:
            plt.savefig(engine.dout+'/'+engine.name.full+suffix+'.png')
        plt.close()

def VCAOP(engine,app):
    engine.rundependence(app.id)
    engine.cache.pop('pt_mesh',None)
    nmatrix=len(engine.operators['sp'])
    app.ms=zeros((len(app.terms),nmatrix,nmatrix),dtype=complex128)
    app.ops=zeros(len(app.terms))
    for i,term in enumerate(app.terms):
        buff=deepcopy(term);buff.value=1
        m=zeros((nmatrix,nmatrix),dtype=complex128)
        for opt in Generator(bonds=engine.lattice.bonds,config=engine.config,table=engine.config.table(nambu=engine.nambu),terms=[buff]).operators.values():
            m[opt.seqs]+=opt.value
        m+=conjugate(m.T)
        app.ms[i,:,:]=m
    fx=lambda omega,m: (sum(trace(dot(engine.gf_mix_kmesh(omega=engine.mu+1j*omega,kmesh=app.BZ.mesh['k']),m),axis1=1,axis2=2)-trace(m)/(1j*omega-engine.mu-app.p))).real
    for i,m in enumerate(app.ms):
        app.ops[i]=quad(fx,0,float(inf),args=(m))[0]/app.BZ.rank['k']/nmatrix*2/pi
    for term,op in zip(app.terms,app.ops):
        print term.id+':',op
