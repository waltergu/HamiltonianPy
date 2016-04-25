from ONRPy import *
from VCA_Fortran import *
from Hamiltonian.Core.BasicAlgorithm.IntegrationPy import *
from Hamiltonian.Core.BasicAlgorithm.BerryCurvaturePy import *
from numpy.linalg import det,inv
from scipy import interpolate
from scipy.integrate import quad
from scipy.optimize import newton,brenth,brentq,broyden1,broyden2
class VCA(ONR):
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
        basis: BasisE
            The occupation number basis of the system.
            When ensemble is 'c', basis.basis_type must be 'ES' or 'EP' and when ensemble is 'g', basis.basis_type must be 'EG'.
        nspin: integer
            It makes sense only when basis.basis_type is 'ES'.
            It should be 1 or 2.
            When it is set to be 1, only spin-down parts of the Green's function is computed and when it is set to be 2, both spin-up and spin-down parts of the Green's function is computed.
        cell: Lattice
            The unit cell of the system.
        lattice: Lattice
            The cluster the system uses.
        terms: list of Term
            The terms of the system.
            The weiss terms are not included in this list.
        weiss: list of Term
            The Weiss terms of the system.
        nambu: logical
            A flag to tag whether the anomalous Green's function are computed.
        generators: dict of Generator
            It has three entries:
            1) 'h': Generator
                The generator for the cluster Hamiltonian including Weiss terms.
            2) 'pt_h': Generator
                The generator for the perturbation coming from the inter-cluster single-particle terms.
            3) 'pt_w': Generator
                The generator for the perturbation cominig from the Weiss terms.
        operators: dict of OperatorList
            It has four entries:
            1) 'h': OperatorList
                The 'half' of the operators for the cluster Hamiltonian,including Weiss terms.
            2) 'pt': OperatorList
                The 'half' of the operators for the perturbation, including Weiss terms.
            3) 'sp': OperatorList
                The single-particle operators in the cluster.
                When nspin is 1 and basis.basis_type is 'es', only spin-down single particle operators are included.
            4) 'csp': OperatorList
                The single-particle operators in the unit cell.
                When nspin is 1 and basis.basis_type is 'es', only spin-down single particle operators are included.
        clmap: dict
            This dict is used to restore the translation symmetry broken by the explicit tilling of the original lattice.
            It has two entries:
            1) 'seqs': 2D ndarray of integers
                clmap['seq'][i,j] stores the index sequence with respect to operators['sp'] of the j-th cluster-single-particle operator that corresponds to unit-cell-single-particle operator whose index sequence with respsect to operators['csp'] is i after the periodization;
            2) 'coords': 3D ndarray of floats
                clmap['seq'][i,j] stores the rcoords of the j-th cluster-single-particle operator that corresponds to unit-cell-single-particle operator whose index sequence with respsect to operator['csp'] is i after the periodization;
        matrix: csr_matrix
            The sparse matrix representation of the cluster Hamiltonian.
        cache: dict
            The cache during the process of calculation, usually to store some meshes.
    Supported methods include:
        1) VCAEB: calculates the energy bands.
        2) VCADOS: calculates the density of states.
        3) VCACP: calculates the chemical potential, behaves bad.
        4) VCAFF: calculates the filling factor.
        4) VCACN: calculates the Chern number and Berry curvature.
        5) VCAGP: calculates the grand potential.
        6) VCAGPS: calculates the grand potential surface.
        7) VCATEB: calculates the topological Hamiltonian's spectrum.
        8) VCAOP: calculates the order parameter.
        9) VCAFS: calculates the Fermi surface.
    '''

    def __init__(self,ensemble='c',filling=0.5,mu=0,basis=None,nspin=1,cell=None,lattice=None,terms=None,weiss=None,nambu=False,**karg):
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
        self.nspin=nspin if basis.basis_type=='ES' else 2
        self.cell=cell
        self.lattice=lattice
        self.terms=terms
        self.weiss=weiss
        self.nambu=nambu
        self.generators={}
        self.generators['h']=Generator(
                    bonds=      [bond for bond in lattice.bonds if bond.is_intra_cell()],
                    table=      lattice.table(nambu=False),
                    terms=      terms if weiss is None else terms+weiss,
                    nambu=      False,
                    half=       True
                    )
        self.generators['pt_h']=Generator(
                    bonds=      [bond for bond in lattice.bonds if not bond.is_intra_cell()],
                    table=      lattice.table(nambu=nambu) if self.nspin==2 else subset(lattice.table(nambu=nambu),mask=lambda index: True if index.spin==0 else False),
                    terms=      [term for term in terms if isinstance(term,Quadratic)],
                    nambu=      nambu,
                    half=       True
                    )
        self.generators['pt_w']=Generator(
                    bonds=      [bond for bond in lattice.bonds if bond.is_intra_cell()],
                    table=      lattice.table(nambu=nambu) if self.nspin==2 else subset(lattice.table(nambu=nambu),mask=lambda index: True if index.spin==0 else False),
                    terms=      None if weiss is None else [term*(-1) for term in weiss],
                    nambu=      nambu,
                    half=       True
                    )
        self.name.update(const=self.generators['h'].parameters['const'])
        self.name.update(alter=self.generators['h'].parameters['alter'])
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

    def set_operators_perturbation(self):
        self.operators['pt']=OperatorList()
        table=self.generators['pt_h'].table 
        for opt in self.generators['pt_h'].operators:
            if opt.indices[1] in table: self.operators['pt'].append(opt)
        for opt in self.generators['pt_w'].operators:
            if opt.indices[1] in table: self.operators['pt'].append(opt)

    def set_operators_cell_single_particle(self):
        self.operators['csp']=OperatorList()
        table=self.cell.table(nambu=self.nambu) if self.nspin==2 else subset(self.cell.table(nambu=self.nambu),mask=lambda index: True if index.spin==0 else False)
        for index,seq in table.iteritems():
            self.operators['csp'].append(E_Linear(1,indices=[index],rcoords=[self.cell.points[index.scope+str(index.site)].rcoord],icoords=[self.cell.points[index.scope+str(index.site)].icoord],seqs=[seq]))
        self.operators['csp'].sort(key=lambda opt: opt.seqs[0])

    def update(self,**karg):
        '''
        Update the alterable operators, such as the weiss terms.
        '''
        for generator in self.generators.itervalues():
            generator.update(**karg)
        self.name.update(alter=self.generators['h'].parameters['alter'])
        self.set_operators_hamiltonian()
        self.set_operators_perturbation()

    def set_clmap(self):
        '''
        Prepare self.clmap.
        '''
        nsp,ncsp,ndim=len(self.operators['sp']),len(self.operators['csp']),len(self.operators['csp'][0].rcoords[0])
        buff=[]
        for i in xrange(ncsp):
            buff.append(OperatorList())
        for optl in self.operators['sp']:
            for i,optc in enumerate(self.operators['csp']):
                if optl.indices[0].orbital==optc.indices[0].orbital and optl.indices[0].spin==optc.indices[0].spin and optl.indices[0].nambu==optc.indices[0].nambu and has_integer_solution(optl.rcoords[0]-optc.rcoords[0],self.cell.vectors):
                    buff[i].append(optl)
                    break
        self.clmap['seqs'],self.clmap['coords']=zeros((ncsp,nsp/ncsp),dtype=int64),zeros((ncsp,nsp/ncsp,ndim),dtype=float64)
        for i in xrange(ncsp):
            for j,optj in enumerate(buff[i]):
                self.clmap['seqs'][i,j],self.clmap['coords'][i,j,:]=optj.seqs[0]+1,optj.rcoords[0]

    def pt(self,k):
        '''
        Returns the matrix form of the perturbation.
        '''
        ngf=len(self.operators['sp'])
        result=zeros((ngf,ngf),dtype=complex128)
        for opt in self.operators['pt']:
            result[opt.seqs]+=opt.value*(1 if len(k)==0 else exp(-1j*inner(k,opt.icoords[0])))
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
    engine.cache.pop('pt_mesh',None)
    erange=linspace(app.emin,app.emax,app.ne)
    result=zeros((app.path.rank['k'],app.ne))
    for i,omega in enumerate(erange):
        result[:,i]=-2*imag((trace(engine.gf_vca_kmesh(omega+engine.mu+app.eta*1j,app.path.mesh['k']),axis1=1,axis2=2)))
    if app.save_data:
        buff=zeros((app.path.rank['k']*app.ne,3))
        for k in xrange(buff.shape[0]):
            i,j=divmod(k,app.path.rank['k'])
            buff[k,0]=j
            buff[k,1]=erange[i]
            buff[k,2]=result[j,i]
        savetxt(engine.dout+'/'+engine.name.full+'_EB.dat',buff)
    if app.plot:
        krange=array(xrange(app.path.rank['k']))
        plt.title(engine.name.full+'_EB')
        plt.colorbar(plt.pcolormesh(tensordot(krange,ones(app.ne),axes=0),tensordot(ones(app.path.rank['k']),erange,axes=0),result))
        if app.show:
            plt.show()
        else:
            plt.savefig(engine.dout+'/'+engine.name.full+'_EB.png')
        plt.close()

def VCAFF(engine,app):
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
    engine.cache.pop('pt_mesh',None)
    result=-2*imag((trace(engine.gf_vca_kmesh(engine.mu+app.eta*1j,app.BZ.mesh['k']),axis1=1,axis2=2)))
    if app.save_data:
        savetxt(engine.dout+'/'+engine.name.full+'_FS.dat',append(app.BZ.mesh['k'],result.reshape((app.BZ.rank['k'],1)),axis=1))
    if app.plot:
        plt.title(engine.name.full+"_FS")
        plt.axis('equal')
        N=int(round(sqrt(app.BZ.rank['k'])))
        plt.colorbar(plt.pcolormesh(app.BZ.mesh['k'][:,0].reshape((N,N)),app.BZ.mesh['k'][:,1].reshape(N,N),result.reshape(N,N)))
        if app.show:
            plt.show()
        else:
            plt.savefig(engine.dout+'/'+engine.name.full+'_FS.png')
        plt.close()

def VCADOS(engine,app):
    engine.cache.pop('pt_mesh',None)
    erange=linspace(app.emin,app.emax,app.ne)
    result=zeros((app.ne,2))
    for i,omega in enumerate(erange):
        result[i,0]=omega
        result[i,1]=-2*imag(sum((trace(engine.gf_vca_kmesh(omega+engine.mu+app.eta*1j,app.BZ.mesh['k']),axis1=1,axis2=2))))
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

def VCAGP(engine,app):
    engine.cache.pop('pt_mesh',None)
    ngf=len(engine.operators['sp'])
    app.gp=0
    fx=lambda omega: sum(log(abs(det(eye(ngf)-dot(engine.pt_mesh(app.BZ.mesh['k']),engine.gf(omega=omega*1j+engine.mu))))))
    app.gp=quad(fx,0,float(inf))[0]
    app.gp=(engine.apps['GFC'].gse-2/engine.nspin*app.gp/(pi*app.BZ.rank['k']))/engine.clmap['seqs'].shape[1]
    app.gp=app.gp+real(sum(trace(engine.pt_mesh(app.BZ.mesh['k']),axis1=1,axis2=2))/app.BZ.rank['k']/engine.clmap['seqs'].shape[1])
    app.gp=app.gp-engine.mu*engine.filling*len(engine.operators['csp'])*2/engine.nspin
    app.gp=app.gp/len(engine.cell.points)
    print 'gp:',app.gp

def VCAGPS(engine,app):
    ngf=len(engine.operators['sp'])
    result=zeros((product(app.BS.rank.values()),len(app.BS.mesh.keys())+1),dtype=float64)
    for i,paras in enumerate(app.BS('*')):
        print paras
        result[i,0:len(app.BS.mesh.keys())]=array(paras.values())
        engine.cache.pop('pt_mesh',None)
        engine.update(**paras)
        engine.runapps('GFC')
        engine.runapps('GP')
        result[i,len(app.BS.mesh.keys())]=engine.apps['GP'].gp
        print
    if app.save_data:
        savetxt(engine.dout+'/'+engine.name.const+'_GPS.dat',result)
    if app.plot:
        if len(app.BS.mesh.keys())==1:
            plt.title(engine.name.const+'_GPS')
            X=linspace(result[:,0].min(),result[:,0].max(),300)
            for i in xrange(1,result.shape[1]):
                tck=interpolate.splrep(result[:,0],result[:,i],k=3)
                Y=interpolate.splev(X,tck,der=0)
                plt.plot(X,Y)
            plt.plot(result[:,0],result[:,1],'r.')
            if app.show:
                plt.show()
            else:
                plt.savefig(engine.dout+'/'+engine.name.const+'_GPS.png')
            plt.close()

def VCACN(engine,app):
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
    if app.save_data:
        savetxt(engine.dout+'/'+engine.name.full+'_BC.dat',buff)
    if app.plot:
        nk=int(round(sqrt(app.BZ.rank['k'])))
        plt.title(engine.name.full+'_BC')
        plt.axis('equal')
        plt.colorbar(plt.pcolormesh(buff[:,0].reshape((nk,nk)),buff[:,1].reshape((nk,nk)),buff[:,2].reshape((nk,nk))))
        if app.show:
            plt.show()
        else:
            plt.savefig(engine.dout+'/'+engine.name.full+'_BC.png')
        plt.close()

def VCATEB(engine,app):
    engine.gf(omega=engine.mu)
    H=lambda kx,ky: -inv(engine.gf_vca(k=[kx,ky]))
    result=zeros((app.path.rank['k'],len(engine.operators['csp'])+1))
    for i,paras in enumerate(app.path()):
        result[i,0]=i
        result[i,1:]=eigh(H(paras['k'][0],paras['k'][1]),eigvals_only=True)
    if app.save_data:
        savetxt(engine.dout+'/'+engine.name.full+'_TEB.dat',result)
    if app.plot:
        plt.title(engine.name.full+'_TEB')
        plt.plot(result[:,0],result[:,1:])
        if app.show:
            plt.show()
        else:
            plt.savefig(engine.dout+'/'+engine.name.full+'_TEB.png')
        plt.close()

def VCAOP(engine,app):
    engine.cache.pop('pt_mesh',None)
    nmatrix=len(engine.operators['sp'])
    app.ms=zeros((len(app.terms),nmatrix,nmatrix),dtype=complex128)
    app.ops=zeros(len(app.terms))
    for i,term in enumerate(app.terms):
        buff=deepcopy(term);buff.value=1
        m=zeros((nmatrix,nmatrix),dtype=complex128)
        for opt in Generator(bonds=engine.lattice.bonds,table=engine.lattice.table(engine.nambu),terms=[buff],nambu=engine.nambu,half=True).operators:
            m[opt.seqs]+=opt.value
        m+=conjugate(m.T)
        app.ms[i,:,:]=m
    fx=lambda omega,m: (sum(trace(dot(engine.gf_mix_kmesh(omega=engine.mu+1j*omega,kmesh=app.BZ.mesh['k']),m),axis1=1,axis2=2)-trace(m)/(1j*omega-engine.mu-app.p))).real
    for i,m in enumerate(app.ms):
        app.ops[i]=quad(fx,0,float(inf),args=(m))[0]/app.BZ.rank['k']/nmatrix*2/pi
    for term,op in zip(app.terms,app.ops):
        print term.tag+':',op
