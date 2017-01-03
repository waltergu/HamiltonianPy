'''
Variational cluster approach, including:
1) classes: VCA, EB, GPM, CP, FF, OP
s2) functions: VCAEB, VCADOS, VCAFS, VCABC, VCATEB, VCAGP, VCAGPM, VCACP, VCAFF, VCAOP
'''

__all__=['VCA','EB','VCAEB','VCADOS','VCAFS','VCABC','VCATEB','VCAGP','GPM','VCAGPM','CP','VCACP','FF','VCAFF','OP','VCAOP']

from VCA_Fortran import gf_contract
from numpy.linalg import det,inv
from scipy.linalg import eigh
from scipy import interpolate
from scipy.sparse import csr_matrix
from scipy.integrate import quad
from scipy.optimize import minimize,newton,brenth,brentq,broyden1,broyden2
from copy import deepcopy
import numpy as np
import HamiltonianPy as HP
import HamiltonianPy.ED as ED
import matplotlib.pyplot as plt
import os

class VCA(ED.ED):
    '''
    This class implements the algorithm of the variational cluster approach of an electron system.
    Attributes:
        preloads: 2-list
            preloads[0]: HP.ED.GF
                The cluster Green's function.
            preloads[1]: HP.GF
                The VCA Green's function.
        basis: BasisF
            The occupation number basis of the system.
        filling: float
            The filling factor.
        mu: float
            The chemical potential.
        cell: Lattice
            The unit cell of the system.
        lattice: Lattice
            The cluster the system uses.
        config: IDFConfig
            The configuration of the degrees of freedom on the lattice.
        terms: list of Term
            The terms of the system.
            The weiss terms are not included in this list.
        weiss: list of Term
            The Weiss terms of the system.
        dtype: np.float64, np.complex128
            The data type of the matrix representation of the Hamiltonian.
        generators: dict of Generator
            It has four entries:
            1) 'h': Generator
                The generator for the cluster Hamiltonian, not including the Weiss terms.
            2) 'h_w': Generator
                The generator for the cluster Hamiltonian of the Weiss terms.
            3) 'pt_h': Generator
                The generator for the perturbation coming from the inter-cluster single-particle terms.
            4) 'pt_w': Generator
                The generator for the perturbation cominig from the Weiss terms.
        operators: dict of OperatorCollection
            It has three entries:
            1) 'h': OperatorCollection
                The 'half' of the operators for the cluster Hamiltonian, including the Weiss terms.
            2) 'pt_h': OperatorCollection
                The 'half' of the operators for the perturbation, not including Weiss terms.
            3) 'pt_w': OperatorCollection
                The 'half' of the operators for the perturbation of Weiss terms.
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
        1)  VCAEB: calculates the single particle spectrum along a path in Brillouin zone.
        2)  VCADOS: calculates the single particle density of states.
        3)  VCAFS: calculates the single particle spectrum at the Fermi surface.
        4)  VCABC: calculates the Berry curvature and Chern number based on the so-called topological Hamiltonian (PRX 2, 031008 (2012)).
        5)  VCATEB: calculates the topological Hamiltonian's spectrum.
        6)  VCAGP: calculates the grand potential.
        7)  VCAGPM: minimizes the grand potential.
        8)  VCACP: calculates the chemical potential, behaves bad.
        9)  VCAFF: calculates the filling factor.
        10) VCAOP: calculates the order parameter.
    '''

    def __init__(self,cgf,basis=None,filling=0.5,mu=0,cell=None,lattice=None,config=None,terms=None,weiss=None,dtype=np.complex128,**karg):
        '''
        Constructor.
        Parameters:
            cgf: HP.ED.GF
                The cluster Green's function.
            basis: BasisF, optional
                The occupation number basis of the system.
            filling: float, optional
                The filling factor.
            mu: float, optional
                The chemical potential.
            cell: Lattice, optional
                The unit cell of the system.
            lattice: Lattice, optional
                The lattice of the system.
            config: IDFConfig, optional
                The configuration of the internal degrees of freedom on the lattice.
            terms: list of Term, optional
                The terms of the system.
            weiss: lsit of Term, optional
                The Weiss terms of the system.
            dtype: np.float64, np.complex128
                The data type of the matrix representation of the Hamiltonian.
        '''
        # initialize the cgf and gf
        assert isinstance(cgf,ED.GF)
        gf=HP.GF()
        cgf.reinitialization(operators=HP.GF.fsp_operators(table=cgf.table(config),lattice=lattice))
        gf.reinitialization(operators=HP.GF.fsp_operators(table=cgf.table(config.subset(cell.__contains__)),lattice=cell))
        self.preloads.extend([cgf,gf])
        # initialize the ordinary attributes
        nspin,mask=cgf.nspin,cgf.mask
        self.basis=basis
        if basis.mode=='FG':
            assert nspin==2
            self.filling=filling
            self.mu=[term for term in terms if term.id=='mu'][0].value
            self.status.update(alter={'mu':self.mu})
        else:
            if basis.mode=='FP':assert nspin==2
            self.filling=1.0*basis.nparticle.sum()/basis.nstate.sum()
            self.mu=mu
            self.status.update(const={'filling':self.filling})
        self.cell=cell
        self.lattice=lattice
        self.config=config
        self.terms=terms
        self.weiss=weiss
        self.dtype=dtype
        # initialize the generators
        self.generators={}
        self.generators['h']=HP.Generator(
            bonds=      [bond for bond in lattice.bonds if bond.is_intra_cell()],
            config=     config,
            table=      config.table(mask=['nambu']),
            terms=      terms
            )
        self.generators['h_w']=HP.Generator(
            bonds=      [bond for bond in lattice.bonds],
            config=     config,
            table=      config.table(mask=['nambu']),
            terms=      weiss
            )
        self.generators['pt_h']=HP.Generator(
            bonds=      [bond for bond in lattice.bonds if not bond.is_intra_cell()],
            config=     config,
            table=      config.table(mask=mask) if nspin==2 else config.table(mask=mask).subset(select=lambda index: True if index.spin==0 else False),
            terms=      [term for term in terms if isinstance(term,HP.Quadratic)],
            )
        self.generators['pt_w']=HP.Generator(
            bonds=      [bond for bond in lattice.bonds],
            config=     config,
            table=      config.table(mask=mask) if nspin==2 else config.table(mask=mask).subset(select=lambda index: True if index.spin==0 else False),
            terms=      None if weiss is None else [term*(-1) for term in weiss],
            )
        # update the status
        self.status.update(const=self.generators['h'].parameters['const'])
        self.status.update(alter=self.generators['h'].parameters['alter'])
        self.status.update(const=self.generators['h_w'].parameters['const'])
        self.status.update(alter=self.generators['h_w'].parameters['alter'])
        # initialize the operators
        self.operators={}
        self.operators['h']=self.generators['h'].operators+self.generators['h_w'].operators
        self.operators['pt_h']=self.generators['pt_h'].operators
        self.operators['pt_w']=self.generators['pt_w'].operators
        self.set_clmap()
        self.cache={}

    def set_clmap(self):
        '''
        Prepare self.clmap.
        '''
        self.clmap={}
        cgf,gf=self.preloads
        buff=[[] for i in xrange(gf.nopt)]
        for copt in cgf.operators:
            for i,opt in enumerate(gf.operators):
                if copt.indices[0].iid==opt.indices[0].iid and HP.belong_to_lattice(copt.rcoords[0]-opt.rcoords[0],self.cell.vectors):
                    buff[i].append(copt)
                    break
        self.clmap['seqs']=np.zeros((gf.nopt,cgf.nopt/gf.nopt),dtype=np.int64)
        self.clmap['coords']=np.zeros((gf.nopt,cgf.nopt/gf.nopt,len(gf.operators[0].rcoords[0])),dtype=np.float64)
        for i in xrange(gf.nopt):
            for j,opt in enumerate(sorted(buff[i],key=lambda operator: operator.seqs[0])):
                self.clmap['seqs'][i,j]=opt.seqs[0]+1
                self.clmap['coords'][i,j,:]=opt.rcoords[0]

    def update(self,**karg):
        '''
        Update the alterable operators, such as the weiss terms.
        '''
        for generator in self.generators.itervalues():
            generator.update(**karg)
        self.operators['h']=self.generators['h'].operators+self.generators['h_w'].operators
        self.operators['pt_h']=self.generators['pt_h'].operators
        self.operators['pt_w']=self.generators['pt_w'].operators
        self.status.update(alter=self.generators['h'].parameters['alter'])
        self.status.update(alter=self.generators['h_w'].parameters['alter'])

    def cgf(self,omega=None):
        '''
        Return the cluster Green's function.
        Parameter:
            omega: np.complex128, optional
                The frequency of the cluster Green's function.
        Returns: 2d ndarray
            The cluster Green's function.
        '''
        app=self.preloads[0]
        if omega is not None:
            app.omega=omega
            app.run(self,app)
        return app.gf

    def pt(self,k=[]):
        '''
        Returns the matrix form of the inter-cluster perturbations.
        Parameter:
            k: 1d ndarray like, optional
                The momentum of the inter-cluster perturbations.
        Returns: 2d ndarray
            The matrix form of the inter-cluster perturbations.
        '''
        result=np.zeros(self.preloads[0].gf.shape,dtype=np.complex128)
        for opt in self.operators['pt_h'].values():
            result[opt.seqs]+=opt.value*(1 if len(k)==0 else np.exp(-1j*np.inner(k,opt.icoords[0])))
        for opt in self.operators['pt_w'].values():
            result[opt.seqs]+=opt.value
        return result+result.T.conjugate()

    def pt_kmesh(self,kmesh):
        '''
        Returns the mesh of the inter-cluster perturbations.
        Parameters:
            kmesh: (n+1)d ndarray like
                The kmesh of the inter-cluster perturbations.
                And n is the spatial dimension of the system.
        Returns: 3d ndarray
            The pt mesh.
        '''
        if 'pt_kmesh' in self.cache:
            return self.cache['pt_kmesh']
        else:
            result=np.zeros((kmesh.shape[0],)+self.preloads[0].gf.shape,dtype=np.complex128)
            for i,k in enumerate(kmesh):
                result[i,:,:]=self.pt(k)
            self.cache['pt_kmesh']=result
            return result

    def mgf(self,omega=None,k=[]):
        '''
        Returns the Green's function in the mixed representation.
        Parameters:
            omega: np.complex128, optional
                The frequency of the mixed Green's function.
            k: 1d ndarray like, optional
                The momentum of the mixed Green's function.
        Returns: 2d ndarray
            The mixed Green's function.
        '''
        cgf=self.cgf(omega)
        return cgf.dot(inv(np.identity(cgf.shape[0],dtype=np.complex128)-self.pt(k).dot(cgf)))

    def mgf_kmesh(self,omega,kmesh):
        '''
        Returns the mesh of the Green's functions in the mixed representation with respect to momentums.
        Parameters:
            omega: np.complex128
                The frequency of the mixed Green's functions.
            kmesh: (n+1)d ndarray like
                The kmesh of the mixed Green's functions.
                And n is the spatial dimension of the system.
        Returns: 3d ndarray
            The mesh of the mixed Green's functions.
        '''
        cgf=self.cgf(omega)
        return np.einsum('jk,ikl->ijl',cgf,inv(np.identity(cgf.shape[0],dtype=np.complex128)-self.pt_kmesh(kmesh).dot(cgf)))

    def gf(self,omega=None,k=[]):
        '''
        Returns the VCA Green's function.
        Parameters:
            omega: np.complex128, optional
                The frequency of the VCA Green's function.
            k: 1d ndarray like, optional
                The momentum of the VCA Green's function.
        Returns: 2d ndarray
            The VCA Green's function.
        '''
        return gf_contract(k=k,mgf=self.mgf(omega,k),seqs=self.clmap['seqs'],coords=self.clmap['coords'])/(self.preloads[0].nopt/self.preloads[1].nopt)

    def gf_kmesh(self,omega,kmesh):
        '''
        Returns the mesh of the VCA Green's functions with respect to momentums.
        Parameters:
            omega: np.complex128
                The frequency of the VCA Green's functions.
            kmesh: (n+1)d ndarray like
                The kmesh of the VCA Green's functions.
                And n is the spatial dimension of the system.
        Returns: 3d ndarray
            The mesh of the VCA Green's functions.
        '''
        mgf_kmesh=self.mgf_kmesh(omega,kmesh)
        result=np.zeros((kmesh.shape[0],)+self.preloads[1].gf.shape,dtype=np.complex128)
        for n,k in enumerate(kmesh):
            result[n,:,:]=gf_contract(k=k,mgf=mgf_kmesh[n,:,:],seqs=self.clmap['seqs'],coords=self.clmap['coords'])
        return result/(self.preloads[0].nopt/self.preloads[1].nopt)

class EB(HP.EB):
    '''
    Single particle spectrum along a path in the Brillouin zone.
    Attribues:
        emin,emax: np.float64
            The energy range of the single particle spectrum.
        ne: integer
            The number of sample points in the energy range.
        eta: np.float64
            The damping factor.
    '''

    def __init__(self,emin=-10.0,emax=10.0,ne=401,eta=0.05,**karg):
        '''
        Constructor.
        Parameters:
            emin,emax: np.float64
                The energy range of the single particle spectrum.
            ne: integer
                The number of sample points in the energy range.
            eta: np.float64
                The damping factor.
        '''
        super(EB,self).__init__(**karg)
        self.emin=emin
        self.emax=emax
        self.ne=ne
        self.eta=eta

def VCAEB(engine,app):
    '''
    This method calculates the single particle spectrum along a path in the Brillouin zone.
    '''
    engine.rundependences(app.status.name)
    engine.cache.pop('pt_kmesh',None)
    erange=np.linspace(app.emin,app.emax,app.ne)
    result=np.zeros((app.path.rank['k'],app.ne))
    for i,omega in enumerate(erange):
        result[:,i]=-(np.trace(engine.gf_kmesh(omega+engine.mu+app.eta*1j,app.path.mesh['k']),axis1=1,axis2=2)).imag/engine.preloads[1].nopt/np.pi
    if app.save_data:
        buff=np.zeros((app.path.rank['k']*app.ne,3))
        for k in xrange(buff.shape[0]):
            i,j=divmod(k,app.path.rank['k'])
            buff[k,0]=j
            buff[k,1]=erange[i]
            buff[k,2]=result[j,i]
        np.savetxt('%s/%s_%s.dat'%(engine.dout,engine.status,app.status.name),buff)
    if app.plot:
        krange=np.array(xrange(app.path.rank['k']))
        plt.title('%s_%s'%(engine.status,app.status.name))
        plt.colorbar(plt.pcolormesh(np.tensordot(krange,np.ones(app.ne),axes=0),np.tensordot(np.ones(app.path.rank['k']),erange,axes=0),result))
        if app.show:
            plt.show()
        else:
            plt.savefig('%s/%s_.png'%(engine.dout,engine.status))
        plt.close()

def VCADOS(engine,app):
    '''
    This method calculates the density of the single particle states.
    '''
    engine.rundependences(app.status.name)
    engine.cache.pop('pt_kmesh',None)
    erange=np.linspace(app.emin,app.emax,app.ne)
    result=np.zeros((app.ne,2))
    for i,omega in enumerate(erange):
        result[i,0]=omega
        result[i,1]=-np.trace(engine.mgf_kmesh(omega+engine.mu+app.eta*1j,app.BZ.mesh['k']),axis1=1,axis2=2).sum().imag/engine.preloads[0].nopt/app.BZ.rank['k']/np.pi
    engine.log<<'Sum of DOS: %s\n'%(sum(result[:,1])*(app.emax-app.emin)/app.ne)
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

def VCAFS(engine,app):
    '''
    This method calculates the single particle spectrum at the Fermi surface.
    '''
    engine.rundependences(app.status.name)
    engine.cache.pop('pt_kmesh',None)
    result=-np.trace(engine.gf_kmesh(engine.mu+app.eta*1j,app.BZ.mesh['k']),axis1=1,axis2=2).imag/engine.preloads[0].nopt/np.pi
    if app.save_data:
        np.savetxt('%s/%s_%s.dat'%(engine.dout,engine.status,app.status.name),np.append(app.BZ.mesh['k'],result.reshape((app.BZ.rank['k'],1)),axis=1))
    if app.plot:
        plt.title('%s_%s'%(engine.status,app.status.name))
        plt.axis('equal')
        N=int(round(np.sqrt(app.BZ.rank['k'])))
        plt.colorbar(plt.pcolormesh(app.BZ.mesh['k'][:,0].reshape((N,N)),app.BZ.mesh['k'][:,1].reshape(N,N),result.reshape(N,N)))
        if app.show:
            plt.show()
        else:
            plt.savefig('%s/%s_%s.png'%(engine.dout,engine.status,app.status.name))
        plt.close()

def VCABC(engine,app):
    '''
    This method calculates the Berry curvature and Chern number based on the so-called topological Hamiltonian (PRX 2, 031008 (2012))
    '''
    engine.rundependences(app.status.name)
    engine.gf(omega=engine.mu)
    app.set(H=lambda kx,ky: -inv(engine.gf(k=[kx,ky])),mu=0)
    engine.log<<'Chern number(mu): %s(%s)\n'%(app.cn,engine.mu)
    if app.save_data or app.plot:
        buff=np.zeros((app.BZ.rank['k'],3))
        buff[:,0:2]=app.BZ.mesh['k']
        buff[:,2]=app.bc
    if app.save_data:
        np.savetxt('%s/%s_%s.dat'%(engine.dout,engine.status,app.status.name),buff)
    if app.plot:
        nk=int(round(np.sqrt(app.BZ.rank['k'])))
        plt.title('%s_%s'%(engine.status,app.status.name))
        plt.axis('equal')
        plt.colorbar(plt.pcolormesh(buff[:,0].reshape((nk,nk)),buff[:,1].reshape((nk,nk)),buff[:,2].reshape((nk,nk))))
        if app.show:
            plt.show()
        else:
            plt.savefig('%s/%s_%s.png'%(engine.dout,engine.status,app.status.name))
        plt.close()

def VCATEB(engine,app):
    '''
    This method calculates the topological Hamiltonian's spectrum.
    '''
    engine.rundependences(app.status.name)
    engine.gf(omega=engine.mu)
    H=lambda kx,ky: -inv(engine.gf(k=[kx,ky]))
    result=np.zeros((app.path.rank['k'],engine.preloads[1].nopt+1))
    for i,paras in enumerate(app.path()):
        result[i,0]=i
        result[i,1:]=eigh(H(paras['k'][0],paras['k'][1]),eigvals_only=True)
    if app.save_data:
        np.savetxt('%s/%s_%s.dat'%(engine.dout,engine.status,app.status.name),result)
    if app.plot:
        plt.title('%s_%s'%(engine.status,app.status.name))
        plt.plot(result[:,0],result[:,1:])
        if app.show:
            plt.show()
        else:
            plt.savefig('%s/%s_%s.png'%(engine.dout,engine.status,app.status.name))
        plt.close()

def VCAGP(engine,app):
    '''
    This method calculates the grand potential.
    '''
    if app.status.name in engine.apps: engine.rundependences(app.status.name)
    engine.cache.pop('pt_kmesh',None)
    ncgf=engine.preloads[0].nopt
    app.gp=0
    fx=lambda omega: np.log(np.abs(det(np.eye(ncgf)-engine.pt_kmesh(app.BZ.mesh['k']).dot(engine.cgf(omega=omega*1j+engine.mu))))).sum()
    app.gp=quad(fx,0,np.float(np.inf))[0]
    app.gp=(engine.preloads[0].gse-2/engine.preloads[0].nspin*app.gp/(np.pi*app.BZ.rank['k']))/engine.clmap['seqs'].shape[1]
    app.gp=app.gp+np.trace(engine.pt_kmesh(app.BZ.mesh['k']),axis1=1,axis2=2).sum().real/app.BZ.rank['k']/engine.clmap['seqs'].shape[1]
    app.gp=app.gp-engine.mu*engine.filling*engine.preloads[1].nopt*2/engine.preloads[0].nspin
    app.gp=app.gp/len(engine.cell)
    print 'gp(%s): %s'%(', '.join(['%s: %s'%(key,value) for key,value in engine.status.data.items()]),app.gp)

class GPM(HP.App):
    '''
    Grand potential minimization.
    Attribues:
        BS: BaseSpace or dict
            When BaseSpace, it is the basespace on which the grand potential is to be computed;
            When dict, it is the initial guess of the minimum point in the basespace.
        attrs: dict, optional
            It exists only when BS is a dict.
            entry 'fout': string
                The output file that contains the results.
            entry 'method', entry 'options':
                Please refer to http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html for details.
        bsm: dict
            The minimum point in the base space.
        gpm: np.float64
            The minimum value of the grand potential.
    '''

    def __init__(self,BS,fout=None,method=None,options=None,**karg):
        '''
        Constructor.
        Parameters:
            BS: BaseSpace or dict
                When BaseSpace, it is the basespace on which the grand potential is to be computed;
                When dict, it is the initial guess of the minimum point in the basespace.
            fout: string, optional
                The output file that contains the results.
            method, options:
                Please refer to http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html for details.
            NOTE: fout, method and options will be omitted if BS is an instance of BaseSpace.
        '''
        self.BS=BS
        if isinstance(BS,dict): self.attrs={'fout':fout,'method':method,'options':options}
        self.bsm={}
        self.gpm=0.0

def VCAGPM(engine,app):
    '''
    This method minimizes the grand potential.
    '''
    def gp(values,keys):
        engine.cache.pop('pt_kmesh',None)
        engine.update(**{key:value for key,value in zip(keys,values)})
        engine.rundependences(app.status.name)
        print
        return app.dependences[2].gp
    if isinstance(app.BS,HP.BaseSpace):
        nbs=len(app.BS.mesh.keys())
        result=np.zeros((np.product(app.BS.rank.values()),nbs+1),dtype=np.float64)
        for i,paras in enumerate(app.BS('*')):
            print paras
            result[i,0:nbs]=np.array(paras.values())
            result[i,nbs]=gp(paras.values(),paras.keys())
        app.gpm=np.amin(result[:,nbs])
        index=np.argmin(result[:,nbs])
        app.bsm={key:value for key,value in zip(paras.keys(),result[index,0:nbs])}
        print 'Minimum value(%s) at point %s'%(app.gpm,app.bsm)
        if app.save_data:
            np.savetxt('%s/%s_%s.dat'%(engine.dout,engine.status.const,app.status.name),result)
        if app.plot:
            if len(app.BS.mesh.keys())==1:
                plt.title('%s_%s'%(engine.status.const,app.status.name))
                X=np.linspace(result[:,0].min(),result[:,0].max(),300)
                for i in xrange(1,result.shape[1]):
                    tck=interpolate.splrep(result[:,0],result[:,i],k=3)
                    Y=interpolate.splev(X,tck,der=0)
                    plt.plot(X,Y)
                plt.plot(result[:,0],result[:,1],'r.')
                if app.show:
                    plt.show()
                else:
                    plt.savefig('%s/%s_%s.png'%(engine.dout,engine.status.const,app.status.name))
                plt.close()
    else:
        temp=minimize(gp,app.BS.values(),args=(app.BS.keys()),method=app.attrs['method'],options=app.attrs['options'])
        app.bsm,app.gpm={key:value for key,value in zip(app.BS.keys(),temp.x)},temp.fun
        print 'Minimum value(%s) at point %s'%(app.gpm,app.bsm)
        if app.save_data:
            result=np.array([app.bsm.values()+[app.gpm]])
            if app.fout is None:
                np.savetxt('%s/%s_%s.dat'%(engine.dout,engine.status.const,app.status.name),result)
            else:
                if os.path.isfile(app.fout):
                    with open(app.fout,'a') as fout:
                        fout.write(' '.join(['%.18e'%data for data in result[0,:]]))
                        fout.write('\n')
                else:
                    np.savetxt(app.fout,result)

class CP(HP.CP):
    '''
    Chemical potential.
    Attribues:
        p: np.float64
            A tunale parameter used in the calculation.
            For details, please refer arXiv:0806.2690.
        tol: np.float64
            The tolerance of the result.
    '''

    def __init__(self,p=1.0,tol=10**-6,**karg):
        '''
        Constructor.
        Parameters:
            p: np.float64
                A tunale parameter used in the calculation.
                For details, please refer arXiv:0806.2690.
            tol: np.float64
                The tolerance of the result.
        '''
        super(CP,self).__init__(**karg)
        self.p=p
        self.tol=10**-6

def VCACP(engine,app):
    '''
    This method calculates the chemical potential, but behaves rather bad due to the singularity of the density of states.
    '''
    engine.rundependences(app.status.name)
    engine.cache.pop('pt_kmesh',None)
    nk,nmatrix=app.BZ.rank['k'],engine.preloads[0].nopt
    fx=lambda omega,mu: (np.trace(engine.mgf_kmesh(omega=mu+1j*omega,kmesh=app.BZ.mesh['k']),axis1=1,axis2=2)-nmatrix/(1j*omega-mu-app.p)).sum().real
    gx=lambda mu: quad(fx,0,np.float(np.inf),args=(mu))[0]/nk/nmatrix/np.pi-engine.filling
    app.mu=broyden2(gx,engine.mu,verbose=True,reduction_method='svd',maxiter=20,x_tol=app.tol)
    engine.mu=app.mu
    engine.log<<'mu,error: %s, %s\n'%(engine.mu,gx(engine.mu))

class FF(HP.FF):
    '''
    Filling factor.
    Attribues:
        p: np.float64
            A tunale parameter used in the calculation.
            For details, please refer arXiv:0806.2690.
    '''

    def __init__(self,p=1.0,**karg):
        '''
        Constructor.
        Parameter:
            p: np.float64, optional
                A tunale parameter used in the calculation.
                For details, please refer arXiv:0806.2690.
        '''
        super(FF,self).__init__(**karg)
        self.p=p

def VCAFF(engine,app):
    '''
    This method calculates the filling factor.
    '''
    engine.rundependences(app.status.name)
    engine.cache.pop('pt_kmesh',None)
    nk,nmatrix=app.BZ.rank['k'],engine.preloads[0].nopt
    fx=lambda omega: (np.trace(engine.mgf_kmesh(omega=engine.mu+1j*omega,kmesh=app.BZ.mesh['k']),axis1=1,axis2=2)-nmatrix/(1j*omega-engine.mu-app.p)).sum().real
    app.filling=quad(fx,0,np.float(np.inf))[0]/nk/nmatrix/np.pi
    engine.filling=app.filling
    engine.log<<'Filling factor: %s\n'%app.filling

class OP(HP.App):
    '''
    Order parameter.
    Attribues:
        terms: list of Term
            The terms representing the orders.
        BZ: BaseSpace
            The first Brillouin zone.
        p: np.float64
            A tunale parameter used in the calculation.
            For details, please refer arXiv:0806.2690.
        ops: list of number
            The values of the order parameters.
    '''

    def __init__(self,terms,BZ=None,p=1.0,**karg):
        '''
        Constructor.
        Parameters:
            term: list of Term
                The terms representing the orders.
            BZ: BaseSpace, optional
                The first Brillouin zone.
            p: float, optional
                A tunale parameter used in the calculation.
                For details, please refer arXiv:0806.2690.
        '''
        self.terms=terms
        self.BZ=BZ
        self.p=p
        self.ops=np.zeros(len(terms),dtype=np.complex128)

def VCAOP(engine,app):
    '''
    This methods calculates the order parameters.
    '''
    engine.rundependences(app.status.name)
    engine.cache.pop('pt_kmesh',None)
    nmatrix=engine.preloads[0].nopt
    ms=np.zeros((len(app.terms),nmatrix,nmatrix),dtype=np.complex128)
    for i,term in enumerate(app.terms):
        buff=deepcopy(term)
        buff.value=1
        m=np.zeros((nmatrix,nmatrix),dtype=np.complex128)
        generator=HP.Generator(bonds=engine.lattice.bonds,config=engine.config,table=engine.config.table(mask=engine.preloads[0].mask),terms=[buff])
        for opt in generator.operators.values():
            m[opt.seqs]+=opt.value
        m+=m.T.conjugate()
        ms[i,:,:]=m
    fx=lambda omega,m: (np.trace(engine.mgf_kmesh(omega=engine.mu+1j*omega,kmesh=app.BZ.mesh['k']).dot(m),axis1=1,axis2=2)-np.trace(m)/(1j*omega-engine.mu-app.p)).sum().real
    for i,m in enumerate(ms):
        app.ops[i]=quad(fx,0,np.float(np.inf),args=(m))[0]/app.BZ.rank['k']/nmatrix*2/np.pi
    for term,op in zip(app.terms,app.ops):
        engine.log<<'%s: %s\n'%(term.id,op)
