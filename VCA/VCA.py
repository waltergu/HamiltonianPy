'''
============================================================
Cluster perturbation theory and variational cluster approach
============================================================

CPT and VCA, including:
    * classes: VCA, EB, GPM, CPFF, OP, DTBT
    * functions: VCAEB, VCADOS, VCAFS, VCABC, VCATEB, VCAGP, VCAGPM, VCACPFF, VCAOP, VCADTBT
'''

__all__=['VCA','EB','VCAEB','VCADOS','VCAFS','VCABC','VCATEB','VCAGP','GPM','VCAGPM','CPFF','VCACPFF','OP','VCAOP','DTBT','VCADTBT']

from gf_contract import *
from numpy.linalg import det,inv
from scipy.linalg import eigh
from scipy import interpolate
from scipy.sparse import csr_matrix
from scipy.integrate import quad
from scipy.optimize import minimize,newton,brenth,brentq,broyden1,broyden2
from collections import OrderedDict
from copy import deepcopy
import numpy as np
import HamiltonianPy as HP
import HamiltonianPy.ED as ED
import matplotlib.pyplot as plt
import os

def _gf_contract_(k,mgf,seqs,coords):
    '''
    Python wrapper for gf_contract_4 and gf_contract_8.

    Parameters
    ----------
    k : 1d ndarray
        The k point.
    mgf : 2d ndarray
        The cluster single-particle Green's functions in the mixed representation.
    seqs,coords : 2d ndarray, 3d ndarray
        Auxiliary arrays.

    Returns
    -------
    2d ndarray
        The VCA single-particle Green's functions.
    '''
    if mgf.dtype==np.complex64:
        return gf_contract_4(k=k,mgf=mgf,seqs=seqs,coords=coords)
    if mgf.dtype==np.complex128:
        return gf_contract_8(k=k,mgf=mgf,seqs=seqs,coords=coords)
    else:
        raise ValueError("_gf_contract_ error: mgf must be of type np.complex64 or np.complex128.")

class VCA(ED.FED):
    '''
    This class implements the algorithm of the variational cluster approach of an electron system.

    Attributes
    ----------
    preloads : 2 list
        * preloads[0]: HP.ED.GF
            The cluster Green's function.
        * preloads[1]: HP.GF
            The VCA Green's function.
    basis : FBasis
        The occupation number basis of the system.
    cell : Lattice
        The unit cell of the system.
    lattice : Lattice
        The cluster the system uses.
    config : IDFConfig
        The configuration of the internal degrees of freedom on the lattice.
    terms : list of Term
        The terms of the system.
        The weiss terms are not included in this list.
    weiss : list of Term
        The Weiss terms of the system.
    mask : [] or ['nambu']
        * []: using the nambu space and computing the anomalous Green's functions;
        * ['nambu']: not using the nambu space and not computing the anomalous Green's functions.
    dtype : np.float32, np.float64, np.complex64, np.complex128
        The data type of the matrix representation of the Hamiltonian.
    generator : Generator
        The generator for the cluster Hamiltonian, including the Weiss terms.
    pthgenerator : Generator
        The generator for the perturbation coming from the inter-cluster single-particle terms.
    ptwgenerator : Generator
        The generator for the perturbation cominig from the Weiss terms.
    operators : Operators
        The 'half' of the operators for the cluster Hamiltonian, including the Weiss terms.
    pthoperators : Operators
        The 'half' of the operators for the perturbation, not including Weiss terms.
    ptwoperators : Operators
        The 'half' of the operators for the perturbation of Weiss terms.
    periodization : dict
        It contains two entries, the necessary information to restore the translation symmetry broken by the explicit tiling of the original lattice:
        1) 'seqs': 2d ndarray of integers
        2) 'coords': 3d ndarray of floats
    matrix : csr_matrix
        The sparse matrix representation of the cluster Hamiltonian.
    cache : dict
        The cache during the process of calculation, usually to store some meshes.


    Supported methods:
        =========   ======================================================================================================================
        METHODS     DESCRIPTION
        =========   ======================================================================================================================
        `VCAEB`     calculates the single particle spectrum along a path in Brillouin zone
        `VCADOS`    calculates the single particle density of states
        `VCAFS`     calculates the single particle spectrum at the Fermi surface
        `VCABC`     calculates the Berry curvature and Chern number based on the so-called topological Hamiltonian (PRX 2, 031008 (2012))
        `VCATEB`    calculates the topological Hamiltonian's spectrum
        `VCAGP`     calculates the grand potential
        `VCAGPM`    minimizes the grand potential
        `VCACPFF`   calculates the chemical potential or filling factor
        `VCAOP`     calculates the order parameter
        `VCADTBT`   calculates the distribution of fermions along a path in the Brillouin zone
        =========   ======================================================================================================================
    '''

    def __init__(self,cgf,basis,cell,lattice,config,terms=[],weiss=[],mask=['nambu'],dtype=np.complex128,**karg):
        '''
        Constructor.

        Parameters
        ----------
        cgf : HP.ED.GF
            The cluster Green's function.
        basis : FBasis
            The occupation number basis of the system.
        cell : Lattice
            The unit cell of the system.
        lattice : Lattice
            The lattice of the system.
        config : IDFConfig
            The configuration of the internal degrees of freedom on the lattice.
        terms : list of Term, optional
            The terms of the system.
        weiss : lsit of Term, optional
            The Weiss terms of the system.
        mask : [] or ['nambu']
            * []: using the nambu space and computing the anomalous Green's functions;
            * ['nambu']: not using the nambu space and not computing the anomalous Green's functions.
        dtype : np.float32, np.float64, np.complex64, np.complex128
            The data type of the matrix representation of the Hamiltonian.
        '''
        cellconfig=HP.IDFConfig(priority=config.priority,pids=cell.pids,map=config.map)
        self.preloads.extend([cgf,HP.GF(operators=HP.fspoperators(cellconfig.table(),cell),dtype=cgf.dtype)])
        self.basis=basis
        self.cell=cell
        self.lattice=lattice
        self.config=config
        self.terms=terms
        self.weiss=weiss
        self.mask=mask
        self.dtype=dtype
        self.generator=HP.Generator(
            bonds=      [bond for bond in lattice.bonds if bond.isintracell()],
            config=     config,
            table=      config.table(mask=['nambu']),
            terms=      terms+weiss,
            dtype=      dtype
            )
        self.pthgenerator=HP.Generator(
            bonds=      [bond for bond in lattice.bonds if not bond.isintracell()],
            config=     config,
            table=      config.table(mask=mask),
            terms=      [term for term in terms if isinstance(term,HP.Quadratic)],
            dtype=      dtype
            )
        self.ptwgenerator=HP.Generator(
            bonds=      [bond for bond in lattice.bonds if bond.isintracell()],
            config=     config,
            table=      config.table(mask=mask),
            terms=      None if weiss is None else [term*(-1) for term in weiss],
            dtype=      dtype
            )
        self.status.update(const=self.generator.parameters['const'])
        self.status.update(alter=self.generator.parameters['alter'])
        self.operators=self.generator.operators
        self.pthoperators=self.pthgenerator.operators
        self.ptwoperators=self.ptwgenerator.operators
        self.periodize()
        self.cache={}

    def periodize(self):
        '''
        Set self.periodization.
        '''
        self.periodization={}
        cgf,gf=self.preloads
        groups=[[] for i in xrange(gf.nopt)]
        for copt in cgf.operators:
            for i,opt in enumerate(gf.operators):
                if copt.indices[0].iid==opt.indices[0].iid and HP.issubordinate(copt.rcoords[0]-opt.rcoords[0],self.cell.vectors):
                    groups[i].append(copt)
                    break
        self.periodization['seqs']=np.zeros((gf.nopt,cgf.nopt/gf.nopt),dtype=np.int64)
        self.periodization['coords']=np.zeros((gf.nopt,cgf.nopt/gf.nopt,len(gf.operators[0].rcoords[0])),dtype=np.float64)
        for i in xrange(gf.nopt):
            for j,opt in enumerate(sorted(groups[i],key=lambda operator: operator.seqs[0])):
                self.periodization['seqs'][i,j]=opt.seqs[0]+1
                self.periodization['coords'][i,j,:]=opt.rcoords[0]

    def update(self,**karg):
        '''
        Update the engine.
        '''
        self.generator.update(**karg)
        self.pthgenerator.update(**karg)
        self.ptwgenerator.update(**karg)
        self.operators=self.generator.operators
        self.pthoperators=self.pthgenerator.operators
        self.ptwoperators=self.ptwgenerator.operators
        self.status.update(alter=self.generator.parameters['alter'])

    @property
    def ncopt(self):
        '''
        The number of the cluster single particle operators.
        '''
        return self.preloads[0].nopt

    @property
    def nopt(self):
        '''
        The number of the unit cell single particle operators.
        '''
        return self.preloads[1].nopt

    def cgf(self,omega=None):
        '''
        Return the cluster Green's function.

        Parameters
        ----------
        omega : np.complex128/np.complex64, optional
            The frequency of the cluster Green's function.

        Returns
        -------
        2d ndarray
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

        Parameters
        ----------
        k : 1d ndarray like, optional
            The momentum of the inter-cluster perturbations.

        Returns
        -------
        2d ndarray
            The matrix form of the inter-cluster perturbations.
        '''
        result=np.zeros((self.ncopt,self.ncopt),dtype=np.complex128)
        for opt in self.pthoperators.itervalues():
            result[opt.seqs]+=opt.value*(1 if len(k)==0 else np.exp(-1j*np.inner(k,opt.icoords[0])))
        for opt in self.ptwoperators.itervalues():
            result[opt.seqs]+=opt.value
        return result+result.T.conjugate()

    def pt_kmesh(self,kmesh):
        '''
        Returns the mesh of the inter-cluster perturbations.

        Parameters
        ----------
        kmesh : (n+1)d ndarray like
            The kmesh of the inter-cluster perturbations.
            And n is the spatial dimension of the system.

        Returns
        -------
        3d ndarray
            The pt mesh.
        '''
        if 'pt_kmesh' in self.cache:
            return self.cache['pt_kmesh']
        else:
            result=np.zeros((kmesh.shape[0],self.ncopt,self.ncopt),dtype=np.complex128)
            for i,k in enumerate(kmesh):
                result[i,:,:]=self.pt(k)
            self.cache['pt_kmesh']=result
            return result

    def mgf(self,omega=None,k=[]):
        '''
        Returns the Green's function in the mixed representation.

        Parameters
        ----------
        omega : np.complex128/np.complex64, optional
            The frequency of the mixed Green's function.
        k : 1d ndarray like, optional
            The momentum of the mixed Green's function.

        Returns
        -------
        2d ndarray
            The mixed Green's function.
        '''
        cgf=self.cgf(omega)
        return cgf.dot(inv(np.identity(cgf.shape[0],dtype=cgf.dtype)-self.pt(k).dot(cgf)))

    def mgf_kmesh(self,omega,kmesh):
        '''
        Returns the mesh of the Green's functions in the mixed representation with respect to momentums.

        Parameters
        ----------
        omega : np.complex128/np.complex64
            The frequency of the mixed Green's functions.
        kmesh : (n+1)d ndarray like
            The kmesh of the mixed Green's functions.
            And n is the spatial dimension of the system.

        Returns
        -------
        3d ndarray
            The mesh of the mixed Green's functions.
        '''
        cgf=self.cgf(omega)
        return np.einsum('jk,ikl->ijl',cgf,inv(np.identity(cgf.shape[0],dtype=cgf.dtype)-self.pt_kmesh(kmesh).dot(cgf)))

    def gf(self,omega=None,k=[]):
        '''
        Returns the VCA Green's function.

        Parameters
        ----------
        omega : np.complex128/np.complex64, optional
            The frequency of the VCA Green's function.
        k : 1d ndarray like, optional
            The momentum of the VCA Green's function.

        Returns
        -------
        2d ndarray
            The VCA Green's function.
        '''
        app=self.preloads[1]
        app.gf[...]= _gf_contract_(k,self.mgf(omega,k),self.periodization['seqs'],self.periodization['coords'])/(self.ncopt/self.nopt)
        return app.gf

    def gf_kmesh(self,omega,kmesh):
        '''
        Returns the mesh of the VCA Green's functions with respect to momentums.

        Parameters
        ----------
        omega : np.complex128/np.complex64
            The frequency of the VCA Green's functions.
        kmesh : (n+1)d ndarray like
            The kmesh of the VCA Green's functions.
            And n is the spatial dimension of the system.

        Returns
        -------
        3d ndarray
            The mesh of the VCA Green's functions.
        '''
        mgf_kmesh=self.mgf_kmesh(omega,kmesh)
        result=np.zeros((kmesh.shape[0],self.nopt,self.nopt),dtype=np.complex128)
        for n,k in enumerate(kmesh):
            result[n,:,:]=_gf_contract_(k,mgf_kmesh[n,:,:],self.periodization['seqs'],self.periodization['coords'])
        return result/(self.ncopt/self.nopt)

class EB(HP.EB):
    '''
    Single particle spectrum along a path in the Brillouin zone.

    Attributes
    ----------
    emin,emax : np.float64
        The energy range of the single particle spectrum.
    ne : integer
        The number of sample points in the energy range.
    eta : np.float64
        The damping factor.
    '''

    def __init__(self,emin=-10.0,emax=10.0,ne=401,eta=0.05,**karg):
        '''
        Constructor.

        Parameters
        ----------
        emin,emax : np.float64
            The energy range of the single particle spectrum.
        ne : integer
            The number of sample points in the energy range.
        eta : np.float64
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
    erange,kmesh,nk=np.linspace(app.emin,app.emax,app.ne),app.path.mesh('k'),app.path.rank('k')
    result=np.zeros((nk,app.ne))
    for i,omega in enumerate(erange):
        result[:,i]=-(np.trace(engine.gf_kmesh(omega+app.mu+app.eta*1j,kmesh),axis1=1,axis2=2)).imag/engine.nopt/np.pi
    if app.save_data:
        data=np.zeros((nk*app.ne,3))
        for k in xrange(data.shape[0]):
            i,j=divmod(k,nk)
            data[k,0]=j
            data[k,1]=erange[i]
            data[k,2]=result[j,i]
        np.savetxt('%s/%s_%s.dat'%(engine.dout,engine.status,app.status.name),data)
    if app.plot:
        krange=np.array(xrange(nk))
        plt.title('%s_%s'%(engine.status,app.status.name))
        plt.colorbar(plt.pcolormesh(np.tensordot(krange,np.ones(app.ne),axes=0),np.tensordot(np.ones(nk),erange,axes=0),result))
        if app.show and app.suspend: plt.show()
        if app.show and not app.suspend: plt.pause(app.SUSPEND_TIME)
        if app.save_fig: plt.savefig('%s/%s_%s.png'%(engine.dout,engine.status,app.status.name))
        plt.close()

def VCADOS(engine,app):
    '''
    This method calculates the density of the single particle states.
    '''
    engine.rundependences(app.status.name)
    engine.cache.pop('pt_kmesh',None)
    erange,kmesh,nk=np.linspace(app.emin,app.emax,app.ne),app.BZ.mesh('k'),app.BZ.rank('k')
    result=np.zeros((app.ne,2))
    for i,omega in enumerate(erange):
        result[i,0]=omega
        result[i,1]=-np.trace(engine.mgf_kmesh(omega+app.mu+app.eta*1j,kmesh),axis1=1,axis2=2).sum().imag/engine.ncopt/nk/np.pi
    engine.log<<'Sum of DOS: %s\n'%(sum(result[:,1])*(app.emax-app.emin)/app.ne)
    if app.save_data:
        np.savetxt('%s/%s_%s.dat'%(engine.dout,engine.status,app.status.name),result)
    if app.plot:
        plt.title('%s_%s'%(engine.status,app.status.name))
        plt.plot(result[:,0],result[:,1])
        if app.show and app.suspend: plt.show()
        if app.show and not app.suspend: plt.pause(app.SUSPEND_TIME)
        if app.save_fig: plt.savefig('%s/%s_%s.png'%(engine.dout,engine.status,app.status.name))
        plt.close()

def VCAFS(engine,app):
    '''
    This method calculates the single particle spectrum at the Fermi surface.
    '''
    engine.rundependences(app.status.name)
    engine.cache.pop('pt_kmesh',None)
    kmesh,nk=app.BZ.mesh('k'),app.BZ.rank('k')
    result=-np.trace(engine.gf_kmesh(app.mu+app.eta*1j,kmesh),axis1=1,axis2=2).imag/engine.ncopt/np.pi
    if app.save_data:
        np.savetxt('%s/%s_%s.dat'%(engine.dout,engine.status,app.status.name),np.append(kmesh,result.reshape((nk,1)),axis=1))
    if app.plot:
        plt.title('%s_%s'%(engine.status,app.status.name))
        plt.axis('equal')
        N=int(round(np.sqrt(nk)))
        plt.colorbar(plt.pcolormesh(kmesh[:,0].reshape((N,N)),kmesh[:,1].reshape(N,N),result.reshape(N,N)))
        if app.show and app.suspend: plt.show()
        if app.show and not app.suspend: plt.pause(app.SUSPEND_TIME)
        if app.save_fig: plt.savefig('%s/%s_%s.png'%(engine.dout,engine.status,app.status.name))
        plt.close()

def VCABC(engine,app):
    '''
    This method calculates the Berry curvature and Chern number based on the so-called topological Hamiltonian (PRX 2, 031008 (2012))
    '''
    engine.rundependences(app.status.name)
    mu,app.mu=app.mu,0.0
    engine.gf(omega=mu)
    app.set(H=lambda kx,ky: -inv(engine.gf(k=[kx,ky])))
    app.mu=mu
    engine.log<<'Chern number(mu): %s(%s)\n'%(app.cn,app.mu)
    if app.save_data or app.plot:
        data=np.zeros((app.BZ.rank('k'),3))
        data[:,0:2]=app.BZ.mesh('k')
        data[:,2]=app.bc
    if app.save_data:
        np.savetxt('%s/%s_%s.dat'%(engine.dout,engine.status,app.status.name),data)
    if app.plot:
        nk=int(round(np.sqrt(app.BZ.rank('k'))))
        plt.title('%s_%s'%(engine.status,app.status.name))
        plt.axis('equal')
        plt.colorbar(plt.pcolormesh(data[:,0].reshape((nk,nk)),data[:,1].reshape((nk,nk)),data[:,2].reshape((nk,nk))))
        if app.show and app.suspend: plt.show()
        if app.show and not app.suspend: plt.pause(app.SUSPEND_TIME)
        if app.save_fig: plt.savefig('%s/%s_%s.png'%(engine.dout,engine.status,app.status.name))
        plt.close()

def VCATEB(engine,app):
    '''
    This method calculates the topological Hamiltonian's spectrum.
    '''
    engine.rundependences(app.status.name)
    engine.gf(omega=app.mu)
    H=lambda kx,ky: -inv(engine.gf(k=[kx,ky]))
    result=np.zeros((app.path.rank('k'),engine.nopt+1))
    for i,paras in enumerate(app.path()):
        result[i,0]=i
        result[i,1:]=eigh(H(paras['k'][0],paras['k'][1]),eigvals_only=True)
    if app.save_data:
        np.savetxt('%s/%s_%s.dat'%(engine.dout,engine.status,app.status.name),result)
    if app.plot:
        plt.title('%s_%s'%(engine.status,app.status.name))
        plt.plot(result[:,0],result[:,1:])
        if app.show and app.suspend: plt.show()
        if app.show and not app.suspend: plt.pause(app.SUSPEND_TIME)
        if app.save_fig: plt.savefig('%s/%s_%s.png'%(engine.dout,engine.status,app.status.name))
        plt.close()

def VCAGP(engine,app):
    '''
    This method calculates the grand potential.
    '''
    if app.status.name in engine.apps: engine.rundependences(app.status.name)
    engine.cache.pop('pt_kmesh',None)
    cgf,kmesh,nk=engine.preloads[0],app.BZ.mesh('k'),app.BZ.rank('k')
    fx=lambda omega: np.log(np.abs(det(np.eye(engine.ncopt)-engine.pt_kmesh(kmesh).dot(engine.cgf(omega=omega*1j+app.mu))))).sum()
    part1=-quad(fx,0,np.float(np.inf))[0]/np.pi
    part2=np.trace(engine.pt_kmesh(kmesh),axis1=1,axis2=2).sum().real
    app.gp=(cgf.gse+(part1+part2)/nk)/(engine.ncopt/engine.nopt)/len(engine.cell)
    engine.log<<'gp: %s\n\n'%app.gp

class GPM(HP.App):
    '''
    Grand potential minimization.

    Attributes
    ----------
    BS : BaseSpace or dict
        * When BaseSpace, it is the basespace on which the grand potential is to be computed;
        * When dict, it is the initial guess of the minimum point in the basespace.
    extras : dict, optional
        It exists only when BS is a dict.

        * entry 'fout': string
            The output file that contains the results.
        * entry 'method', entry 'options':
            Please refer to http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html for details.

    bsm : dict
        The minimum point in the base space.
    gpm : np.float64
        The minimum value of the grand potential.
    '''

    def __init__(self,BS,fout=None,method=None,options=None,**karg):
        '''
        Constructor.

        Parameters
        ----------
        BS : BaseSpace or dict
            * When BaseSpace, it is the basespace on which the grand potential is to be computed;
            * When dict, it is the initial guess of the minimum point in the basespace.
        fout : string, optional
            The output file that contains the results.
        method, options:
            Please refer to http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html for details.

        Notes
        -----
        `fout`, `method` and `options` will be omitted if `BS` is an instance of `BaseSpace`.
        '''
        self.BS=BS
        if isinstance(BS,dict): self.extras={'fout':fout,'method':method,'options':options}
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
        return app.dependences[2].gp
    if isinstance(app.BS,HP.BaseSpace):
        nbs=len(app.BS.tags)
        result=np.zeros((np.product([app.BS.rank(tag) for tag in app.BS.tags]),nbs+1),dtype=np.float64)
        for i,paras in enumerate(app.BS('*')):
            result[i,0:nbs]=np.array(paras.values())
            result[i,nbs]=gp(paras.values(),paras.keys())
        app.gpm=np.amin(result[:,nbs])
        index=np.argmin(result[:,nbs])
        app.bsm={key:value for key,value in zip(paras.keys(),result[index,0:nbs])}
        engine.log<<'Summary of Minimization:\n%s\n'%HP.Info.from_ordereddict(OrderedDict(app.bsm.items()+[('value',app.gpm)]))
        if app.save_data:
            np.savetxt('%s/%s_%s.dat'%(engine.dout,engine.status.const,app.status.name),result)
        if app.plot:
            if len(app.BS.tags)==1:
                plt.title('%s_%s'%(engine.status.const,app.status.name))
                X=np.linspace(result[:,0].min(),result[:,0].max(),300)
                for i in xrange(1,result.shape[1]):
                    tck=interpolate.splrep(result[:,0],result[:,i],k=3)
                    Y=interpolate.splev(X,tck,der=0)
                    plt.plot(X,Y)
                plt.plot(result[:,0],result[:,1],'r.')
                if app.show and app.suspend: plt.show()
                if app.show and not app.suspend: plt.pause(app.SUSPEND_TIME)
                if app.save_fig: plt.savefig('%s/%s_%s.png'%(engine.dout,engine.status.const,app.status.name))
                plt.close()
    else:
        temp=minimize(gp,app.BS.values(),args=(app.BS.keys()),method=app.extras['method'],options=app.extras['options'])
        app.bsm,app.gpm={key:value for key,value in zip(app.BS.keys(),temp.x)},temp.fun
        engine.log<<'Summary of Minimization:\n%s\n'%HP.Info.from_ordereddict(OrderedDict(app.bsm.items()+[('gp',app.gpm)]))
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

class CPFF(HP.CPFF):
    '''
    Chemical potential or filling factor.

    Attributes
    ----------
    p : np.float64
        A tunale parameter used in the calculation.
        For details, please refer arXiv:0806.2690.
    tol : np.float64
        The tolerance of the result.
    '''

    def __init__(self,p=1.0,tol=10**-6,**karg):
        '''
        Constructor.

        Parameters
        -----------
        p : np.float64
            A tunale parameter used in the calculation.
        tol : np.float64
            The tolerance of the result.
        '''
        super(CPFF,self).__init__(**karg)
        self.p=p
        self.tol=10**-6

def VCACPFF(engine,app):
    '''
    This method calculates the chemical potential or filling factor.
    '''
    engine.rundependences(app.status.name)
    engine.cache.pop('pt_kmesh',None)
    kmesh,nk=app.BZ.mesh('k'),app.BZ.rank('k')
    fx=lambda omega,mu: (np.trace(engine.mgf_kmesh(omega=mu+1j*omega,kmesh=kmesh),axis1=1,axis2=2)-engine.ncopt/(1j*omega-mu-app.p)).sum().real
    if app.task=='CP':
        gx=lambda mu: quad(fx,0,np.float(np.inf),args=mu)[0]/nk/engine.ncopt/np.pi-app.filling
        app.mu=broyden2(gx,app.mu,verbose=True,reduction_method='svd',maxiter=20,x_tol=app.tol)
        engine.log<<'mu,error: %s, %s\n'%(app.mu,gx(engine.mu))
    else:
        app.filling=quad(fx,0,np.float(np.inf),args=app.mu)[0]/nk/engine.ncopt/np.pi
        engine.filling=app.filling
        engine.log<<'Filling factor: %s\n'%app.filling

class OP(HP.App):
    '''
    Order parameter.

    Attributes
    ----------
    terms : list of Term
        The terms representing the orders.
    BZ : BaseSpace
        The first Brillouin zone.
    mu : np.float64
        The Fermi level.
    p : np.float64
        A tunale parameter used in the calculation.
        For details, please refer arXiv:0806.2690.
    dtypes : list of np.float32/np.float64/np.complex64,np.complex128, optional
        The data types of the order parameters.
    ops : list of number
        The values of the order parameters.
    '''

    def __init__(self,terms,BZ=None,mu=0.0,p=1.0,dtypes=None,**karg):
        '''
        Constructor.

        Parameters
        ----------
        term : list of Term
            The terms representing the orders.
        BZ : BaseSpace, optional
            The first Brillouin zone.
        mu : np.float64, optional
            The Fermi level.
        p : float, optional
            A tunale parameter used in the calculation.
        dtypes : list of np.float32/np.float64/np.complex64,np.complex128, optional
            The data types of the order parameters.
        '''
        self.terms=terms
        self.BZ=BZ
        self.mu=mu
        self.p=p
        self.dtypes=[np.float64]*len(terms) if dtypes is None else dtypes
        assert len(self.dtypes)==len(self.terms)
        self.ops=[None]*len(terms)

def VCAOP(engine,app):
    '''
    This method calculates the order parameters.
    '''
    engine.rundependences(app.status.name)
    engine.cache.pop('pt_kmesh',None)
    cgf,kmesh,nk=engine.preloads[0],app.BZ.mesh('k'),app.BZ.rank('k')
    ms=np.zeros((len(app.terms),engine.ncopt,engine.ncopt),dtype=np.complex128)
    for i,term in enumerate(app.terms):
        order=deepcopy(term)
        order.value=1
        m=np.zeros((engine.ncopt,engine.ncopt),dtype=np.complex128)
        for opt in HP.Generator(engine.lattice.bonds,engine.config,table=engine.config.table(mask=engine.mask),terms=[order]).operators.values():
            m[opt.seqs]+=opt.value
        m+=m.T.conjugate()
        ms[i,:,:]=m
    fx=lambda omega,m: (np.trace(engine.mgf_kmesh(omega=app.mu+1j*omega,kmesh=kmesh).dot(m),axis1=1,axis2=2)-np.trace(m)/(1j*omega-app.mu-app.p)).sum().real
    for i,(m,dtype) in enumerate(zip(ms,app.dtypes)):
        app.ops[i]=quad(fx,0,np.float(np.inf),args=m)[0]/nk/engine.ncopt*2/np.pi
        if dtype in (np.float32,np.float64): app.ops[i]=app.ops[i].real
    for term,op in zip(app.terms,app.ops):
        engine.log<<'%s: %s\n'%(term.id,op)

class DTBT(HP.App):
    '''
    Distribution of fermions.

    Attributes
    ----------
    path : BaseSpace
        The path in the Brillouin zone.
    mu : np.float64
        The Fermi level.
    p : np.float64
        A tunale parameter used in the calculation.
    '''

    def __init__(self,path,mu,p=1.0,**karg):
        '''
        Constructor.

        Parameters
        ----------
        path : BaseSpace
            The path in the Brillouin zone.
        mu : np.float64
            The Fermi level.
        p : np.float64, optional
            A tunale parameter used in the calculation.
        '''
        self.path=path
        self.mu=mu
        self.p=p

def VCADTBT(engine,app):
    '''
    This method calculates the ditribution of fermions along a path in the Brillouin zone.
    '''
    engine.rundependences(app.status.name)
    nk,kmesh=app.path.rank('k'),app.path.mesh('k')
    nwk=lambda omega,k: (np.trace(engine.gf(omega*1j+app.mu,k))-engine.nopt/(omega*1j-app.mu-app.p)).real
    result=np.zeros((nk,2))
    for i,k in enumerate(kmesh):
        result[i,0]=i
        result[i,1]=quad(nwk,0,np.float(np.inf),args=k)[0]/np.pi
    if app.save_data:
        np.savetxt('%s/%s_%s.dat'%(engine.dout,engine.status,app.status.name),result)
    if app.plot:
        plt.title('%s_%s'%(engine.status,app.status.name))
        plt.plot(result[:,0],result[:,1])
        plt.ylim([0.0,engine.nopt])
        if app.show and app.suspend: plt.show()
        if app.show and not app.suspend: plt.pause(app.SUSPEND_TIME)
        if app.save_fig: plt.savefig('%s/%s_%s.png'%(engine.dout,engine.status,app.status.name))
        plt.close()
