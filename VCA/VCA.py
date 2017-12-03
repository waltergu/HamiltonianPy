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
from scipy.integrate import quad
from scipy.optimize import minimize,broyden2
from collections import OrderedDict
from copy import deepcopy
import numpy as np
import HamiltonianPy as HP
import HamiltonianPy.FreeSystem as TBA
import HamiltonianPy.Misc as HM
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
    sector : str
        The sector of the system.
    sectors : dict of FBasis
        The occupation number bases of the system.
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
        The generator for the perturbation coming from the Weiss terms.
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
        `VCAGPM`    implements the grand potential based methods
        `VCACPFF`   calculates the chemical potential or filling factor
        `VCAOP`     calculates the order parameter
        `VCADTBT`   calculates the distribution of fermions along a path in the Brillouin zone
        =========   ======================================================================================================================
    '''

    def __init__(self,cgf,sectors,cell,lattice,config,terms=(),weiss=(),mask=('nambu',),dtype=np.complex128,**karg):
        '''
        Constructor.

        Parameters
        ----------
        cgf : HP.ED.GF
            The cluster Green's function.
        sectors : iterable of FBasis
            The occupation number bases of the system.
        cell : Lattice
            The unit cell of the system.
        lattice : Lattice
            The lattice of the system.
        config : IDFConfig
            The configuration of the internal degrees of freedom on the lattice.
        terms : list of Term, optional
            The terms of the system.
        weiss : list of Term, optional
            The Weiss terms of the system.
        mask : [] or ['nambu']
            * []: using the nambu space and computing the anomalous Green's functions;
            * ['nambu']: not using the nambu space and not computing the anomalous Green's functions.
        dtype : np.float32, np.float64, np.complex64, np.complex128
            The data type of the matrix representation of the Hamiltonian.
        '''
        self.preload(cgf)
        self.preload(HP.GF(operators=HP.fspoperators(HP.IDFConfig(priority=config.priority,pids=cell.pids,map=config.map).table(),cell),dtype=cgf.dtype))
        self.sectors={sector.rep:sector for sector in sectors}
        self.cell=cell
        self.lattice=lattice
        self.config=config
        self.terms=terms
        self.weiss=weiss
        self.mask=mask
        self.dtype=dtype
        self.sector=None
        self.generator=HP.Generator(
            bonds=      [bond for bond in lattice.bonds if bond.isintracell()],
            config=     config,
            table=      config.table(mask=['nambu']),
            terms=      deepcopy(terms+weiss),
            dtype=      dtype,
            half=       True
            )
        self.bcgenerator=HP.Generator(
            bonds=      [bond for bond in lattice.bonds if not bond.isintracell()],
            config=     config,
            table=      config.table(mask=['nambu']),
            terms=      deepcopy(weiss),
            dtype=      dtype,
            half=       True
            )
        self.pthgenerator=HP.Generator(
            bonds=      [bond for bond in lattice.bonds if not bond.isintracell()],
            config=     config,
            table=      config.table(mask=mask),
            terms=      [deepcopy(term) for term in terms if isinstance(term,HP.Quadratic)],
            dtype=      dtype,
            half=       True
            )
        self.ptwgenerator=HP.Generator(
            bonds=      lattice.bonds,
            config=     config,
            table=      config.table(mask=mask),
            terms=      None if weiss is None else [deepcopy(term)*(-1) for term in weiss],
            dtype=      dtype,
            half=       True
            )
        if self.map is None: self.parameters.update(OrderedDict((term.id,term.value) for term in terms+weiss))
        self.operators=self.generator.operators+self.bcgenerator.operators
        self.pthoperators=self.pthgenerator.operators
        self.ptwoperators=self.ptwgenerator.operators
        self.periodize()
        self.cache={}

    def periodize(self):
        '''
        Set self.periodization.
        '''
        self.periodization={}
        cgf,gf=self.apps[self.preloads[0]],self.apps[self.preloads[1]]
        groups=[[] for i in xrange(gf.nopt)]
        for copt in cgf.operators:
            for i,opt in enumerate(gf.operators):
                if copt.indices[0].iid==opt.indices[0].iid and HP.issubordinate(copt.rcoord-opt.rcoord,self.cell.vectors):
                    groups[i].append(copt)
                    break
        self.periodization['seqs']=np.zeros((gf.nopt,cgf.nopt/gf.nopt),dtype=np.int64)
        self.periodization['coords']=np.zeros((gf.nopt,cgf.nopt/gf.nopt,len(gf.operators[0].rcoord)),dtype=np.float64)
        for i in xrange(gf.nopt):
            for j,opt in enumerate(sorted(groups[i],key=lambda operator: operator.seqs[0])):
                self.periodization['seqs'][i,j]=opt.seqs[0]+1
                self.periodization['coords'][i,j,:]=opt.rcoord

    def matrix(self,sector,reset=True):
        '''
        The matrix representation of the Hamiltonian.

        Parameters
        ----------
        sector : str
            The sector of the matrix representation of the Hamiltonian.
        reset : logical, optional
            True for resetting the matrix cache and False for not.

        Returns
        -------
        csr_matrix
            The matrix representation of the Hamiltonian.
        '''
        if reset:
            self.generator.set_matrix(sector,HP.foptrep,self.sectors[sector],transpose=False,dtype=self.dtype)
            self.bcgenerator.set_matrix(sector,HP.foptrep,self.sectors[sector],transpose=False,dtype=self.dtype)
        self.sector=sector
        matrix=self.generator.matrix(sector)+self.bcgenerator.matrix(sector)
        return matrix.T+matrix.conjugate()

    def update(self,**karg):
        '''
        Update the engine.
        '''
        if len(karg)>0:
            self.apps[self.preloads[0]].virgin=True
            super(ED.ED,self).update(**karg)
            karg=self.data(karg)
            self.generator.update(**karg)
            self.bcgenerator.update(**karg)
            self.pthgenerator.update(**karg)
            self.ptwgenerator.update(**karg)
            self.operators=self.generator.operators+self.bcgenerator.operators
            self.pthoperators=self.pthgenerator.operators
            self.ptwoperators=self.ptwgenerator.operators

    @property
    def ncopt(self):
        '''
        The number of the cluster single particle operators.
        '''
        return self.apps[self.preloads[0]].nopt

    @property
    def nopt(self):
        '''
        The number of the unit cell single particle operators.
        '''
        return self.apps[self.preloads[1]].nopt

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
        app=self.apps[self.preloads[0]]
        if omega is not None:
            app.omega=omega
            if app.virgin:
                app.virgin=False
                if app.prepare is not None: app.prepare(self,app)
            self.records[app.name]=app.run(self,app)
        return self.records[app.name]

    def pt(self,k=()):
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
            result[opt.seqs]+=opt.value*(1 if len(k)==0 else np.exp(-1j*np.inner(k,opt.icoord)))
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

    def mgf(self,omega=None,k=()):
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
        return np.tensordot(cgf,inv(np.identity(cgf.shape[0],dtype=cgf.dtype)-self.pt_kmesh(kmesh).dot(cgf)),axes=([1],[1])).transpose((1,0,2))

    def gf(self,omega=None,k=()):
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
        return _gf_contract_(k,self.mgf(omega,k),self.periodization['seqs'],self.periodization['coords'])/(self.ncopt/self.nopt)

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

    def totba(self):
        '''
        Convert the free part of the system to tba.
        '''
        return TBA.TBA(
            dlog=       self.log.dir,
            din=        self.din,
            dout=       self.dout,
            name=       self.name,
            parameters= self.parameters,
            map=        self.map,
            lattice=    self.lattice,
            config=     self.config,
            terms=      [term for term in self.terms+self.weiss if isinstance(term,HP.Quadratic)],
            mask=       self.mask,
            dtype=      self.dtype
            )

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
    engine.rundependences(app.name)
    engine.cache.pop('pt_kmesh',None)
    erange,kmesh,nk=np.linspace(app.emin,app.emax,app.ne),app.path.mesh('k'),app.path.rank('k')
    result=np.zeros((nk,app.ne,3))
    result[:,:,0]=np.tensordot(np.array(xrange(nk)),np.ones(app.ne),axes=0)
    result[:,:,1]=np.tensordot(np.ones(nk),erange,axes=0)
    for i,omega in enumerate(erange):
        result[:,i,2]=-(np.trace(engine.gf_kmesh(omega+app.mu+app.eta*1j,kmesh),axis1=1,axis2=2)).imag/engine.nopt/np.pi
    name='%s_%s'%(engine,app.name)
    if app.savedata: np.savetxt('%s/%s.dat'%(engine.dout,name),result.reshape((nk*app.ne,3)))
    if app.plot: app.figure('P',result,'%s/%s'%(engine.dout,name))
    if app.returndata: return result

def VCADOS(engine,app):
    '''
    This method calculates the density of the single particle states.
    '''
    engine.rundependences(app.name)
    engine.cache.pop('pt_kmesh',None)
    erange,kmesh,nk=np.linspace(app.emin,app.emax,app.ne),app.BZ.mesh('k'),app.BZ.rank('k')
    result=np.zeros((app.ne,2))
    for i,omega in enumerate(erange):
        result[i,0]=omega
        result[i,1]=-np.trace(engine.mgf_kmesh(omega+app.mu+app.eta*1j,kmesh),axis1=1,axis2=2).sum().imag/engine.ncopt/nk/np.pi
    engine.log<<'Sum of DOS: %s\n'%(sum(result[:,1])*(app.emax-app.emin)/app.ne)
    name='%s_%s'%(engine,app.name)
    if app.savedata: np.savetxt('%s/%s.dat'%(engine.dout,name),result)
    if app.plot: app.figure('L',result,'%s/%s'%(engine.dout,name))
    if app.returndata: return result

def VCAFS(engine,app):
    '''
    This method calculates the single particle spectrum at the Fermi surface.
    '''
    engine.rundependences(app.name)
    engine.cache.pop('pt_kmesh',None)
    kmesh,nk=app.BZ.mesh('k'),app.BZ.rank('k')
    result=np.zeros((nk,3))
    result[:,0:2]=kmesh
    result[:,2]=-np.trace(engine.gf_kmesh(app.mu+app.eta*1j,kmesh),axis1=1,axis2=2).imag/engine.ncopt/np.pi
    name='%s_%s'%(engine,app.name)
    if app.savedata: np.savetxt('%s/%s.dat'%(engine.dout,name),result)
    if app.plot: app.figure('P',result.reshape((int(np.sqrt(nk)),int(np.sqrt(nk)),3)),'%s/%s'%(engine.dout,name),axis='equal')
    if app.returndata: return result

def VCABC(engine,app):
    '''
    This method calculates the Berry curvature and Chern number based on the so-called topological Hamiltonian (PRX 2, 031008 (2012))
    '''
    engine.rundependences(app.name)
    mu,app.mu=app.mu,0.0
    engine.gf(omega=mu)
    bc,cn=app.set(H=lambda kx,ky: -inv(engine.gf(k=[kx,ky])))
    app.mu=mu
    engine.log<<'Chern number(mu): %s(%s)\n'%(cn,app.mu)
    kmesh,nk=app.BZ.mesh('k'),app.BZ.rank('k')
    result=np.zeros((nk,3))
    result[:,0:2]=kmesh
    result[:,2]=bc
    name='%s_%s'%(engine,app.name)
    if app.savedata: np.savetxt('%s/%s.dat'%(engine.dout,name),result)
    if app.plot: app.figure('P',result.reshape((int(np.sqrt(nk)),int(np.sqrt(nk)),3)),'%s/%s'%(engine.dout,name),axis='equal')
    if app.returndata: return cn if app.bcoff else cn,result

def VCATEB(engine,app):
    '''
    This method calculates the topological Hamiltonian's spectrum.
    '''
    engine.rundependences(app.name)
    engine.gf(omega=app.mu)
    H=lambda kx,ky: -inv(engine.gf(k=[kx,ky]))
    result=np.zeros((app.path.rank('k'),engine.nopt+1))
    for i,paras in enumerate(app.path('+')):
        result[i,0]=i
        result[i,1:]=eigh(H(paras['k'][0],paras['k'][1]),eigvals_only=True)
    name='%s_%s'%(engine,app.name)
    if app.savedata: np.savetxt('%s/%s.dat'%(engine.dout,name),result)
    if app.plot: app.figure('L',result,'%s/%s'%(engine.dout,name))
    if app.returndata: return result

def VCAGP(engine,app):
    '''
    This method calculates the grand potential.
    '''
    if app.name in engine.apps: engine.rundependences(app.name)
    engine.cache.pop('pt_kmesh',None)
    cgf,kmesh,nk=engine.apps[engine.preloads[0]],app.BZ.mesh('k'),app.BZ.rank('k')
    fx=lambda omega: np.log(np.abs(det(np.eye(engine.ncopt)-engine.pt_kmesh(kmesh).dot(engine.cgf(omega=omega*1j+app.mu))))).sum()
    part1=-quad(fx,0,np.float(np.inf))[0]/np.pi
    part2=np.trace(engine.pt_kmesh(kmesh),axis1=1,axis2=2).sum().real
    if np.abs(part2)>HP.RZERO: part2=part2*app.filling
    gp=(cgf.gse+(part1+part2)/nk)/(engine.ncopt/engine.nopt)/len(engine.cell)
    engine.log<<'gp(mu): %s(%s)\n\n'%(gp,app.mu)
    if app.returndata: return gp

class GPM(HP.App):
    '''
    Grand potential based methods.

    Attributes
    ----------
    BS : BaseSpace or dict
        * BaseSpace: the basespace on which to compute the grand potential;
        * dict: the initial guess in the basespace.
    job : 'min','der'
        * 'min': minimize the grand potential
        * 'der': derivate the grand potential
    options : dict
        The extra parameters to help handle the job.
            * job=='min': passed to scipy.optimize.minimize, please refer to http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html for details.
            * job=='der': entry 'nder', the number of derivates to calculates.
    '''

    def __init__(self,BS,job='min',options={},**karg):
        '''
        Constructor.

        Parameters
        ----------
        BS : BaseSpace or dict
            * BaseSpace: the basespace on which to compute the grand potential;
            * dict: the initial guess in the basespace.
        job : 'min','der', optional
            * 'min': minimize the grand potential
            * 'der': derivate the grand potential
        options : dict, optional
            The extra parameters to help handle the job.
                * job=='min': the optional parameters passed to scipy.optimize.minimize.
                * job=='der': entry 'nder', the number of derivates to calculates.
        '''
        self.BS=BS
        self.job=job
        assert job in ('min','der')
        assert isinstance(BS,HP.BaseSpace) or isinstance(BS,dict)
        if job=='min': self.options={'method':options.get('method',None),'options':options.get('options',None)} if isinstance(BS,dict) else {'mode':'+'}
        if job=='der': self.options={'mode':'+','nder':options.get('nder',2)}

def VCAGPM(engine,app):
    '''
    This method implements the grand potential based methods.
    '''
    def gp(values,keys):
        engine.cache.pop('pt_kmesh',None)
        engine.update(**{key:value for key,value in zip(keys,values)})
        engine.rundependences(app.name)
        return engine.records[app.dependences[0]]
    if isinstance(app.BS,HP.BaseSpace):
        mode,nbs,nder=app.options.get('mode','+'),len(app.BS.tags),app.options.get('nder',0)
        if app.job=='der' or app.plot: assert mode=='+' or nbs==1
        result=np.zeros((app.BS.rank(0),nder+2)) if mode=='+' else np.zeros((np.product([app.BS.rank(i) for i in xrange(nbs)]),nbs+nder+1))
        for i,paras in enumerate(app.BS(mode)):
            result[i,0:(1 if mode=='+' else nbs)]=paras.values()[0] if mode=='+' else np.array(paras.values())
            result[i,1 if mode=='+' else nbs]=gp(paras.values(),paras.keys())
        if app.job=='der': result[:,2:]=HM.derivatives(result[:,0],result[:,1],ders=range(1,nder+1)).T
        index=np.argmin(result[:,-1]) if app.job=='min' else np.argmax(np.abs(result[:,-1]))
        engine.log<<'Summary:\n%s\n'%HP.Sheet(cols=app.BS.tags+['value'],contents=np.append(result[index,0:nbs],result[index,-1]).reshape((1,-1)))
        name='%s_%s'%(engine.tostr(mask=app.BS.tags),app.name)
        if app.savedata: np.savetxt('%s/%s.dat'%(engine.dout,name),result)
        if app.plot: app.figure('L',result,'%s/%s'%(engine.dout,name),interpolate=True,legend=['%sgp'%('%s der of '%HP.ordinal(k-1) if k>0 else '') for k in xrange(nder+1)])
        if app.returndata: return result
    else:
        assert app.job=='min'
        result=minimize(gp,app.BS.values(),args=(app.BS.keys()),**app.options)
        engine.log<<'Summary:\n%s\n'%HP.Sheet(cols=app.BS.keys()+['value'],contents=np.append(result.x,result.fun).reshape((1,-1)))
        if app.savedata: np.savetxt('%s/%s_%s.dat'%(engine.dout,engine.tostr(mask=app.BS.keys()),app.name),np.append(result.x,result.fun))
        if app.returndata: return {key:value for key,value in zip(app.BS.keys(),result.x)},result.fun

class CPFF(HP.CPFF):
    '''
    Chemical potential or filling factor.

    Attributes
    ----------
    p : np.float64
        A tunable parameter used in the calculation. Refer arXiv:0806.2690 for details.
    options : dict
        Extra options.
    '''

    def __init__(self,p=1.0,options={},**karg):
        '''
        Constructor.

        Parameters
        -----------
        p : np.float64, optional
            A tunable parameter used in the calculation.
        options : dict, optional
            Extra options.
        '''
        super(CPFF,self).__init__(**karg)
        self.p=p
        self.options=options

def VCACPFF(engine,app):
    '''
    This method calculates the chemical potential or filling factor.
    '''
    engine.rundependences(app.name)
    engine.cache.pop('pt_kmesh',None)
    kmesh,nk=app.BZ.mesh('k'),app.BZ.rank('k')
    fx=lambda omega,mu: (np.trace(engine.mgf_kmesh(omega=mu+1j*omega,kmesh=kmesh),axis1=1,axis2=2)-engine.ncopt/(1j*omega-app.p)).sum().real
    if app.task=='CP':
        gx=lambda mu: quad(fx,0,np.float(np.inf),args=mu)[0]/nk/engine.ncopt/np.pi-app.cf
        mu=broyden2(gx,app.options.pop('x0',0.0),**app.options)
        engine.log<<'mu(error): %s(%s)\n'%(mu,gx(mu))
        if app.returndata: return mu
    else:
        filling=quad(fx,0,np.float(np.inf),args=app.cf)[0]/nk/engine.ncopt/np.pi
        engine.log<<'Filling factor: %s\n'%filling
        if app.returndata: return filling

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
        A tunable parameter used in the calculation. Refer arXiv:0806.2690 for details.
    dtypes : list of np.float32/np.float64/np.complex64/np.complex128
        The data types of the order parameters.
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
            A tunable parameter used in the calculation.
        dtypes : list of np.float32/np.float64/np.complex64/np.complex128, optional
            The data types of the order parameters.
        '''
        self.terms=terms
        self.BZ=BZ
        self.mu=mu
        self.p=p
        self.dtypes=[np.float64]*len(terms) if dtypes is None else dtypes
        assert len(self.dtypes)==len(self.terms)

def VCAOP(engine,app):
    '''
    This method calculates the order parameters.
    '''
    engine.rundependences(app.name)
    engine.cache.pop('pt_kmesh',None)
    cgf,kmesh,nk=engine.apps[engine.preloads[0]],app.BZ.mesh('k'),app.BZ.rank('k')
    ops,ms={},np.zeros((len(app.terms),engine.ncopt,engine.ncopt),dtype=np.complex128)
    for i,term in enumerate(app.terms):
        order=deepcopy(term)
        order.value=1.0
        for opt in HP.Generator(engine.lattice.bonds,engine.config,table=engine.config.table(mask=engine.mask),terms=[order],half=True).operators.itervalues():
            ms[i,opt.seqs[0],opt.seqs[1]]+=opt.value
        ms[i,:,:]+=ms[i,:,:].T.conjugate()
    fx=lambda omega,m: (np.trace(engine.mgf_kmesh(omega=app.mu+1j*omega,kmesh=kmesh).dot(m),axis1=1,axis2=2)-np.trace(m)/(1j*omega-app.p)).sum().real
    for term,m,dtype in zip(app.terms,ms,app.dtypes):
        ops[term.id]=quad(fx,0,np.float(np.inf),args=m)[0]/nk/engine.ncopt*2/np.pi
        if dtype in (np.float32,np.float64): ops[term.id]=ops[term.id].real
    engine.log<<HP.Sheet(corner='Order',rows=['Value'],cols=ops.keys(),contents=np.array(ops.values()).reshape((1,-1)))<<'\n'
    if app.returndata: return ops

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
        A tunable parameter used in the calculation.
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
            A tunable parameter used in the calculation.
        '''
        self.path=path
        self.mu=mu
        self.p=p

def VCADTBT(engine,app):
    '''
    This method calculates the distribution of fermions along a path in the Brillouin zone.
    '''
    engine.rundependences(app.name)
    nk,kmesh=app.path.rank('k'),app.path.mesh('k')
    nwk=lambda omega,k: (np.trace(engine.gf(omega*1j+app.mu,k))-engine.nopt/(omega*1j-app.p)).real
    result=np.zeros((nk,2))
    result[:,0]=np.array(xrange(nk))
    for i,k in enumerate(kmesh):
        result[i,1]=quad(nwk,0,np.float(np.inf),args=k)[0]/np.pi
    name='%s_%s'%(engine,app.name)
    if app.savedata: np.savetxt('%s/%s.dat'%(engine.dout,name),result)
    if app.plot: app.figure('L',result,'%s/%s'%(engine.dout,name))
    if app.returndata: return result
