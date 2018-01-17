'''
============================================================
Cluster perturbation theory and variational cluster approach
============================================================

CPT and VCA, including:
    * classes: VG, FVCA, EB, GPM, CPFF, OP, DTBT
    * functions: VCAEB, VCADOS, VCAFS, VCABC, VCATEB, VCAGP, VCAGPM, VCACPFF, VCAOP, VCADTBT
'''

__all__=['VGF','VCA','EB','VCAEB','VCADOS','VCAFS','VCABC','VCATEB','VCAGP','GPM','VCAGPM','CPFF','VCACPFF','OP','VCAOP','DTBT','VCADTBT']

from gf_contract import *
from numpy.linalg import det,inv
from scipy.linalg import eigh
from scipy.integrate import quad
from scipy.optimize import broyden2
from collections import OrderedDict
from copy import deepcopy
import numpy as np
import HamiltonianPy as HP
import HamiltonianPy.FreeSystem as TBA
import HamiltonianPy.Misc as HM
import HamiltonianPy.ED as ED
import itertools as it
import matplotlib.pyplot as plt
import time
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

class VGF(ED.GF):
    '''
    VCA single-particle Green's function with baths degrees of freedom.

    Attributes
    ----------
    lindices : 1d ndarray
        The indices of the lattice degrees of freedom.
    bindices : 1d ndarray
        The indices of the bath degrees of freedom.
    '''

    def __init__(self,operators=(),generate=ED.fedspgen,compose=ED.fedspcom,**karg):
        '''
        Constructor.

        Parameters
        ----------
        operators : list of FLinear, optional
            The single-particle operators of the Green's function.
        generate : callable, optional
            The function that generates the blocks of the Green's function.
        compose : callable, optional
            The function that composes the Green's function from its blocks.
        '''
        super(VGF,self).__init__(operators=operators,generate=generate,compose=compose,**karg)
        self.lindices=np.asarray([i for i,operator in enumerate(self.operators) if 'BATH' not in operator.index.scope],dtype=np.int)
        self.bindices=np.asarray([i for i,operator in enumerate(self.operators) if 'BATH' in operator.index.scope],dtype=np.int)

    def resetopts(self,operators):
        '''
        Reset the operators of the Green's function.
        '''
        self.operators=operators
        self.lindices=np.asarray([i for i,operator in enumerate(self.operators) if 'BATH' not in operator.index.scope],dtype=np.int)
        self.bindices=np.asarray([i for i,operator in enumerate(self.operators) if 'BATH' in operator.index.scope],dtype=np.int)

    @property
    def loperators(self):
        '''
        The lattice single-particle operators.
        '''
        return [self.operators[index] for index in self.lindices]

    @property
    def boperators(self):
        '''
        The bath single-particle operators.
        '''
        return [self.operators[index] for index in self.bindices]

    @property
    def nlopt(self):
        '''
        The number of lattice single-particle operators.
        '''
        return len(self.lindices)

    @property
    def nbopt(self):
        '''
        The number of bath single-particle operators.
        '''
        return len(self.bindices)

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
    baths : list of Term
        The bath terms of the system.
    mask : () or ('nambu',)
        * (): using the nambu space and computing the anomalous Green's functions;
        * ('nambu',): not using the nambu space and not computing the anomalous Green's functions.
    dtype : np.float32, np.float64, np.complex64, np.complex128
        The data type of the matrix representation of the Hamiltonian.
    hgenerator,wgenerator,bgenerator : Generator
        The generator of the original/Weiss/bath part of the cluster Hamiltonian.
    pthgenerator,ptwgenerator,ptbgenerator : Generator
        The generator of the original/Weiss/bath part of single-particle perturbation terms.
    operators : Operators
        The 'half' of the operators of the cluster Hamiltonian.
    pthoperators,ptwoperators,ptboperators : Operators
        The 'half' of the operators of the original/Weiss/bath part of single-particle perturbation terms.
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

    def __init__(self,cgf,sectors,cell,lattice,config,terms=(),weiss=(),baths=(),mask=('nambu',),dtype=np.complex128,**karg):
        '''
        Constructor.

        Parameters
        ----------
        cgf : VGF
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
        baths : list of Term, optional
            The bath terms of the system.
        mask : () or ('nambu',), optional
            * (): using the nambu space and computing the anomalous Green's functions;
            * ('nambu',): not using the nambu space and not computing the anomalous Green's functions.
        dtype : np.float32, np.float64, np.complex64, np.complex128
            The data type of the matrix representation of the Hamiltonian.
        '''
        assert isinstance(cgf,VGF)
        cgf.resetopts(HP.fspoperators(config.table(),lattice))
        self.preload(cgf)
        self.preload(HP.GF(operators=HP.fspoperators(HP.IDFConfig(priority=config.priority,pids=cell.pids,map=config.map).table(),cell),dtype=cgf.dtype))
        self.sectors={sector.rep:sector for sector in sectors}
        self.cell=cell
        self.lattice=lattice
        self.config=config
        self.terms=terms
        self.weiss=weiss
        self.baths=baths
        self.mask=mask
        self.dtype=dtype
        self.sector=None
        self.hgenerator=HP.Generator(
            bonds=      [bond for bond in lattice.bonds if bond.isintracell() and 'BATH' not in bond.spoint.pid.scope and 'BATH' not in bond.epoint.pid.scope],
            config=     config,
            table=      config.table(mask=('nambu',)),
            terms=      deepcopy(terms),
            dtype=      dtype,
            half=       True
            )
        self.wgenerator=HP.Generator(
            bonds=      [bond for bond in lattice.bonds if 'BATH' not in bond.spoint.pid.scope and 'BATH' not in bond.epoint.pid.scope],
            config=     config,
            table=      config.table(mask=('nambu',)),
            terms=      deepcopy(weiss),
            dtype=      dtype,
            half=       True
            )
        self.bgenerator=HP.Generator(
            bonds=      [bond for bond in lattice.bonds if 'BATH' in bond.spoint.pid.scope or 'BATH' in bond.epoint.pid.scope],
            config=     config,
            table=      config.table(mask=('nambu',)),
            terms=      deepcopy(baths),
            dtype=      dtype,
            half=       True
            )
        self.pthgenerator=HP.Generator(
            bonds=      [bond for bond in lattice.bonds if not bond.isintracell() and 'BATH' not in bond.spoint.pid.scope and 'BATH' not in bond.epoint.pid.scope],
            config=     config,
            table=      config.table(mask=mask),
            terms=      [deepcopy(term) for term in terms if isinstance(term,HP.Quadratic)],
            dtype=      dtype,
            half=       True
            )
        self.ptwgenerator=HP.Generator(
            bonds=      [bond for bond in lattice.bonds if 'BATH' not in bond.spoint.pid.scope and 'BATH' not in bond.epoint.pid.scope],
            config=     config,
            table=      config.table(mask=mask),
            terms=      deepcopy(weiss),
            dtype=      dtype,
            half=       True
            )
        self.ptbgenerator=HP.Generator(
            bonds=      [bond for bond in lattice.bonds if 'BATH' in bond.spoint.pid.scope or 'BATH' in bond.epoint.pid.scope],
            config=     config,
            table=      config.table(mask=mask),
            terms=      deepcopy(baths),
            dtype=      dtype,
            half=       True
            )
        if self.map is None: self.parameters.update(OrderedDict((term.id,term.value) for term in it.chain(terms,weiss,baths)))
        self.operators=self.hgenerator.operators+self.wgenerator.operators+self.bgenerator.operators
        self.pthoperators=self.pthgenerator.operators
        self.ptwoperators=self.ptwgenerator.operators
        self.ptboperators=self.ptbgenerator.operators
        self.periodize()
        self.cache={}
        self.logging()

    def periodize(self):
        '''
        Set self.periodization.
        '''
        self.periodization={}
        cgf,gf=self.CGF,self.GF
        groups=[[] for i in xrange(gf.nopt)]
        for index,copt in enumerate(cgf.loperators):
            for i,opt in enumerate(gf.operators):
                if copt.indices[0].iid==opt.indices[0].iid and HP.issubordinate(copt.rcoord-opt.rcoord,self.cell.vectors):
                    groups[i].append((index,copt))
                    break
        self.periodization['seqs']=np.zeros((gf.nopt,cgf.nlopt/gf.nopt),dtype=np.int64)
        self.periodization['coords']=np.zeros((gf.nopt,cgf.nlopt/gf.nopt,len(gf.operators[0].rcoord)),dtype=np.float64)
        for i in xrange(gf.nopt):
            for j,(index,opt) in enumerate(sorted(groups[i],key=lambda entry: entry[0])):
                self.periodization['seqs'][i,j]=index+1
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
            self.hgenerator.set_matrix(sector,HP.foptrep,self.sectors[sector],transpose=False,dtype=self.dtype)
            self.wgenerator.set_matrix(sector,HP.foptrep,self.sectors[sector],transpose=False,dtype=self.dtype)
            self.bgenerator.set_matrix(sector,HP.foptrep,self.sectors[sector],transpose=False,dtype=self.dtype)
        self.sector=sector
        matrix=self.hgenerator.matrix(sector)+self.wgenerator.matrix(sector)+self.bgenerator.matrix(sector)
        return matrix.T+matrix.conjugate()

    def update(self,**karg):
        '''
        Update the engine.
        '''
        if len(karg)>0:
            self.CGF.virgin=True
            super(ED.ED,self).update(**karg)
            data=self.data
            self.hgenerator.update(**data)
            self.wgenerator.update(**data)
            self.bgenerator.update(**data)
            self.pthgenerator.update(**data)
            self.ptwgenerator.update(**data)
            self.ptbgenerator.update(**data)
            self.operators=self.hgenerator.operators+self.wgenerator.operators+self.bgenerator.operators
            self.pthoperators=self.pthgenerator.operators
            self.ptwoperators=self.ptwgenerator.operators
            self.ptboperators=self.ptbgenerator.operators

    @property
    def CGF(self):
        '''
        The cluster Green's function.
        '''
        return self.apps[self.preloads[0]]

    @property
    def GF(self):
        '''
        The VCA Green's function.
        '''
        return self.apps[self.preloads[1]]

    @property
    def ncopt(self):
        '''
        The number of the single particle operators of the cluster.
        '''
        return self.CGF.nopt

    @property
    def nclopt(self):
        '''
        The number of the lattice single particle operators of the cluster.
        '''
        return self.CGF.nlopt

    @property
    def ncbopt(self):
        '''
        The number of the bath single particle operators of the cluster.
        '''
        return self.CGF.nbopt

    @property
    def nopt(self):
        '''
        The number of the single particle operators of the unit cell.
        '''
        return self.GF.nopt

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
        app=self.CGF
        if omega is not None:
            app.omega=omega
            if app.virgin:
                app.virgin=False
                if app.prepare is not None: app.prepare(self,app)
            self.records[app.name]=app.run(self,app)
        return self.records[app.name]

    def pt(self,k=()):
        '''
        Returns the matrix form of the perturbations.

        Parameters
        ----------
        k : 1d ndarray like, optional
            The momentum of the perturbations.

        Returns
        -------
        2d ndarray
            The matrix form of the perturbations.
        '''
        result=np.zeros((self.ncopt,self.ncopt),dtype=np.complex128)
        for opt in self.pthoperators.itervalues():
            result[opt.seqs]+=opt.value*(1 if len(k)==0 else np.exp(-1j*np.inner(k,opt.icoord)))
        for opt in self.ptwoperators.itervalues():
            result[opt.seqs]-=opt.value
        for opt in self.ptboperators.itervalues():
            result[opt.seqs]-=opt.value
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
        ginv=inv(self.cgf(omega))-self.pt(k)
        if self.ncbopt==0:
            return inv(ginv)
        else:
            return inv(ginv[self.CGF.lindices,:][:,self.CGF.lindices])

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
        ginv=inv(self.cgf(omega))-self.pt_kmesh(kmesh)
        if self.ncbopt==0:
            return inv(ginv)
        else:
            return inv(ginv[:,self.CGF.lindices,:][:,:,self.CGF.lindices])

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
        return _gf_contract_(k,self.mgf(omega,k),self.periodization['seqs'],self.periodization['coords'])/(self.nclopt/self.nopt)

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
        return result/(self.nclopt/self.nopt)

    def totba(self,weisson=False):
        '''
        Convert the free part of the system to tba.

        Parameters
        ----------
        weisson : logical, optional
            True for including the weiss terms in the converted tba and False for not.

        Returns
        -------
        TBA
            The converted tba.
        '''
        return TBA.TBA(
            dlog=       self.log.dir,
            din=        self.din,
            dout=       self.dout,
            name=       self.name,
            parameters= self.parameters,
            map=        self.map,
            lattice=    self.lattice.sublattice(self.lattice.name,[pid for pid in self.lattice.pids if 'BATH' not in pid.scope]),
            config=     self.config,
            terms=      [term for term in it.chain(self.terms,self.weiss if weisson else ()) if isinstance(term,HP.Quadratic)],
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
        result[i,1]=-np.trace(engine.mgf_kmesh(omega+app.mu+app.eta*1j,kmesh),axis1=1,axis2=2).sum().imag/engine.nclopt/nk/np.pi
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
    result[:,2]=-np.trace(engine.gf_kmesh(app.mu+app.eta*1j,kmesh),axis1=1,axis2=2).imag/engine.nclopt/np.pi
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
    engine.rundependences(app.name)
    engine.cache.pop('pt_kmesh',None)
    stime=time.time()
    cgf,pt_kmesh,nk=engine.CGF,engine.pt_kmesh(app.BZ.mesh('k')),app.BZ.rank('k')
    fx=lambda omega: np.log(np.abs(det(np.eye(engine.ncopt)-np.tensordot(pt_kmesh,engine.cgf(omega=omega*1j+app.mu),axes=(2,0))))).sum()
    rquad=quad(fx,0,np.float(np.inf),full_output=2,epsrel=1.49e-12)
    part1=-rquad[0]/np.pi
    part2=np.trace(pt_kmesh,axis1=1,axis2=2).sum().real/2
    gp=(cgf.gse+(part1+part2)/nk)/(engine.nclopt/engine.nopt)/len(engine.cell)
    etime=time.time()
    engine.log<<'gp(mu=%s,err=%.2e,neval=%s,time=%.2es): %s\n\n'%(HP.decimaltostr(app.mu),rquad[1],rquad[2]['neval'],etime-stime,gp)
    if app.returndata: return gp

class GPM(HP.App):
    '''
    Grand potential based methods.

    Attributes
    ----------
    BS : BaseSpace or dict
        * BaseSpace: the basespace on which to compute the grand potential
        * dict: the initial guess in the basespace.
    options : dict
        The extra options.
            * BS is BaseSpace: entry 'nder','minormax'
            * BS is dict: see HamiltonianPy.Misc.fstable for details.
    '''

    def __init__(self,BS,options=None,**karg):
        '''
        Constructor.

        Parameters
        ----------
        BS : BaseSpace or dict
            * BaseSpace: the basespace on which to compute the grand potential
            * dict: the initial guess in the basespace.
        options : dict, optional
            The extra options.
                * BS is BaseSpace: entry 'nder','minormax'
                * BS is dict: see HamiltonianPy.Misc.fstable for details.
        '''
        assert isinstance(BS,HP.BaseSpace) or isinstance(BS,dict)
        self.BS=BS
        self.options={} if options is None else options

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
        mode,nbs,nder,minormax='+',len(app.BS.tags),app.options.get('nder',0),app.options.get('minormax','min')
        result=np.zeros((app.BS.rank(0),nbs+nder+1))
        for i,paras in enumerate(app.BS(mode)):
            result[i,0:nbs]=np.array(paras.values())
            result[i,nbs]=gp(paras.values(),paras.keys())
        if nder>0:result[:,nbs+1:]=HM.derivatives(result[:,0],result[:,nbs],ders=range(1,nder+1)).T
        index=np.argmin(result[:,-1]) if minormax=='min' else np.argmax(result[:,-1]) if minormax=='max' else np.argmax(np.abs(result[:,-1]))
        engine.log<<'Summary:\n%s\n'%HP.Sheet(
                                cols=           app.BS.tags+['%sgp'%('' if nder==0 else '%s der of '%HP.ordinal(nder-1))],
                                contents=       np.append(result[index,0:nbs],result[index,-1]).reshape((1,-1))
                                )
        name='%s_%s'%(engine.tostr(mask=app.BS.tags),app.name)
        if app.savedata: np.savetxt('%s/%s.dat'%(engine.dout,name),result)
        if app.plot: app.figure('L',result,'%s/%s'%(engine.dout,name),interpolate=True,legend=['%sgp'%('%s der of '%HP.ordinal(k-1) if k>0 else '') for k in xrange(nder+1)])
        if app.returndata: return result
    else:
        result=HM.fstable(gp,app.BS.values(),args=(app.BS.keys(),),**app.options)
        engine.log<<'Summary:\n%s\n'%HP.Sheet(cols=app.BS.keys()+['niter','nfev','gp'],contents=np.append(result.x,[result.nit,result.nfev,result.fun]).reshape((1,-1)))
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

    def __init__(self,p=1.0,options=None,**karg):
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
        self.options={} if options is None else options

def VCACPFF(engine,app):
    '''
    This method calculates the chemical potential or filling factor.
    '''
    engine.rundependences(app.name)
    engine.cache.pop('pt_kmesh',None)
    kmesh,nk=app.BZ.mesh('k'),app.BZ.rank('k')
    fx=lambda omega,mu: (np.trace(engine.mgf_kmesh(omega=mu+1j*omega,kmesh=kmesh),axis1=1,axis2=2)-engine.nclopt/(1j*omega-app.p)).sum().real
    if app.task=='CP':
        gx=lambda mu: quad(fx,0,np.float(np.inf),args=mu)[0]/nk/engine.nclopt/np.pi-app.cf
        mu=broyden2(gx,app.options.pop('x0',0.0),**app.options)
        engine.log<<'mu(error): %s(%s)\n'%(mu,gx(mu))
        if app.returndata: return mu
    else:
        rquad=quad(fx,0,np.float(np.inf),args=app.cf,full_output=2)
        filling=rquad[0]/nk/engine.nclopt/np.pi
        engine.log<<'Filling factor(mu=%s,err=%.2e,neval=%s): %s\n'%(HP.decimaltostr(app.cf),rquad[1],rquad[2]['neval'],filling)
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
    cgf,kmesh,nk=engine.CGF,app.BZ.mesh('k'),app.BZ.rank('k')
    ops,ms={},np.zeros((len(app.terms),engine.nclopt,engine.nclopt),dtype=np.complex128)
    table=HP.Table([operator.index for operator in cgf.loperators])
    for i,term in enumerate(app.terms):
        order=deepcopy(term)
        order.value=1.0
        for opt in HP.Generator(engine.lattice.bonds,engine.config,table=table,terms=[order],half=True).operators.itervalues():
            ms[i,opt.seqs[0],opt.seqs[1]]+=opt.value
        ms[i,:,:]+=ms[i,:,:].T.conjugate()
    fx=lambda omega,m: (np.trace(np.tensordot(engine.mgf_kmesh(omega=app.mu+1j*omega,kmesh=kmesh),m,axes=(2,0)),axis1=1,axis2=2)-np.trace(m)/(1j*omega-app.p)).sum().real
    for term,m,dtype in zip(app.terms,ms,app.dtypes):
        ops[term.id]=quad(fx,0,np.float(np.inf),args=m)[0]/nk/engine.nclopt*2/np.pi
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
