'''
Tight Binding Approximation for fermionic systems, including:
1) classes: TBA
2) functions: TBAEB, TBADOS, TBACP, TBACN
'''

__all__=['TBA','TBAEB','TBADOS','TBACP','TBACN']

from ..Math.BerryCurvature import *
from ..Basics import *
from numpy import *
from scipy.linalg import eigh
import matplotlib.pyplot as plt 

class TBA(Engine):
    '''
    This class provides a general algorithm to calculate physical quantities of non-interacting fermionic systems based on the tight-binding approximation.
    The BdG systems, i.e. phenomenological superconducting systems based on mean-field theory are also supported in a unified way.
    Attributes:
        filling: float
            The filling factor of the system.
        mu: float
            The chemical potential of the system.
        lattice: Lattice
            The lattice of the system.
        config: IDFConfig
            The configuration of degrees of freedom.
        terms: list of Term
            The terms of the system.
        nambu: logical
            A flag to tag whether the Nambu space is used.
        generators: dict of Generator
            The operator generators, which has the following entries:
            1) entry 'h': the generator for the Hamiltonian.
    Supported methods include:
        1) TBAEB: calculate the energy bands.
        2) TBADOS: calculate the density of states.
        3) TBACP: calculate the chemical potential.
        4) TBACN: calculate the Chern number and Berry curvature.
    '''
    
    def __init__(self,filling=0,mu=0,lattice=None,config=None,terms=None,nambu=False,**karg):
        '''
        Constructor.
        Parameters:
            filling: float
                The filling factor of the system.
            mu: float
                The chemical potential of the system.
            lattice: Lattice
                The lattice of the system.
            config: Configuration
                The configuration of degrees of freedom.
            terms: list of Term
                The terms of the system.
            nambu: logical
                A flag to tag whether the Nambu space is used.
        '''
        self.filling=filling
        self.mu=mu
        self.lattice=lattice
        self.config=config
        self.terms=terms
        self.nambu=nambu
        self.generators={}
        self.generators['h']=Generator(bonds=lattice.bonds,config=config,table=config.table(nambu=nambu),terms=terms)
        self.name.update(const=self.generators['h'].parameters['const'],alter=self.generators['h'].parameters['alter'])

    def update(self,**karg):
        '''
        This method update the engine.
        '''
        self.generators['h'].update(**karg)
        self.name._alter.update(**karg)

    def matrix(self,k=[],**karg):
        '''
        This method returns the matrix representation of the Hamiltonian.
        Parameters:
            k: 1D array-like, optional
                The coords of a point in K-space.
            karg: dict, optional
                Other parameters.
        Returns:
            result: 2D ndarray
                The matrix representation of the Hamiltonian.
        '''
        self.update(**karg)
        nmatrix=len(self.generators['h'].table)
        result=zeros((nmatrix,nmatrix),dtype=complex128)
        for opt in self.generators['h'].operators.values():
            phase=1 if len(k)==0 else exp(-1j*inner(k,opt.rcoords[0]))
            result[opt.seqs]+=opt.value*phase
            if self.nambu:
                i,j=opt.seqs
                if i<nmatrix/2 and j<nmatrix/2: result[j+nmatrix/2,i+nmatrix/2]+=-opt.value*conjugate(phase)
        result+=conjugate(result.T)
        return result

    def matrices(self,basespace=None,mode='*'):
        '''
        This method returns a generator which iterates over all the Hamiltonians living on the input basespace.
        Parameters:
            basespace: BaseSpace,optional
                The base space on which the Hamiltonians lives.
            mode: string,optional
                The mode which the generators takes to iterate over the base space.
        Returns:
            yield a 2D ndarray.
        '''
        if basespace is None:
            yield self.matrix()
        else:
            for paras in basespace(mode):
                yield self.matrix(**paras)

    def eigvals(self,basespace=None):
        '''
        This method returns all the eigenvalues of the Hamiltonian.
        Parameters:
            basespace: BaseSpace, optional
                The base space on which the Hamiltonian is defined.
        Returns:
            result: 1D ndarray
                All the eigenvalues.
        '''
        nmatrix=len(self.generators['h'].table)
        result=zeros(nmatrix*(1 if basespace==None else product(basespace.rank.values())))
        if basespace is None:
            result[...]=eigh(self.matrix(),eigvals_only=True)
        else:
            for i,paras in enumerate(basespace('*')):
                result[i*nmatrix:(i+1)*nmatrix]=eigh(self.matrix(**paras),eigvals_only=True)
        return result

    def set_mu(self,kspace=None):
        nelectron=int(round(self.filling*(1 if kspace is None else kspace.rank['k'])*len(self.generators['h'].table)))
        eigvals=sort((self.eigvals(kspace)))
        self.mu=(eigvals[nelectron]+eigvals[nelectron-2])/2

def TBAEB(engine,app):
    nmatrix=len(engine.generators['h'].table)
    if app.path!=None:
        key=app.path.mesh.keys()[0]
        result=zeros((app.path.rank[key],nmatrix+1))
        if len(app.path.mesh[key].shape)==1:
            result[:,0]=app.path.mesh[key]
        else:
            result[:,0]=array(xrange(app.path.rank[key]))
        for i,parameter in enumerate(list(app.path.mesh[key])):
            result[i,1:]=eigh(engine.matrix(**{key:parameter}),eigvals_only=True)
    else:
        result=zeros((2,nmatrix+1))
        result[:,0]=array(xrange(2))
        result[0,1:]=eigh(engine.matrix(),eigvals_only=True)
        result[1,1:]=result[0,1:]
    if app.save_data:
        savetxt(engine.dout+'/'+engine.name.full+'_EB.dat',result)
    if app.plot:
        plt.title(engine.name.full+'_EB')
        plt.plot(result[:,0],result[:,1:])
        if app.show:
            plt.show()
        else:
            plt.savefig(engine.dout+'/'+engine.name.full+'_EB.png')
        plt.close()

def TBADOS(engine,app):
    result=zeros((app.ne,2))
    eigvals=engine.eigvals(app.BZ)
    for i,v in enumerate(linspace(eigvals.min(),eigvals.max(),num=app.ne)):
       result[i,0]=v
       result[i,1]=sum(app.eta/((v-eigvals)**2+app.eta**2))
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

def TBACP(engine,app):
    nelectron=int(round(engine.filling*app.BZ.rank['k']*len(engine.generators['h'].table)))
    eigvals=sort((engine.eigvals(app.BZ)))
    app.mu=(eigvals[nelectron]+eigvals[nelectron-2])/2
    engine.mu=app.mu
    print 'mu:',app.mu

def TBACN(engine,app):
    H=lambda kx,ky: engine.matrix(k=[kx,ky])
    app.bc=zeros(app.BZ.rank['k'])
    for i,paras in enumerate(app.BZ()):
        app.bc[i]=berry_curvature(H,paras['k'][0],paras['k'][1],engine.mu,d=app.d)
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

def TBAGF(engine,app):
    pass
