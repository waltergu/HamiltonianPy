'''
Tight Binding Approximation for fermionic systems, including:
1) classes: TBA
2) functions: TBAEB, TBADOS, TBACP, TBABC
'''

__all__=['TBA','TBAEB','TBADOS','TBACP','TBABC']

from ..Basics import *
from numpy import *
from scipy.linalg import eigh
import matplotlib.pyplot as plt 

class TBA(Engine):
    '''
    Tight-binding approximation for fermionic systems.
    Also support BdG systems (phenomenological superconducting systems at the mean-field level).
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
        mask: list of string
            A list to tell whether or not to use the nambu space.
        generators: dict of Generator
            The operator generators, which has the following entries:
            1) entry 'h': the generator for the Hamiltonian.
    Supported methods include:
        1) TBAEB: calculate the energy bands.
        2) TBADOS: calculate the density of states.
        3) TBACP: calculate the chemical potential.
        4) TBABC: calculate the Berry curvature and Chern number.
    '''
    
    def __init__(self,filling=0,mu=0,lattice=None,config=None,terms=None,mask=['nambu'],**karg):
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
            mask: list of string
                A list to tell whether or not to use the nambu space.
        '''
        self.filling=filling
        self.mu=mu
        self.lattice=lattice
        self.config=config
        self.terms=terms
        self.mask=mask
        self.generators={}
        self.generators['h']=Generator(bonds=lattice.bonds,config=config,table=config.table(mask=mask),terms=terms)
        self.status.update(const=self.generators['h'].parameters['const'],alter=self.generators['h'].parameters['alter'])

    def update(self,**karg):
        '''
        This method update the engine.
        '''
        self.generators['h'].update(**karg)
        self.status.update(alter=karg)

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
            if 'nambu' not in self.mask:
                i,j=opt.seqs
                if i<nmatrix/2 and j<nmatrix/2: result[j+nmatrix/2,i+nmatrix/2]+=-opt.value*conjugate(phase)
        result+=conjugate(result.T)
        return result

    def matrices(self,basespace=None,mode='*'):
        '''
        This method returns a generator iterating over the matrix representations of the Hamiltonian defined on the input basespace.
        Parameters:
            basespace: BaseSpace,optional
                The base space on which the Hamiltonian is defined.
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
        '''
        Set the chemical potential of the Hamiltonian.
        Parameters:
            kspace: BaseSpace, optional
                The basespace of the on which the Hamiltonian is defined.
        '''
        nelectron=int(round(self.filling*(1 if kspace is None else kspace.rank['k'])*len(self.generators['h'].table)))
        eigvals=sort((self.eigvals(kspace)))
        self.mu=(eigvals[nelectron]+eigvals[nelectron-2])/2

def TBAEB(engine,app):
    '''
    This method calculates the energy bands of the Hamiltonian.
    '''
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
        savetxt('%s/%s_EB.dat'%(engine.dout,engine.status),result)
    if app.plot:
        plt.title('%s_EB'%(engine.status))
        plt.plot(result[:,0],result[:,1:])
        if app.show:
            plt.show()
        else:
            plt.savefig('%s/%s_EB.png'%(engine.dout,engine.status))
        plt.close()

def TBADOS(engine,app):
    '''
    This method calculates the density of states of the Hamiltonian.
    '''
    result=zeros((app.ne,2))
    eigvals=engine.eigvals(app.BZ)
    emin=eigvals.min() if app.emin is None else app.emin
    emax=eigvals.max() if app.emax is None else app.emax
    for i,v in enumerate(linspace(emin,emax,num=app.ne)):
       result[i,0]=v
       result[i,1]=sum(app.eta/((v-eigvals)**2+app.eta**2))
    if app.save_data:
        savetxt('%s/%s_DOS.dat'%(engine.dout,engine.status),result)
    if app.plot:
        plt.title('%s_DOS'%(engine.status))
        plt.plot(result[:,0],result[:,1])
        if app.show:
            plt.show()
        else:
            plt.savefig('%s/%s_DOS.png'%(engine.dout,engine.status))
        plt.close()

def TBACP(engine,app):
    '''
    This method calculates the chemical potential of the Hamiltonian.
    '''
    nelectron=int(round(engine.filling*app.BZ.rank['k']*len(engine.generators['h'].table)))
    eigvals=sort((engine.eigvals(app.BZ)))
    app.mu=(eigvals[nelectron]+eigvals[nelectron-2])/2
    engine.mu=app.mu
    engine.log<<'mu: %s'%(app.mu)<<'\n'

def TBABC(engine,app):
    '''
    This method calculates the total Berry curvature and Chern number of the filled bands of the Hamiltonian.
    '''
    H=lambda kx,ky: engine.matrix(k=[kx,ky])
    app.set(H,engine.mu)
    engine.log<<'Chern number(mu): %s(%s)'%(app.cn,engine.mu)<<'\n'
    if app.save_data or app.plot:
        buff=zeros((app.BZ.rank['k'],3))
        buff[:,0:2]=app.BZ.mesh['k']
        buff[:,2]=app.bc
    if app.save_data:
        savetxt('%s/%s_BC.dat'%(engine.dout,engine.status),buff)
    if app.plot:
        nk=int(round(sqrt(app.BZ.rank['k'])))
        plt.title('%s_BC'%(engine.status))
        plt.axis('equal')
        plt.colorbar(plt.pcolormesh(buff[:,0].reshape((nk,nk)),buff[:,1].reshape((nk,nk)),buff[:,2].reshape((nk,nk))))
        if app.show:
            plt.show()
        else:
            plt.savefig('%s/%s_BC.png'%(engine.dout,engine.status))
        plt.close()

def TBAGF(engine,app):
    pass
