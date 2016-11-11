'''
Floquet algorithm, including:
1) classes: FLQT
2) functions: FLQTEB
'''

__all__=['FLQT','FLQTEB']

from numpy import *
from TBA import *
from scipy.linalg import expm2,eig
import matplotlib.pyplot as plt

class FLQT(TBA):
    '''
    This class deals with floquet problems. All its attributes are inherited from TBA.
    Supported methods include:
    1) FLQTEB: calculate the quasi-energy bands.
    '''
    def __init__(self,filling=0,mu=0,lattice=None,config=None,terms=None,mask=['nambu'],**karg):
        super(FLQT,self).__init__(
            filling=    filling,
            mu=         mu,
            lattice=    lattice,
            config=     config,
            terms=      terms,
            mask=      mask
            )

    def evolution(self,t=[],**karg):
        '''
        This method returns the matrix representation of the time evolution operator.
        Parameters:
            t: 1D array-like
                The time mesh.
            karg: dict, optional
                Other parameters.
        Returns:
            result: 2D ndarray
                The matrix representation of the time evolution operator.
        '''
        nmatrix=len(self.generators['h'].table)
        result=eye(nmatrix,dtype=complex128)
        nt=len(t)
        for i,time in enumerate(t):
            if i<nt-1:
                result=dot(expm2(-1j*self.matrix(t=time,**karg)*(t[i+1]-time)),result)
        return result

def FLQTEB(engine,app):
    nmatrix=len(engine.generators['h'].table)
    if app.path!=None:
        result=zeros((app.path.rank,nmatrix+1))
        key=app.path.mesh.keys()[0]
        if len(app.path.mesh[key].shape)==1:
            result[:,0]=app.path.mesh[key]
        else:
            result[:,0]=array(xrange(app.path.rank[key]))
        for i,parameter in enumerate(list(app.path.mesh[key])):
            result[i,1:]=phase(eig(engine.evolution(t=app.ts.mesh['t'],**{key:parameter}))[0])/app.ts.volume['t']
    else:
        result=zeros((2,nmatrix+1))
        result[:,0]=array(xrange(2))
        result[0,1:]=angle(eig(engine.evolution(t=app.ts.mesh['t']))[0])/app.ts.volume['t']
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
