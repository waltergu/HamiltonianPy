'''
=======
Floquet
=======

Floquet algorithm, including:
    * classes: FLQT, QEB
    * functions: FLQTQEB
'''

__all__=['FLQT','QEB','FLQTQEB']

from numpy import *
from TBA import *
from scipy.linalg import expm2,eig
import matplotlib.pyplot as plt
import HamiltonianPy as HP

class FLQT(TBA):
    '''
    This class deals with floquet problems. All its attributes are inherited from TBA.

    Supported methods:
        =========     ================================
        METHODS       DESCRIPTION
        =========     ================================
        `FLQTQEB`     calculate the quasi-energy bands
        =========     ================================
    '''

    def evolution(self,t=(),**karg):
        '''
        This method returns the matrix representation of the time evolution operator.

        Parameters
        ----------
        t : 1d array-like
            The time mesh.
        karg : dict, optional
            Other parameters.

        Returns
        -------
        2d ndarray
            The matrix representation of the time evolution operator.
        '''
        result=eye(self.nmatrix,dtype=complex128)
        nt=len(t)
        for i,time in enumerate(t):
            if i<nt-1:
                result=dot(expm2(-1j*self.matrix(t=time,**karg)*(t[i+1]-time)),result)
        return result

class QEB(HP.EB):
    '''
    Floquet quasi-energy bands.

    Attributes
    ----------
    ts : BaseSpace
        The time domain of the Floquet process.
    '''

    def __init__(self,ts,**karg):
        '''
        Constructor.

        Parameters
        ----------
        ts : BaseSpace
            The time domain of the Floquet process.
        '''
        super(QEB,self).__init__(**karg)
        self.ts=ts

def FLQTQEB(engine,app):
    '''
    This method calculates the Floquet quasi-energy bands.
    '''
    if app.path is not None:
        assert len(app.path.tags)==1
        rank,mesh=app.path.rank(0),app.path.mesh(0)
        result=zeros((rank,engine.nmatrix+1))
        if mesh.ndim==1:
            result[:,0]=mesh
        else:
            result[:,0]=array(xrange(rank))
        for i,paras in app.path():
            result[i,1:]=angle(eig(engine.evolution(t=app.ts.mesh('t'),**paras))[0])/app.ts.volume('t')
    else:
        result=zeros((2,engine.nmatrix+1))
        result[:,0]=array(xrange(2))
        result[0,1:]=angle(eig(engine.evolution(t=app.ts.mesh('t')))[0])/app.ts.volume('t')
        result[1,1:]=result[0,1:]
    if app.save_data:
        savetxt('%s/%s_%s.dat'%(engine.dout,engine.status,app.status.name),result)
    if app.plot:
        plt.title('%s_%s'%(engine.status,app.status.name))
        plt.plot(result[:,0],result[:,1:])
        if app.show and app.suspend: plt.show()
        if app.show and not app.suspend: plt.pause(app.SUSPEND_TIME)
        if app.save_fig: plt.savefig('%s/%s_%s.png'%(engine.dout,engine.status,app.status.name))
        plt.close()
