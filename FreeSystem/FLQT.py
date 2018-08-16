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
from .TBA import *
from scipy.linalg import expm,eig
import itertools as it
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

    def evolution(self,ts=(),**karg):
        '''
        This method returns the matrix representation of the time evolution operator.

        Parameters
        ----------
        ts : 1d ndarray-like
            The time mesh.
        karg : dict, optional
            Other parameters.

        Returns
        -------
        2d ndarray
            The matrix representation of the time evolution operator.
        '''
        result=eye(self.nmatrix,dtype=complex128)
        for i in range(len(ts)-1):
            result=dot(expm(-1j*self.matrix(t=ts[i],**karg)*(ts[i+1]-ts[i])),result)
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
    if app.path is None:
        result=zeros((2,engine.nmatrix+1))
        result[:,0]=array(range(2))
        result[0,1:]=angle(eig(engine.evolution(ts=app.ts.mesh('t')))[0])/app.ts.volume('t')
        result[1,1:]=result[0,1:]
    else:
        if isinstance(app.path,str): app.path=HP.KPath(HP.KMap(engine.lattice.reciprocals,path),nk=100)
        rank,mesh=app.path.rank(0),app.path.mesh(0)
        result=zeros((rank,engine.nmatrix+1))
        result[:,0]=mesh if mesh.ndim==1 else array(range(rank))
        for i,paras in app.path('+'):
            result[i,1:]=angle(eig(engine.evolution(ts=app.ts.mesh('t'),**paras))[0])/app.ts.volume('t')
    name='%s_%s'%(engine.tostr(mask=set(it.chain(('t'),() if app.path is None else app.path.tags))),app.name)
    if app.savedata: savetxt('%s/%s.dat'%(engine.dout,name),result)
    if app.plot: app.figure('L',result,'%s/%s'%(engine.dout,name))
    if app.returndata: return result
