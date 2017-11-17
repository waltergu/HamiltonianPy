'''
-----------
KSpace pack
-----------

KSpace pack, including:
    * functions: KMap, line_bz, rectangle_gxm, rectangle_gym, rectangle_bz, square_gxm, square_gym, square_bz, hexagon_gkm, hexagon_bz
'''

__all__=['KMap','line_bz', 'rectangle_gxm', 'rectangle_gym', 'rectangle_bz', 'square_gxm', 'square_gym', 'square_bz', 'hexagon_gkm', 'hexagon_bz']

from ..Utilities import *
from ..BaseSpace import *
from ..Geometry import isintratriangle
import numpy as np
import numpy.linalg as nl
import matplotlib.pyplot as plt

class KMap(list):
    '''
    This class converts path in the str form to the ndarray form.
    '''

    database={
        'L':{   'G' : np.array([0.0]),
                'X' : np.array([0.5]),
                'G1': np.array([0.0]),
                'G2': np.array([1.0]),
                'X1': np.array([0.5]),
                'X2': np.array([-0.5])
            },
        'S':{   'G' : np.array([0.0,0.0]),
                'X' : np.array([0.5,0.0]),
                'Y' : np.array([0.0,0.5]),
                'M' : np.array([0.5,0.5]),
                'X1': np.array([0.5,0.0]),
                'X2': np.array([-0.5,0.0]),
                'Y1': np.array([0.0,0.5]),
                'Y2': np.array([0.0,-0.5]),
                'M1': np.array([0.5,0.5]),
                'M2': np.array([-0.5,0.5]),
                'M3': np.array([-0.5,-0.5]),
                'M4': np.array([0.5,-0.5])
            },
        'H':{   'G' : np.array([0.0,0.0]),
                'K' : np.array([1.0/3.0,1.0/3.0]),
                'M' : np.array([0.5,0.0]),
                'K1': np.array([1.0/3.0,1.0/3.0]),
                'K2': np.array([2.0/3.0,-1.0/3.0]),
                'K3': np.array([1.0/3.0,-2.0/3.0]),
                'K4': np.array([-1.0/3.0,-1.0/3.0]),
                'K5': np.array([-2.0/3.0,1.0/3.0]),
                'K6': np.array([-1.0/3.0,2.0/3.0]),
                'M1': np.array([0.5,0.0]),
                'M2': np.array([0.5,-0.5]),
                'M3': np.array([0.0,-0.5]),
                'M4': np.array([-0.5,0.0]),
                'M5': np.array([-0.5,0.5]),
                'M6': np.array([0.0,0.5])
            }
        }

    def __init__(self,reciprocals,path):
        '''
        Constructor.

        Parameters
        ----------
        reciprocals : iterable of 1d ndarray
            The translation vectors of the reciprocal lattice.
        path : str
            The str-formed path.
        '''
        path=path.replace(' ', '')
        assert path[0] in KMap.database and path[1]==':'
        space,path,database,reciprocals=path[0],path[2:].split(','),KMap.database[path[0]],np.asarray(reciprocals)
        if space=='L':
            assert len(reciprocals)==1
        elif space=='S':
            assert len(reciprocals)==2
            inner=np.inner(reciprocals[0],reciprocals[1])/nl.norm(reciprocals[0])/nl.norm(reciprocals[1])
            assert np.abs(inner)<RZERO
        elif space=='H':
            assert len(reciprocals)==2
            inner=np.inner(reciprocals[0],reciprocals[1])/nl.norm(reciprocals[0])/nl.norm(reciprocals[1])
            assert np.abs(np.abs(inner)-0.5)<RZERO
            if np.abs(inner+0.5)<RZERO: reciprocals[1]=-reciprocals[1]
        for segment in path:
            segment=segment.split('-')
            assert len(segment)==2
            self.append([reciprocals.T.dot(database[segment[0]]),reciprocals.T.dot(database[segment[1]])])

    @staticmethod
    def view(key,reciprocals=None,show=True,suspend=False,save=True,name='KMap'):
        '''
        View the KMap.

        Parameters
        ----------
        key : 'L','S','H'
            The key of the database of KMap.
        reciprocals : iterable of 1d ndarray, optional
            The translation vectors of the reciprocal lattice.
        show : logical, optional
            True for showing the view. Otherwise not.
        suspend : logical, optional
            True for suspending the view. Otherwise not.
        save : logical, optional
            True for saving the view. Otherwise not.
        name : str, optional
            The title and filename of the view. Otherwise not.
        '''
        assert key in KMap.database
        if key=='L': reciprocals=np.asarray(reciprocals) or np.array([1.0])*2*np.pi
        elif key=='S': reciprocals=np.asarray(reciprocals) or np.array([[1.0,0.0],[0.0,1.0]])*2*np.pi
        elif key=='H': reciprocals=np.asarray(reciprocals) or np.array([[1.0,-1.0/np.sqrt(3.0)],[0,-2.0/np.sqrt(3.0)]])*2*np.pi
        plt.title(name)
        plt.axis('equal')
        for tag,position in KMap.database[key].iteritems():
            if '1' not in tag:
                coords=reciprocals.T.dot(position)
                assert len(coords)==2
                plt.scatter(coords[0],coords[1])
                plt.text(coords[0],coords[1],'%s(%s1)'%(tag,tag) if len(tag)==1 else tag,ha='center',va='bottom',color='green',fontsize=14)
        if show and suspend: plt.show()
        if show and not suspend: plt.pause(1)
        if save: plt.savefig('%s.png'%name)
        plt.close()

def line_bz(reciprocals=None,nk=100):
    '''
    The BZ of 1D K-space.
    '''
    return KSpace(reciprocals=reciprocals or [np.array([2*np.pi])],nk=nk)

def rectangle_gxm(reciprocals=None,nk=100):
    '''
    The Gamma-X-M-Gamma path in the rectangular BZ.
    '''
    return KPath(KMap(reciprocals or [np.array([2*np.pi,0.0]),np.array([0.0,2*np.pi])],'S:G-X,X-M,M-G'),nk=nk)

def rectangle_gym(reciprocals=None,nk=100):
    '''
    The Gamma-Y-M-Gamma path in the rectangular BZ.
    '''
    return KPath(KMap(reciprocals or [np.array([2*np.pi,0.0]),np.array([0.0,2*np.pi])],'S:G-Y,Y-M,M-G'),nk=nk)

def rectangle_bz(reciprocals=None,nk=100):
    '''
    The whole rectangular BZ.
    '''
    return KSpace(reciprocals=reciprocals or [np.array([2*np.pi,0.0]),np.array([0.0,2*np.pi])],nk=nk)

square_gxm=rectangle_gxm
square_gym=rectangle_gym
square_bz=rectangle_bz

def hexagon_gkm(reciprocals=None,nk=100,vh='H'):
    '''
    The Gamma-K-M-Gamma path in the hexagonal BZ.
    '''
    assert vh.upper() in ('H','V')
    if vh.upper()=='H':
        b1=np.array([np.sqrt(3.0)/2,0.5])*4*np.pi/np.sqrt(3.0)
        b2=np.array([np.sqrt(3.0)/2,-0.5])*4*np.pi/np.sqrt(3.0)
    else:
        b1=np.array([1.0,0.0])*4*np.pi/np.sqrt(3.0)
        b2=np.array([0.5,np.sqrt(3.0)/2])*4*np.pi/np.sqrt(3.0)
    return KPath(KMap(reciprocals or [b1,b2],'H:G-K,K-M,M-G'),nk=nk)

def hexagon_bz(reciprocals=None,nk=100,vh='H'):
    '''
    The whole hexagonal BZ.
    '''
    if reciprocals is not None:
        b1=reciprocals[0]
        b2=reciprocals[1]
        temp=np.inner(b1,b2)/nl.norm(b1)/nl.norm(b2)
        assert np.abs(np.abs(temp)-0.5)<RZERO
        if np.abs(temp+0.5)<RZERO: b2=-b2
    else:
        if vh in ('H','h'):
            b1=np.array([np.sqrt(3.0)/2,0.5])*4*np.pi/np.sqrt(3.0)
            b2=np.array([np.sqrt(3.0)/2,-0.5])*4*np.pi/np.sqrt(3.0)
        else:
            b1=np.array([1.0,0.0])*4*np.pi/np.sqrt(3.0)
            b2=np.array([0.5,np.sqrt(3.0)/2])*4*np.pi/np.sqrt(3.0)
    p0,p1,p2,p3,p4=-(b1+b2)/3,(b1+b2)/3,(b1+b2)*2/3,(b1*2-b2)/3,(b2*2-b1)/3
    mesh=np.zeros((nk**2,b1.shape[0]))
    for i in xrange(nk):
        for j in xrange(nk):
            coords=b1*i/nk+b2*j/nk+p0
            if isintratriangle(coords,p1,p2,p3,vertexes=(False,True,False),edges=(True,True,False)): coords=coords-b1
            if isintratriangle(coords,p1,p2,p4,vertexes=(False,True,False),edges=(True,True,False)): coords=coords-b2
            mesh[i*nk+j,:]=coords
    volume=np.abs(np.cross(b1,b2))
    return BaseSpace(('k',mesh,volume))
