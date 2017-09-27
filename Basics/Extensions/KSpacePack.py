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

def KMap(reciprocals,path):
    '''
    Convert path in the str form to the ndarray form.

    Parameters
    ----------
    reciprocals : iterable of 1d ndarray
        The translation vectors of the reciprocal lattice.
    path : str
        The str-formed path.

    Returns
    -------
    list of 2-list
        The ndarray-formed path.
    '''
    path=path.replace(' ', '')
    assert path[0] in ('L','S','H') and path[1]==':'
    result,space,path=[],path[0],path[2:].split(',')
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
        result.append([])
        for rep in segment:
            p=np.zeros(len(reciprocals))
            if space=='L':
                if rep=='G': pass
                elif rep=='X': p[0]=0.5
                elif rep=='G1': pass
                elif rep=='G2': p[0]=1.0
                elif rep=='X1': p[0]=0.5
                elif rep=='X2': p[0]=-0.5
                else: raise ValueError('KMap error: not supported representation(%s).'%rep)
            elif space=='S':
                if rep=='G': pass
                elif rep=='X': p[0]=0.5
                elif rep=='Y': p[1]=0.5
                elif rep=='M': p[0],p[1]=0.5,0.5
                elif rep=='X1': p[0]=0.5
                elif rep=='X2': p[0]=-0.5
                elif rep=='Y1': p[1]=0.5
                elif rep=='Y2': p[1]=-0.5
                elif rep=='M1': p[0],p[1]=0.5,0.5
                elif rep=='M2': p[0],p[1]=-0.5,0.5
                elif rep=='M3': p[0],p[1]=-0.5,-0.5
                elif rep=='M4': p[0],p[1]=0.5,-0.5
                else: raise ValueError('KMap error: not supported representation(%s).'%rep)
            elif space=='H':
                if rep=='G': pass
                elif rep=='K': p[0],p[1]=1.0/3.0,1.0/3.0
                elif rep=='M': p[0],p[1]=0.5,0.0
                elif rep=='K1': p[0],p[1]=1.0/3.0,1.0/3.0
                elif rep=='K2': p[0],p[1]=2.0/3.0,-1.0/3.0
                elif rep=='K3': p[0],p[1]=1.0/3.0,-2.0/3.0
                elif rep=='K4': p[0],p[1]=-1.0/3.0,-1.0/3.0
                elif rep=='K5': p[0],p[1]=-2.0/3.0,1.0/3.0
                elif rep=='K6': p[0],p[1]=-1.0/3.0,2.0/3.0
                elif rep=='M1': p[0],p[1]=0.5,0.0
                elif rep=='M2': p[0],p[1]=0.5,-0.5
                elif rep=='M3': p[0],p[1]=0.0,-0.5
                elif rep=='M4': p[0],p[1]=-0.5,0.0
                elif rep=='M5': p[0],p[1]=-0.5,0.5
                elif rep=='M6': p[0],p[1]=0.0,0.5
                else: raise ValueError('KMap error: not supported representation(%s).'%rep)
            result[-1].append(np.dot(np.asarray(reciprocals).T,p))
    return result

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
