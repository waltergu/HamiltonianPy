'''
-----------
KSpace pack
-----------

KSpace pack, including:
    * functions: line_bz, rectangle_gxm, rectangle_gym, rectangle_bz, square_gxm, square_gym, square_bz, hexagon_gkm, hexagon_bz
'''

__all__=['line_bz', 'rectangle_gxm', 'rectangle_gym', 'rectangle_bz', 'square_gxm', 'square_gym', 'square_bz', 'hexagon_gkm', 'hexagon_bz']

from ..Utilities import *
from ..BaseSpace import *
from ..Geometry import isintratriangle
import numpy as np
import numpy.linalg as nl

def line_bz(reciprocals=None,nk=100):
    '''
    The BZ of 1D K-space.
    '''
    if reciprocals is None:
        recips=[np.array([2*np.pi])]
    else:
        recips=reciprocals
    return KSpace(reciprocals=recips,nk=nk)

def rectangle_gxm(reciprocals=None,nk=100):
    '''
    The Gamma-X-M-Gamma path in the rectangular BZ.
    '''
    if reciprocals is not None:
        b1=reciprocals[0]
        b2=reciprocals[1]
    else:
        b1=np.array([2*np.pi,0.0])
        b2=np.array([0.0,2*np.pi])
    mesh=np.zeros((3*nk,b1.shape[0]))
    for i in xrange(nk):
        mesh[i,:]=b1/2*i/nk
        mesh[nk+i,:]=b1/2+b2/2*i/nk
        mesh[nk*2+i,:]=(b1+b2)/2*(1-1.0*i/nk)
    return BaseSpace(('k',mesh))

def rectangle_gym(reciprocals=None,nk=100):
    '''
    The Gamma-X-M-Gamma path in the rectangular BZ.
    '''
    if reciprocals is not None:
        b1=reciprocals[1]
        b2=reciprocals[0]
    else:
        b1=np.array([0.0,2*np.pi])
        b2=np.array([2*np.pi,0.0])
    mesh=np.zeros((3*nk,b1.shape[0]))
    for i in xrange(nk):
        mesh[i,:]=b1/2*i/nk
        mesh[nk+i,:]=b1/2+b2/2*i/nk
        mesh[nk*2+i,:]=(b1+b2)/2*(1-1.0*i/nk)
    return BaseSpace(('k',mesh))

def rectangle_bz(reciprocals=None,nk=100):
    '''
    The whole rectangular BZ.
    '''
    if reciprocals is None:
        recips=[np.array([2*np.pi,0.0]),np.array([0.0,2*np.pi])]
    else:
        recips=reciprocals
    return KSpace(reciprocals=recips,nk=nk)

square_gxm=rectangle_gxm
square_gym=rectangle_gym
square_bz=rectangle_bz

def hexagon_gkm(reciprocals=None,nk=100,vh='H'):
    '''
    The Gamma-K-M-Gamma path in the hexagonal BZ.
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
    mesh=np.zeros((3*nk,b1.shape[0]))
    for i in xrange(nk):
        mesh[i,:]=(b1+b2)/3*i/nk
        mesh[nk+i,:]=(b1+b2)/3+(b1-2*b2)/6*i/nk
        mesh[nk*2+i,:]=b1/2*(1-1.0*i/nk)
    return BaseSpace(('k',mesh))

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
