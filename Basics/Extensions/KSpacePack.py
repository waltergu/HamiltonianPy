'''
-----------
KSpace pack
-----------

KSpace pack, including:
    * functions: line_bz, rectangle_gxm, rectangle_gym, rectangle_bz, square_gxm, square_gym, square_bz, hexagon_gkm, hexagon_bz
'''

__all__=['line_bz', 'rectangle_gxm', 'rectangle_gym', 'rectangle_bz', 'square_gxm', 'square_gym', 'square_bz', 'hexagon_gkm', 'hexagon_bz']

from ..Constant import *
from ..BaseSpace import *
from ..Geometry import isintriangle
from numpy import array,zeros,pi,sqrt,cross,dot
from numpy.linalg import norm

def line_bz(reciprocals=None,nk=100):
    '''
    The BZ of 1D K-space.
    '''
    if reciprocals is None:
        recips=[array([2*pi])]
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
        b1=array([2*pi,0.0])
        b2=array([0.0,2*pi])
    mesh=zeros((3*nk,b1.shape[0]))
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
        b1=array([0.0,2*pi])
        b2=array([2*pi,0.0])
    mesh=zeros((3*nk,b1.shape[0]))
    for i in xrange(nk):
        result.mesh[i,:]=b1/2*i/nk
        result.mesh[nk+i,:]=b1/2+b2/2*i/nk
        result.mesh[nk*2+i,:]=(b1+b2)/2*(1-1.0*i/nk)
    return BaseSpace(('k',mesh))

def rectangle_bz(reciprocals=None,nk=100):
    '''
    The whole rectangular BZ.
    '''
    if reciprocals is None:
        recips=[]
        recips.append(array([2*pi,0.0]))
        recips.append(array([0.0,2*pi]))
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
        temp=inner(b1,b2)/norm(b1)/norm(b2)
        assert abs(abs(temp)-0.5)<RZERO
        if abs(temp+0.5)<RZERO: b2=-b2
    else:
        if vh in ('H','h'):
            b1=array([sqrt(3.0)/2,0.5])*4*pi/sqrt(3.0)
            b2=array([sqrt(3.0)/2,-0.5])*4*pi/sqrt(3.0)
        else:
            b1=array([1.0,0.0])*4*pi/sqrt(3.0)
            b2=array([0.5,sqrt(3.0)/2])*4*pi/sqrt(3.0)
    mesh=zeros((3*nk,b1.shape[0]))
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
        temp=inner(b1,b2)/norm(b1)/norm(b2)
        assert abs(abs(temp)-0.5)<RZERO
        if abs(temp+0.5)<RZERO: b2=-b2
    else:
        if vh in ('H','h'):
            b1=array([sqrt(3.0)/2,0.5])*4*pi/sqrt(3.0)
            b2=array([sqrt(3.0)/2,-0.5])*4*pi/sqrt(3.0)
        else:
            b1=array([1.0,0.0])*4*pi/sqrt(3.0)
            b2=array([0.5,sqrt(3.0)/2])*4*pi/sqrt(3.0)
    p0,p1,p2,p3,p4=-(b1+b2)/3,(b1+b2)/3,(b1+b2)*2/3,(b1*2-b2)/3,(b2*2-b1)/3
    mesh=zeros((nk**2,b1.shape[0]))
    for i in xrange(nk):
        for j in xrange(nk):
            coords=b1*i/nk+b2*j/nk+p0
            if isintriangle(coords,p1,p2,p3,vertexes=(False,True,False),edges=(True,True,False)): coords=coords-b1
            if isintriangle(coords,p1,p2,p4,vertexes=(False,True,False),edges=(True,True,False)): coords=coords-b2
            mesh[i*nk+j,:]=coords
    volume=abs(cross(b1,b2))
    return BaseSpace(('k',mesh,volume))
