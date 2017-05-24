'''
-----------------------------
Parameter spaces and K-spaces
-----------------------------

BaseSpace, including
    * classes: BaseSpace
    * functions: KSpace, line_bz, rectangle_gxm, rectangle_gym, rectangle_bz, square_gxm, square_gym, square_bz, hexagon_gkm, hexagon_bz, TSpace.
'''

__all__=['BaseSpace', 'KSpace', 'line_bz', 'rectangle_gxm', 'rectangle_gym', 'rectangle_bz', 'square_gxm', 'square_gym', 'square_bz', 'hexagon_gkm', 'hexagon_bz', 'TSpace']

from numpy import *
from Constant import *
from numpy.linalg import norm,inv
from collections import OrderedDict
import matplotlib.pyplot as plt
import itertools as it

class BaseSpace(object):
    '''
    This class provides a unified description of parameter spaces.

    Attributes
    ----------
    tags : list of string
        The tags of the parameter spaces.
    meshes : list of ndarray
        The meshes of the parameter spaces.
    volumes : list of float64
        The volumes of the parameter spaces.
    '''

    def __init__(self,*paras):
        '''
        Constructor.

        Parameters
        ----------
        paras : list of 2/3-tuples
            * tuple[0]: string
                The tag of the parameter space.
            * tuple[1]: ndarray
                The mesh of the parameter space.
            * tuple[2]: float64, optional
                The volume of the parameter space..
        '''
        self.tags=[para[0] for para in paras]
        self.meshes=[para[1] for para in paras]
        self.volumes=[para[2] if len(para)==3 else None for para in paras]

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return '\n'.join('%s: volume=%s,\nmesh=%s'%(tag,volume,mesh) for tag,volume,mesh in zip(self.tags,self.volumes,self.meshes))

    def __call__(self,mode="*"):
        '''
        Returns a generator which iterates over the whole base space.

        Parameters
        ----------
        mode : string,optional

            A flag to indicate how to construct the generator.
                * "+": direct sum
                * "*": direct product

        Yields
        ------
            A dict in the form {tag1:value1,tag2:value2,...}

        Notes
        -----
        When ``mode=='+'``, all the meshes must have the same rank.
        '''
        if mode=="*":
            for values in it.product(*self.meshes):
                yield {tag:value for tag,value in zip(self.tags,values)}
        elif mode=="+":
            for values in zip(*self.meshes):
                yield {tag:value for tag,value in zip(self.tags,values)}

    def rank(self,tag):
        '''
        The rank, i.e. the number of sample points in the mesh, of the parameter space whose tag is `tag`.
        '''
        return self.meshes[self.tags.index(tag) if isinstance(tag,str) else tag].shape[0]

    def mesh(self,tag):
        '''
        The mesh of the parameter space whose tag is `tag`.
        '''
        return self.meshes[self.tags.index(tag) if isinstance(tag,str) else tag]

    def volume(self,tag):
        '''
        The volume of the parameter space whose tag is `tag`.
        '''
        return self.volumes[self.tags.index(tag) if isinstance(tag,str) else tag]

    def plot(self,show=True,suspend=False,save=True,name='BaseSpace'):
        '''
        Plot the sample points contained in its mesh.
        
        Notes
        -----
        Only two dimensional base spaces are supported.
        '''
        plt.axis('equal')
        for tag,mesh in zip(self.tags,self.meshes):
            x=mesh[:,0]
            y=mesh[:,1]
            plt.scatter(x,y)
            plt.title(name)
            if show and suspend: plt.show()
            if show and not suspend: plt.pause(1)
            if save: plt.savefig('%s_%s.png'%(name,tag))
        plt.close()

def KSpace(reciprocals,nk=100,segments=None,end=False):
    '''
    This function constructs an instance of BaseSpace that represents a region in the reciprocal space, e.g. the first Broullouin zone(FBZ).

    Parameters
    ----------
    reciprocals : list of 1d ndarray
        The translation vectors of the reciprocal lattice.
    nk : integer,optional
        The number of sample points along each translation vector.
    segments : list of 2-tuple, optional
        The relative start and stop positions along each translation vector.
    end : logical, optional
        True for including the endpoint and False for not.
    '''
    nvectors=len(reciprocals)
    segments=[(-0.5,0.5)]*nvectors if segments is None else segments
    assert len(segments)==nvectors and nvectors in (1,2,3)
    vol=(norm if nvectors==1 else (cross if nvectors==2 else volume))(*reciprocals)
    mesh=[dot([a+(b-a)*i/(nk-1 if end else nk) for (a,b),i in zip(segments,pos)],reciprocals) for pos in it.product(*([xrange(nk)]*nvectors))]
    return BaseSpace(('k',asarray(mesh),abs(vol)))

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
            if in_triangle(coords,p1,p2,p3,vertexes=(False,True,False),edges=(True,True,False)): coords=coords-b1
            if in_triangle(coords,p1,p2,p4,vertexes=(False,True,False),edges=(True,True,False)): coords=coords-b2
            mesh[i*nk+j,:]=coords
    volume=abs(cross(b1,b2))
    return BaseSpace(('k',mesh,volume))
    
def in_triangle(p0,p1,p2,p3,vertexes=(True,True,True),edges=(True,True,True)):
    '''
    Judge whether a point belongs to the interior of a triangle.
    
    Parameters
    ----------
    p0 : 1d ndarray
        The coordinates of the point.
    p1,p2,p3 : 1d ndarray
        The coordinates of the vertexes of the triangle.
    vertexes : 3-tuple of logical, optional
        Define whether the "interior" contains the vertexes of the triangle. True for YES and False for NO..
    edges : 3-tuple of logical, optional
        Define whether the "interior" contains the edges of the triangle. True for YES and False for NO.

    Returns
    -------
    logical
        True for belonging to the interior and False for not.

    Notes
    -----
        * Whether or not the boundary of the triangle belongs to its interior is defined by the parameters `vertexes` and `edges`.
        * The vertexes are in the order (p1,p2,p3) and the edges are in the order (p1p2,p2p3,p3p1).
        * The edges do not contain the vertexes.
    '''
    a,b,x,ndim=zeros((3,3)),zeros(3),zeros(3),p0.shape[0]
    a[0:ndim,0]=p2-p1
    a[0:ndim,1]=p3-p1
    a[(2 if ndim==2 else 0):3,2]=cross(p2-p1,p3-p1)
    b[0:ndim]=p0-p1
    x=dot(inv(a),b)
    assert x[2]==0
    onvertexes=[x[0]==0 and x[1]==0,x[0]==1 and x[1]==0,x[0]==0 and x[1]==1]
    onedges=[x[1]==0 and x[0]>0 and x[0]<1,x[0]==0 and x[1]>0 and x[1]<1,x[0]+x[1]==1 and x[0]>0 and x[0]<1]
    if any(onvertexes):
        return any([on and condition for on,condition in zip(onvertexes,vertexes)])
    elif any(onedges):
        return any([on and condition for on,condition in zip(onedges,edges)])
    elif x[0]>0 and x[0]<1 and x[1]>0 and x[1]<1 and x[0]+x[1]<1:
        return True
    else:
        return False

def TSpace(mesh):
    '''
    The time space.
    '''
    return BaseSpace(('t',mesh,mesh.max()-mesh.min()))
