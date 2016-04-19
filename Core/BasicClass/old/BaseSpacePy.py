'''
BaseSpace, including KSpace and TSpace.
'''
from BasicGeometryPy import *
from numpy.linalg import inv
from collections import OrderedDict
import matplotlib.pyplot as plt
import itertools
class BaseSpace:
    '''
    This class provides a unified description of all kinds of parameter spaces.
    Attibutes:
        mesh: OrderedDict with its keys being any hashable object and values being ndarray
            The mesh of the parameter space.
            Its keys represent the name of the parameter space when its length ==1 or the tags of different parameter axis when its length >1.
            Its values contain the corresponding meshes.
        volume: OrderedDict with its keys being any hashable object and values being float
            The volume of the parameter space. 
            This attribute is not always initialized or used.
    '''
    def __init__(self,*paras):
        '''
        Constructor.
        Parameters:
            paras: list of dicts
                Every dict contains the following entries:
                    Entry 'tag': any hashable object
                        It specifies the key used in the attributes mesh and volume.
                    Entry 'mesh': ndarray, optional
                        The corresponding mesh.
                    Entry 'volume': float, optional
                        The corresponding volume.
        '''
        self.mesh=OrderedDict()
        self.volume=OrderedDict()
        for para in paras:
            self.mesh[para['tag']]=para['mesh'] if 'mesh' in para else None
            self.volume[para['tag']]=para['volume'] if 'volume' in para else None

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return str(self.mesh)

    def __call__(self,mode="*"):
        '''
        Returns a generator which iterates over the whole base space.
        Parameters:
            mode: string,optional
                A flag to indicate how to construct the generator.
                "+": direct sum
                     In this case, all the meshes must have the same rank.
                "*": direct product
        Returns:
            yield a dict in the form {key1:value1,key2:value2,...}
        '''
        keys=self.mesh.keys()
        if mode=="*":
            for values in itertools.product(*self.mesh.values()):
                yield {key:value for key,value in zip(keys,values)}
        elif mode=="+":
            for values in zip(*self.mesh.values()):
                yield {key:value for key,value in zip(keys,values)}

    @property
    def rank(self):
        '''
        This method returns a dict containing the number of points in the base space.
        '''
        return {key:self.mesh[key].shape[0] for key in self.mesh.keys()}

    def plot(self,show=True,name='BaseSpace'):
        '''
        Plot the points contained in its mesh. 
        Only two dimensional base spaces are supported.
        '''
        plt.axis('equal')
        for key,mesh in self.mesh.iteritems():
            x=mesh[:,0]
            y=mesh[:,1]
            plt.scatter(x,y)
            if show:
                plt.show()
            else:
                plt.savefig(name+'_'+key+'.png')
        plt.close()

def KSpace(reciprocals=None,nk=100,mesh=None,volume=0.0):
    '''
    This function returns a BaseSpace instance that represents the whole Broullouin zone(BZ), a path in the BZ, or just some isolated points in the BZ.
    It can be used in the following ways:
        1) KSpace(reciprocals=...,nk=...)
        2) KSpace(mesh=...,volume=...)
    Parameters:
        reciprocals: list of 1D ndarrays, optional
            The unit translation vectors of the BZ.
        nk: integer,optional
            The number of mesh points along each unit translation vector.
        mesh: ndarray, optional
            The mesh of the BZ
        volume: float, optional
            The volume of the BZ.
            When the parameter reciprocals is not None, it is omitted since the volume of the BZ will be calculated by the reciprocals.
    '''
    result=BaseSpace({'tag':'k','mesh':mesh,'volume':volume})
    if reciprocals is not None:
        nvectors=len(reciprocals)
        if nvectors==1:
            result.volume['k']=norm(reciprocals[0])
        elif nvectors==2:
            result.volume['k']=abs(cross(reciprocals[0],reciprocals[1]))
        elif nvectors==3:
            result.volume['k']=abs(volume(reciprocals[0],reciprocals[1],reciprocals[2]))
        else:
            raise ValueError("KSpace error: the number of reciprocals should not be greater than 3.")
        ndim=reciprocals[0].shape[0]
        ubi=1;ubj=1;ubk=1
        if nvectors>=1:ubi=nk
        if nvectors>=2:ubj=nk
        if nvectors>=3:ubk=nk
        result.mesh['k']=zeros((ubi*ubj*ubk,ndim))
        for i in xrange(ubi):
            for j in xrange(ubj):
                for k in xrange(ubk):
                    for l in xrange(ndim):
                        for h in xrange(nvectors):
                            if h==0: buff=1.0*i/ubi-0.5
                            if h==1: buff=1.0*j/ubj-0.5
                            if h==2: buff=1.0*k/ubk-0.5
                            result.mesh['k'][(i*ubj+j)*ubk+k,l]=result.mesh['k'][(i*ubj+j)*ubk+k,l]+reciprocals[h][l]*buff
    return result

def line_1d(reciprocals=None,nk=100):
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
    result=KSpace(nk=nk)
    if not reciprocals is None:
        b1=reciprocals[0]
        b2=reciprocals[1]
    else:
        b1=array([2*pi,0.0])
        b2=array([0.0,2*pi])
    ndim=b1.shape[0]
    result.mesh['k']=zeros((3*nk,ndim))
    for i in xrange(nk):
        result.mesh['k'][i,:]=b1/2*i/nk
        result.mesh['k'][nk+i,:]=b1/2+b2/2*i/nk
        result.mesh['k'][nk*2+i,:]=(b1+b2)/2*(1-1.0*i/nk)
    return result

def rectangle_gym(reciprocals=None,nk=100):
    '''
    The Gamma-X-M-Gamma path in the rectangular BZ.
    '''
    result=KSpace(nk=nk)
    if not reciprocals is None:
        b1=reciprocals[1]
        b2=reciprocals[0]
    else:
        b1=array([0.0,2*pi])
        b2=array([2*pi,0.0])
    ndim=b1.shape[0]
    result.mesh['k']=zeros((3*nk,ndim))
    for i in xrange(nk):
        result.mesh['k'][i,:]=b1/2*i/nk
        result.mesh['k'][nk+i,:]=b1/2+b2/2*i/nk
        result.mesh['k'][nk*2+i,:]=(b1+b2)/2*(1-1.0*i/nk)
    return result

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
    result=KSpace(nk=nk)
    if not reciprocals is None:
        b1=reciprocals[0]
        b2=reciprocals[1]
        buff=inner(b1,b2)/norm(b1)/norm(b2)
        if abs(buff+0.5)<RZERO:
            b2=-b2
        elif abs(buff-0.5)>RZERO:
            raise ValueError("Hexagon_gkm error: the reciprocals are too wired.")
    else:
        if vh in ('H','h'):
          b1=array([sqrt(3.0)/2,0.5])*4*pi/sqrt(3.0)
          b2=array([sqrt(3.0)/2,-0.5])*4*pi/sqrt(3.0)
        else:
          b1=array([1.0,0.0])*4*pi/sqrt(3.0)
          b2=array([0.5,sqrt(3.0)/2])*4*pi/sqrt(3.0)
    ndim=b1.shape[0]
    result.mesh['k']=zeros((3*nk,ndim))
    for i in xrange(nk):
        result.mesh['k'][i,:]=(b1+b2)/3*i/nk
        result.mesh['k'][nk+i,:]=(b1+b2)/3+(b1-2*b2)/6*i/nk
        result.mesh['k'][nk*2+i,:]=b1/2*(1-1.0*i/nk)
    return result

def hexagon_bz(reciprocals=None,nk=100,vh='H'):
    '''
    The whole hexagonal BZ.
    '''
    result=KSpace(nk=nk)
    if not reciprocals is None:
        b1=reciprocals[0]
        b2=reciprocals[1]
        buff=inner(b1,b2)/norm(b1)/norm(b2)
        if abs(buff+0.5)<RZERO:
            b2=-b2
        elif abs(buff-0.5)>RZERO:
            raise ValueError("Hexagon_gkm error: the reciprocals are too wired.")
    else:
        if vh in ('H','h'):
          b1=array([sqrt(3.0)/2,0.5])*4*pi/sqrt(3.0)
          b2=array([sqrt(3.0)/2,-0.5])*4*pi/sqrt(3.0)
        else:
          b1=array([1.0,0.0])*4*pi/sqrt(3.0)
          b2=array([0.5,sqrt(3.0)/2])*4*pi/sqrt(3.0)
    ndim=b1.shape[0]
    result.mesh['k']=zeros((nk**2,ndim))
    p0=-(b1+b2)/3
    p1=(b1+b2)/3
    p2=(b1+b2)*2/3
    p3=(b1*2-b2)/3
    p4=(b2*2-b1)/3
    for i in xrange(nk):
        for j in xrange(nk):
          coords=b1*(i-1)/nk+b2*(j-1)/nk+p0
          if in_triangle(coords,p1,p2,p3): coords=coords-b1
          if in_triangle(coords,p1,p2,p4): coords=coords-b2
          result.mesh['k'][i*nk+j,:]=coords
    result.volume['k']=abs(cross(b1,b2))
    return result
    
def in_triangle(p0,p1,p2,p3):
    '''
    Judge whether a point represented by p0 belongs to the interior of a triangle whose vertices are p1,p2 and p3.
    '''
    a=zeros((3,3))
    b=zeros(3)
    x=zeros(3)
    ndim=p0.shape[0]
    a[0:ndim,0]=p2-p1
    a[0:ndim,1]=p3-p1
    a[(2 if ndim==2 else 0):3,2]=cross(p2-p1,p3-p2)
    b[0:ndim]=p0-p1
    x=dot(inv(a),b)
    if x[0]>=0 and x[0]<=1 and x[1]>=0 and x[1]<=1 and x[0]+x[1]<=1:
        return True
    else:
        return False

def TSpace(mesh):
    '''
    The time space.
    '''
    return BaseSpace({'tag':'t','mesh':mesh,'volume':mesh.max()-mesh.min()})
