'''
-----------------------------
Parameter spaces and K-spaces
-----------------------------

BaseSpace, including
    * classes: BaseSpace, FBZ
    * functions: KSpace, KPath, TSpace.
'''

__all__=['BaseSpace', 'KSpace', 'KPath', 'TSpace', 'FBZ']

from collections import OrderedDict
from Geometry import volume,isonline
from QuantumNumber import QuantumNumbers,NewQuantumNumber
import numpy as np
import numpy.linalg as nl
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

    def __init__(self,*contents):
        '''
        Constructor.

        Parameters
        ----------
        contents : list of 2/3-tuples
            * tuple[0]: string
                The tag of the parameter space.
            * tuple[1]: ndarray
                The mesh of the parameter space.
            * tuple[2]: float64, optional
                The volume of the parameter space..
        '''
        self.tags=[para[0] for para in contents]
        self.meshes=[para[1] for para in contents]
        self.volumes=[para[2] if len(para)==3 else None for para in contents]

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return '\n'.join('%s: volume=%s,\nmesh=%s'%(tag,vol,mesh) for tag,vol,mesh in zip(self.tags,self.volumes,self.meshes))

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
                yield OrderedDict((tag,value) for tag,value in zip(self.tags,values))
        elif mode=="+":
            for values in zip(*self.meshes):
                yield OrderedDict((tag,value) for tag,value in zip(self.tags,values))

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
    This function constructs an instance of BaseSpace that represents a region in the reciprocal space, e.g. the first Brillouin zone(FBZ).

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
    vol=(nl.norm if nvectors==1 else (np.cross if nvectors==2 else volume))(*reciprocals)
    mesh=[np.dot([a+(b-a)*i/(nk-1 if end else nk) for (a,b),i in zip(segments,pos)],reciprocals) for pos in it.product(*([xrange(nk)]*nvectors))]
    return BaseSpace(('k',np.asarray(mesh),np.abs(vol)))

def KPath(path,nk=100,ends=None,mode='R'):
    '''
    This functions constructs a path in the K space.

    Parameters
    ----------
    path : iterable with the elements in the form (start,stop)
        * start : 1d ndarray
            The start point in the k space.
        * stop : 1d ndarray
            The stop point in the k space.
    nk : int, optional
        The number of k points for every or the shortest segment.
    ends : iterable of logical, optional
        True for including the stop point of the corresponding segment in the path and False for not.
    mode : 'R', 'E', optional
        * When 'R', the number of k points for the shortest segment is `nk` and the numbers of others are determined by their length relative to the shortest one.
        * When 'E', the number of k points for every segment is 'nk'.

    Returns
    -------
    BaseSpace
        The wanted path.
    '''
    assert mode.upper() in ('R','E')
    lengths=[nl.norm(stop-start) for start,stop in path]
    nks=[nk]*len(path) if mode.upper()=='R' else [int(length/min(lengths)*nk) for length in lengths]
    ends=[False]*len(path) if ends is None else ends
    assert len(ends)==len(path)
    return BaseSpace(('k',np.array([start+(stop-start)/(nk-1 if end else nk)*i for (start,stop),nk,end in zip(path,nks,ends) for i in xrange(nk)])))

def TSpace(mesh):
    '''
    The time space.
    '''
    return BaseSpace(('t',mesh,mesh.max()-mesh.min()))

class FBZ(QuantumNumbers,BaseSpace):
    '''
    First Brillouin zone.

    Attributes
    ----------
    reciprocals : 2d ndarray
        The translation vectors of the reciprocal lattice.
    '''

    def __init__(self,reciprocals,nks=None):
        '''
        Constructor.

        Parameters
        ----------
        reciprocals : iterable of 1d ndarray
            The translation vectors of the reciprocal lattice.
        nks : iterable of int, optional
            The number of points along each translation vector i.e. the periods along each direction.
        '''
        nks=(nks or 100,)*len(reciprocals) if type(nks) in (int,long,type(None)) else nks
        assert len(nks)==len(reciprocals)
        qntype=NewQuantumNumber('kp',tuple('k%s'%(i+1) for i in xrange(len(nks))),nks)
        data=np.array(list(it.product(*[xrange(nk) for nk in nks])))
        counts=np.ones(np.product(nks),dtype=np.int64)
        super(FBZ,self).__init__('C',(qntype,data,counts),protocol=QuantumNumbers.COUNTS)
        self.tags=['k']
        self.volumes=[(nl.norm if len(nks)==1 else (np.cross if len(nks)==2 else volume))(*reciprocals)]
        self.reciprocals=np.asarray(reciprocals)

    @property
    def meshes(self):
        '''
        The mesh of the FBZ.
        '''
        nks=np.array(self.type.periods,dtype=np.float64)
        mesh=np.zeros((self.contents.shape[0],self.reciprocals.shape[1]),dtype=self.reciprocals.dtype)
        for i,icoord in enumerate(self.contents):
            mesh[i,:]=np.dot(self.reciprocals.T,icoord/nks)
        return [mesh]

    def kcoord(self,k):
        '''
        The coordinates of a k point.

        Parameters
        ----------
        k : iterable of integers
            The quantum-number-formed k point.

        Returns
        -------
        1d ndarray
            The corresponding coordinates of the k point.
        '''
        return np.dot(self.reciprocals.T,np.asarray(k)/np.array(self.type.periods,dtype=np.float64))

    def path(self,path,ends=None,mode='P'):
        '''
        Select a path from the FBZ.

        Parameters
        ----------
        path : iterable with the elements in the form (start,stop)
            * start : 1d ndarray
                The start point in the k space.
            * stop : 1d ndarray
                The stop point in the k space.
        ends : iterable of logical, optional
            True for including the stop point of the corresponding segment in the path and False for not.
        mode : 'P','I','B', optional
            'P' for 'point', 'I' for 'index' and 'B' for both.

        Returns
        -------
        * When ``mode=='P'`` : BaseSpace
            The selected path.
        * When ``mode=='I'`` : 1d ndarray
            The indices of the selected path.
        * When ``mode=='B'`` : 2-tuple
            The selected path and its indices.
        '''
        mode,ends=mode.upper(),[False]*len(path) if ends is None else ends
        assert mode in ('P','I','B') and len(ends)==len(path)
        maxp=max(self.type.periods)
        disps=[np.dot(self.reciprocals.T,disp) for disp in list(it.product(*([[0,-1]]*len(self.reciprocals))))]
        psegments,isegments,dsegments=[[] for i in xrange(len(path))],[[] for i in xrange(len(path))],[[] for i in xrange(len(path))]
        for pos,rcoord0 in enumerate(self.mesh('k')):
            for disp in disps:
                rcoord=rcoord0+disp
                for i,((start,stop),end) in enumerate(zip(path,ends)):
                    if isonline(rcoord,start,stop,ends=(True,end),rtol=10**-3/maxp):
                        psegments[i].append(rcoord)
                        isegments[i].append(pos)
                        dsegments[i].append(nl.norm(rcoord-start))
        points,indices=[],[]
        for i,(psegment,isegment,dsegment) in enumerate(zip(psegments,isegments,dsegments)):
            permutation=np.argsort(np.array(dsegment))
            psegment=np.array(psegment)[permutation,:]
            isegment=np.array(isegment)[permutation]
            if i>0 and isegment[0]==indices[-1][-1]:
                psegment=psegment[1:,:]
                isegment=isegment[1:]
            points.append(psegment)
            indices.append(isegment)
        if mode=='P':
            return BaseSpace(('k',np.concatenate(points)))
        if mode=='I':
            return np.concatenate(indices)
        if mode=='B':
            return BaseSpace(('k',np.concatenate(points))),np.concatenate(indices)
