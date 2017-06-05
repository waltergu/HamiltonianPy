'''
-----------------------------
Parameter spaces and K-spaces
-----------------------------

BaseSpace, including
    * classes: BaseSpace
    * functions: KSpace, TSpace.
'''

__all__=['BaseSpace', 'KSpace', 'TSpace']

from numpy import *
from numpy.linalg import norm
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

def TSpace(mesh):
    '''
    The time space.
    '''
    return BaseSpace(('t',mesh,mesh.max()-mesh.min()))
