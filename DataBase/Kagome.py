'''
==============
Kagome lattice
==============

Kagome lattice data base, including:
    * classes: KagomeDataBase
'''

__all__=['KagomeDataBase']

from numpy import array,sqrt,asarray
from ..Basics.Geometry import *

class KagomeDataBase(object):
    '''
    Kagome lattice data base.

    Attributes
    ----------
    name : string
        The name of the cluster.
    rcoords : 2d ndarray
        The rcoords of the cluster.
    vectors : 2d ndarray
        The tanslation vectors of the cluster.
    '''

    def __init__(self,name):
        '''
        Constructor.

        Parameters
        ----------
        name : 'K3'
            The name of the cluster wanted to be constructed.
        '''
        if name not in ['K3']:
            raise ValueError('KagomeDataBase construction error: unexpected name(%s).'%name)
        self.rcoords=[]
        self.vectors=[]
        if name=='K3':
            self.rcoords.append(array([0.0,0.0]))
            self.rcoords.append(array([0.5,0.0]))
            self.rcoords.append(array([0.25,sqrt(3)/4]))
            self.vectors.append(array([1.0,0.0]))
            self.vectors.append(array([0.5,sqrt(3)/2]))
        self.rcoords=asarray(self.rcoords)
        self.vectors=asarray(self.vectors)
