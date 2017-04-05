'''
Triangular lattice data base, including:
1) classes: TriangleDataBase
'''

__all__=['TriangleDataBase']

from numpy import array,sqrt
from ..Basics.Geometry import *

class TriangleDataBase(object):
    '''
    Triangular lattice data base.
    Attributes:
        name: string
            The name of the cluster.
        rcoords: list of 1d ndarray
            The rcoords of the cluster.
        vectors: list of 1d ndarray
            The tanslation vectors of the cluster.
    '''

    def __init__(self,name):
        '''
        Constructor.
        Parameters:
            name: 'T1','T12'
                The name of the cluster wanted to be constructed.
        '''
        if name not in ['T1','T12']:
            raise ValueError('TriangleDataBase construction error: unexpected name(%s).'%name)
        self.rcoords=[]
        self.vectors=[]
        if name=='T1':
            self.rcoords.append(array([0.0,0.0]))
            self.vectors.append(array([1.0,0.0]))
            self.vectors.append(array([0.5,sqrt(3)/2]))
        elif name=='T12':
            self.rcoords.append(array([0.0,0.0]))
            self.rcoords.append(array([1.0,0.0]))
            self.rcoords.append(array([2.0,0.0]))
            self.rcoords.append(array([3.0,0.0]))
            self.rcoords.append(array([0.5,-sqrt(3.0)/2]))
            self.rcoords.append(array([1.5,-sqrt(3.0)/2]))
            self.rcoords.append(array([2.5,-sqrt(3.0)/2]))
            self.rcoords.append(array([1.0,-sqrt(3.0)]))
            self.rcoords.append(array([2.0,-sqrt(3.0)]))
            self.rcoords.append(array([0.5,sqrt(3.0)/2]))
            self.rcoords.append(array([1.5,sqrt(3.0)/2]))
            self.rcoords.append(array([2.5,sqrt(3.0)/2]))
            self.vectors.append(array([0.0,2*sqrt(3.0)]))
            self.vectors.append(array([3.0,sqrt(3.0)]))
