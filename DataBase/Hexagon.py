'''
=================
Hexagonal lattice
=================

Hexagonal lattice data base, including:
    * classes: HexagonDataBase
'''

__all__=['HexagonDataBase']

from numpy import *
from ..Basics.Geometry import *

class HexagonDataBase(object):
    '''
    Hexagonal lattice data base.

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
        name : 'H2','H4','H6','H8O','H8P','H10'
            The name of the cluster wanted to be constructed.
        '''
        if name not in ['H2','H4','H6','H8O','H8P','H10']:
            raise ValueError('HexagonDataBase construction error: unexpected name(%s).'%name)
        self.name=name
        self.rcoords=[]
        self.vectors=[]
        if name=='H2':
            self.rcoords.append(array([0.0,0.0]))
            self.rcoords.append(array([0.0,sqrt(3)/3]))
            self.vectors.append(array([1.0,0.0]))
            self.vectors.append(array([0.5,sqrt(3)/2]))
        elif name=='H4':
            self.rcoords.append(array([0.0,0.0]))
            self.rcoords.append(array([0.0,sqrt(3)/3]))
            self.rcoords.append(array([0.5,sqrt(3)/2]))
            self.rcoords.append(array([0.5,-sqrt(3)/6]))
            self.vectors.append(array([1.0,0.0]))
            self.vectors.append(array([0.0,sqrt(3.0)]))
        elif name=='H6':
            self.rcoords.append(array([0.0,0.0]))
            self.rcoords.append(array([0.0,sqrt(3)/3]))
            self.rcoords.append(array([0.5,sqrt(3)/2]))
            self.rcoords.append(array([0.5,-sqrt(3)/6]))
            self.rcoords.append(array([1.0,0.0]))
            self.rcoords.append(array([1.0,sqrt(3)/3]))
            self.vectors.append(array([1.5,sqrt(3)/2]))
            self.vectors.append(array([1.5,-sqrt(3)/2]))
        elif name=='H8P':
            self.rcoords.append(array([0.0,0.0]))
            self.rcoords.append(array([0.0,sqrt(3)/3]))
            self.rcoords.append(array([0.5,sqrt(3)/2]))
            self.rcoords.append(array([0.5,-sqrt(3)/6]))
            self.rcoords.append(array([1.0,0.0]))
            self.rcoords.append(array([1.0,sqrt(3)/3]))
            self.rcoords.append(array([0.5,-sqrt(3)/2]))
            self.rcoords.append(array([0.5,sqrt(3)*5/6]))
            self.vectors.append(array([1.0,sqrt(3)]))
            self.vectors.append(array([1.5,-sqrt(3)/2]))
        elif name=='H8O':
            self.rcoords.append(array([0.0,0.0]))
            self.rcoords.append(array([0.0,sqrt(3)/3]))
            self.rcoords.append(array([0.5,sqrt(3)/2]))
            self.rcoords.append(array([0.5,-sqrt(3)/6]))
            self.rcoords.append(array([1.0,0.0]))
            self.rcoords.append(array([1.0,sqrt(3)/3]))
            self.rcoords.append(array([1.5,sqrt(3)/2]))
            self.rcoords.append(array([1.5,-sqrt(3)/6]))
            self.vectors.append(array([2.0,0.0]))
            self.vectors.append(array([0.0,sqrt(3)]))
        elif name=='H10':
            self.rcoords.append(array([0.0,0.0]))
            self.rcoords.append(array([0.0,sqrt(3)/3]))
            self.rcoords.append(array([0.5,sqrt(3)/2]))
            self.rcoords.append(array([0.5,-sqrt(3)/6]))
            self.rcoords.append(array([1.0,0.0]))
            self.rcoords.append(array([1.0,sqrt(3)/3]))
            self.rcoords.append(array([1.5,sqrt(3)/2]))
            self.rcoords.append(array([1.5,-sqrt(3)/6]))
            self.rcoords.append(array([2.0,0.0]))
            self.rcoords.append(array([2.0,sqrt(3)/3]))
            self.vectors.append(array([2.5,sqrt(3)/2]))
            self.vectors.append(array([0.0,sqrt(3)]))
        self.rcoords=asarray(self.rcoords)
        self.vectors=asarray(self.vectors)
