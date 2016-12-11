'''
Hexagonal lattice data base, including:
1) classes: HexagonDataBase
'''

__all__=['HexagonDataBase']

from numpy import *
from ..Basics.Geometry import *

class HexagonDataBase(object):
    '''
    Hexagonal lattice data base.
    Attributes:
        name: string
            The name of the cluster.
        points: list of Point
            The cluster points.
        vectors: list of 1d ndarray
            The tanslation vectors of the cluster.
    '''

    def __init__(self,name,scope=None):
        '''
        Constructor.
        Parameters:
            name: 'H2','H4','H6','H8O','H8P','H10'
                The name of the cluster wanted to be constructed.
            scope: string, optional
                The scope of the cluster points' pid.
        '''
        if name not in ['H2','H4','H6','H8O','H8P','H10']:
            raise ValueError('HexagonDataBase construction error: unexpected name(%s).'%name)
        self.name=name
        self.points=[]
        self.vectors=[]
        if name=='H2':
            self.points.append(Point(pid=PID(scope=scope,site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=1),rcoord=[0.0,sqrt(3)/3],icoord=[0.0,0.0]))
            self.vectors.append(array([1.0,0.0]))
            self.vectors.append(array([0.5,sqrt(3)/2]))
        elif name=='H4':
            self.points.append(Point(pid=PID(scope=scope,site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=1),rcoord=[0.0,sqrt(3)/3],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=2),rcoord=[0.5,sqrt(3)/2],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=3),rcoord=[0.5,-sqrt(3)/6],icoord=[0.0,0.0]))
            self.vectors.append(array([1.0,0.0]))
            self.vectors.append(array([0.0,sqrt(3.0)]))
        elif name=='H6':
            self.points.append(Point(pid=PID(scope=scope,site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=1),rcoord=[0.0,sqrt(3)/3],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=2),rcoord=[0.5,sqrt(3)/2],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=3),rcoord=[0.5,-sqrt(3)/6],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=4),rcoord=[1.0,0.0],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=5),rcoord=[1.0,sqrt(3)/3],icoord=[0.0,0.0]))
            self.vectors.append(array([1.5,sqrt(3)/2]))
            self.vectors.append(array([1.5,-sqrt(3)/2]))
        elif name=='H8P':
            self.points.append(Point(pid=PID(scope=scope,site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=1),rcoord=[0.0,sqrt(3)/3],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=2),rcoord=[0.5,sqrt(3)/2],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=3),rcoord=[0.5,-sqrt(3)/6],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=4),rcoord=[1.0,0.0],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=5),rcoord=[1.0,sqrt(3)/3],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=6),rcoord=[0.5,-sqrt(3)/2],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=7),rcoord=[0.5,sqrt(3)*5/6],icoord=[0.0,0.0]))
            self.vectors.append(array([1.0,sqrt(3)]))
            self.vectors.append(array([1.5,-sqrt(3)/2]))
        elif name=='H8O':
            self.points.append(Point(pid=PID(scope=scope,site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=1),rcoord=[0.0,sqrt(3)/3],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=2),rcoord=[0.5,sqrt(3)/2],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=3),rcoord=[0.5,-sqrt(3)/6],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=4),rcoord=[1.0,0.0],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=5),rcoord=[1.0,sqrt(3)/3],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=6),rcoord=[1.5,sqrt(3)/2],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=7),rcoord=[1.5,-sqrt(3)/6],icoord=[0.0,0.0]))
            self.vectors.append(array([2.0,0.0]))
            self.vectors.append(array([0.0,sqrt(3)]))
        elif name=='H10':
            self.points.append(Point(pid=PID(scope=scope,site=0),rcoord=[0.0,0.0],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=1),rcoord=[0.0,sqrt(3)/3],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=2),rcoord=[0.5,sqrt(3)/2],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=3),rcoord=[0.5,-sqrt(3)/6],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=4),rcoord=[1.0,0.0],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=5),rcoord=[1.0,sqrt(3)/3],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=6),rcoord=[1.5,sqrt(3)/2],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=7),rcoord=[1.5,-sqrt(3)/6],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=8),rcoord=[2.0,0.0],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=9),rcoord=[2.0,sqrt(3)/3],icoord=[0.0,0.0]))
            self.vectors.append(array([2.5,sqrt(3)/2]))
            self.vectors.append(array([0.0,sqrt(3)]))
