'''
Triangular lattice data base, including:
1) classes: TriangleDataBase
'''

__all__=['TriangleDataBase']

from numpy import array,sqrt
from ..Basics.GeometryPy import *

class TriangleDataBase:
    '''
    Triangular lattice data base.
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
            name: 'T1','T12'
                The name of the cluster wanted to be constructed.
            scope: string, optional
                The scope of the cluster points' pid.
        '''
        if name not in ['T1','T12']:
            raise ValueError('TriangleDataBase construction error: unexpected name(%s).'%name)
        self.points=[]
        self.vectors=[]
        if name=='T1':
            self.points.append(Point(pid=PID(scope=scope,site=(0,0,0)),rcoord=[0.0,0.0],icoord=[0.0,0.0]))
            self.vectors.append(array([1.0,0.0]))
            self.vectors.append(array([0.5,sqrt(3)/2]))
        elif name=='T12':
            self.points.append(Point(pid=PID(scope=scope,site=(0,0,0)),rcoord=[0.0,0.0],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=(0,0,1)),rcoord=[1.0,0.0],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=(0,0,2)),rcoord=[2.0,0.0],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=(0,0,3)),rcoord=[3.0,0.0],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=(0,0,4)),rcoord=[0.5,-sqrt(3.0)/2],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=(0,0,5)),rcoord=[1.5,-sqrt(3.0)/2],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=(0,0,6)),rcoord=[2.5,-sqrt(3.0)/2],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=(0,0,7)),rcoord=[1.0,-sqrt(3.0)],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=(0,0,8)),rcoord=[2.0,-sqrt(3.0)],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=(0,0,9)),rcoord=[0.5,sqrt(3.0)/2],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=(0,0,10)),rcoord=[1.5,sqrt(3.0)/2],icoord=[0.0,0.0]))
            self.points.append(Point(pid=PID(scope=scope,site=(0,0,11)),rcoord=[2.5,sqrt(3.0)/2],icoord=[0.0,0.0]))
            self.vectors.append(array([0.0,2*sqrt(3.0)]))
            self.vectors.append(array([3.0,sqrt(3.0)]))
