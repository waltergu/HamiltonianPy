'''
------------
Lattice pack
------------

Lattice pack, including:
    * classes: Cluster
    * functions: Square, Hexagon, Triangle, Kagome
'''

__all__=['Cluster','Square','Hexagon','Triangle','Kagome']

from ..Geometry import Lattice,tiling
from numpy import asarray,array,sqrt
import itertools as it
import re

class Cluster(object):
    '''
    Cluster, the building block of a lattice.

    Attributes
    ----------
    name : string
        The name of the cluster.
    rcoords : 2d ndarray
        The rcoords of the cluster.
    vectors : 2d ndarray
        The tanslation vectors of the cluster.
    '''

    def __init__(self,name,rcoords,vectors):
        '''
        Constructor.

        Parameters
        ----------
        name : string
            The name of the cluster.
        rcoords : 2d ndarray
            The rcoords of the cluster.
        vectors : 2d ndarray
            The tanslation vectors of the cluster.
        '''
        self.name=name
        self.rcoords=asarray(rcoords)
        self.vectors=asarray(vectors)

    def __call__(self,tbs=None,nneighbour=1):
        '''
        Construct a lattice acoording the translation and boundary conditions.

        Parameters
        ----------
        tbs : str, optional
            The translation and boundary conditions.
        nneighbour : integer, optional
            The highest order of the neighbours.

        Returns
        -------
        Lattice
            The constructed lattice.
        '''
        tbs=tbs or '-'.join(['1P']*len(self.vectors))
        ts,bcs=re.findall('\d+',tbs),re.findall('[P,p,O,o]',tbs)
        assert len(ts)==len(self.vectors) and len(bcs)==len(self.vectors)
        return Lattice(
                name=       '%s(%s)'%(self.name,tbs.upper()),
                rcoords=    tiling(cluster=self.rcoords,vectors=self.vectors,translations=it.product(*[xrange(int(t)) for t in ts])),
                vectors=    [self.vectors[i] for i,bc in enumerate(bcs) if bc.lower()=='p'],
                nneighbour= nneighbour
                )

    def tiling(self,ts):
        '''
        Construct a new cluster by tiling.

        Parameters
        ----------
        ts : tuple of integer
            The translation and boundary conditions.

        Returns
        -------
        Cluster
            The new cluster after the tiling.
        '''
        assert len(ts)==len(self.vectors)
        return Cluster(
                name=       '%s^%s'%(self.name,'-'.join(str(t) for t in ts)),
                rcoords=    tiling(cluster=self.rcoords,vectors=self.vectors,translations=it.product(*[xrange(int(t)) for t in ts])),
                vectors=    [self.vectors[i]*t for i,t in enumerate(ts)]
                )

def Square(name):
    '''
    Cluster of square lattices.

    Parameters
    ----------
    name : 'S1'
        The name of the cluster.

    Returns
    -------
    Cluster
        The cluster of square lattices.
    '''
    if name not in ['S1']:
        raise ValueError('Square error: unexpected name(%s).'%name)
    rcoords,vectors=[],[]
    if name=='S1':
        rcoords.append(array([0.0,0.0]))
        vectors.append(array([1.0,0.0]))
        vectors.append(array([0.0,1.0]))
    return Cluster(name,rcoords,vectors)

def Hexagon(name):
    '''
    Cluster of hexagonal lattices.

    Parameters
    ----------
    name : 'H2','H4','H6','H8O','H8P','H10'
        The name of the cluster.

    Returns
    -------
    Cluster
        The cluster of hexagonal lattices.
    '''
    if name not in ['H2','H4','H6','H8O','H8P','H10']:
        raise ValueError('Hexagon error: unexpected name(%s).'%name)
    rcoords,vectors=[],[]
    if name=='H2':
        rcoords.append(array([0.0,0.0]))
        rcoords.append(array([0.0,sqrt(3)/3]))
        vectors.append(array([1.0,0.0]))
        vectors.append(array([0.5,sqrt(3)/2]))
    elif name=='H4':
        rcoords.append(array([0.0,0.0]))
        rcoords.append(array([0.0,sqrt(3)/3]))
        rcoords.append(array([0.5,sqrt(3)/2]))
        rcoords.append(array([0.5,-sqrt(3)/6]))
        vectors.append(array([1.0,0.0]))
        vectors.append(array([0.0,sqrt(3.0)]))
    elif name=='H6':
        rcoords.append(array([0.0,0.0]))
        rcoords.append(array([0.0,sqrt(3)/3]))
        rcoords.append(array([0.5,sqrt(3)/2]))
        rcoords.append(array([0.5,-sqrt(3)/6]))
        rcoords.append(array([1.0,0.0]))
        rcoords.append(array([1.0,sqrt(3)/3]))
        vectors.append(array([1.5,sqrt(3)/2]))
        vectors.append(array([1.5,-sqrt(3)/2]))
    elif name=='H8P':
        rcoords.append(array([0.0,0.0]))
        rcoords.append(array([0.0,sqrt(3)/3]))
        rcoords.append(array([0.5,sqrt(3)/2]))
        rcoords.append(array([0.5,-sqrt(3)/6]))
        rcoords.append(array([1.0,0.0]))
        rcoords.append(array([1.0,sqrt(3)/3]))
        rcoords.append(array([0.5,-sqrt(3)/2]))
        rcoords.append(array([0.5,sqrt(3)*5/6]))
        vectors.append(array([1.0,sqrt(3)]))
        vectors.append(array([1.5,-sqrt(3)/2]))
    elif name=='H8O':
        rcoords.append(array([0.0,0.0]))
        rcoords.append(array([0.0,sqrt(3)/3]))
        rcoords.append(array([0.5,sqrt(3)/2]))
        rcoords.append(array([0.5,-sqrt(3)/6]))
        rcoords.append(array([1.0,0.0]))
        rcoords.append(array([1.0,sqrt(3)/3]))
        rcoords.append(array([1.5,sqrt(3)/2]))
        rcoords.append(array([1.5,-sqrt(3)/6]))
        vectors.append(array([2.0,0.0]))
        vectors.append(array([0.0,sqrt(3)]))
    elif name=='H10':
        rcoords.append(array([0.0,0.0]))
        rcoords.append(array([0.0,sqrt(3)/3]))
        rcoords.append(array([0.5,sqrt(3)/2]))
        rcoords.append(array([0.5,-sqrt(3)/6]))
        rcoords.append(array([1.0,0.0]))
        rcoords.append(array([1.0,sqrt(3)/3]))
        rcoords.append(array([1.5,sqrt(3)/2]))
        rcoords.append(array([1.5,-sqrt(3)/6]))
        rcoords.append(array([2.0,0.0]))
        rcoords.append(array([2.0,sqrt(3)/3]))
        vectors.append(array([2.5,sqrt(3)/2]))
        vectors.append(array([0.0,sqrt(3)]))
    return Cluster(name,rcoords,vectors)

def Triangle(name):
    '''
    Cluster of triangular lattices.

    Parameters
    ----------
    name : 'T1','T12'
        The name of the cluster.

    Returns
    -------
    Cluster
        The cluster of triangular lattices.
    '''
    if name not in ['T1','T12']:
        raise ValueError('Triangle error: unexpected name(%s).'%name)
    rcoords,vectors=[],[]
    if name=='T1':
        rcoords.append(array([0.0,0.0]))
        vectors.append(array([1.0,0.0]))
        vectors.append(array([0.5,sqrt(3)/2]))
    elif name=='T12':
        rcoords.append(array([0.0,0.0]))
        rcoords.append(array([1.0,0.0]))
        rcoords.append(array([2.0,0.0]))
        rcoords.append(array([3.0,0.0]))
        rcoords.append(array([0.5,-sqrt(3.0)/2]))
        rcoords.append(array([1.5,-sqrt(3.0)/2]))
        rcoords.append(array([2.5,-sqrt(3.0)/2]))
        rcoords.append(array([1.0,-sqrt(3.0)]))
        rcoords.append(array([2.0,-sqrt(3.0)]))
        rcoords.append(array([0.5,sqrt(3.0)/2]))
        rcoords.append(array([1.5,sqrt(3.0)/2]))
        rcoords.append(array([2.5,sqrt(3.0)/2]))
        vectors.append(array([0.0,2*sqrt(3.0)]))
        vectors.append(array([3.0,sqrt(3.0)]))
    return Cluster(name,rcoords,vectors)

def Kagome(name):
    '''
    Cluster of Kagome lattices.

    Parameters
    ----------
    name : 'K3'
        The name of the cluster.

    Returns
    -------
    Cluster
        The cluster of Kagome lattices.
    '''
    if name not in ['K3']:
        raise ValueError('Kagome error: unexpected name(%s).'%name)
    rcoords,vectors=[],[]
    if name=='K3':
        rcoords.append(array([0.0,0.0]))
        rcoords.append(array([0.5,0.0]))
        rcoords.append(array([0.25,sqrt(3)/4]))
        vectors.append(array([1.0,0.0]))
        vectors.append(array([0.5,sqrt(3)/2]))
    return Cluster(name,rcoords,vectors)
