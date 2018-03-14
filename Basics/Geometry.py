'''
====================
Lattice construction
====================

This module provides enormous functions and classes to construct a lattice, including
    * functions: azimuthd, azimuth, polard, polar, volume, isparallel, isintratriangle, issubordinate, reciprocals, translation, rotation, tiling, minimumlengths, intralinks, interlinks
    * classes: PID, Point, Bond, Link, Lattice, SuperLattice, Cylinder
'''

__all__=['azimuthd', 'azimuth', 'polard', 'polar', 'volume', 'isparallel', 'isintratriangle', 'issubordinate', 'reciprocals', 'translation', 'rotation', 'tiling', 'minimumlengths','intralinks', 'interlinks', 'PID', 'Point', 'Bond', 'Link', 'Lattice', 'SuperLattice','Cylinder']

from Utilities import RZERO
from collections import namedtuple,Iterable
from scipy.spatial import cKDTree
import numpy as np
import numpy.linalg as nl
import matplotlib.pyplot as plt
import itertools as it
import warnings

def azimuthd(self):
    '''
    Azimuth in degrees of an array-like vector.
    '''
    if self[1]>=0:
        return np.degrees(np.arccos(self[0]/nl.norm(self[:-1] if len(self)==3 else self)))
    else:
        return 360-np.degrees(np.arccos(self[0]/nl.norm(self[:-1] if len(self)==3 else self)))

def azimuth(self):
    '''
    Azimuth in radians of an array-like vector.
    '''
    if self[1]>=0:
        return np.arccos(self[0]/nl.norm(self[:-1] if len(self)==3 else self))
    else:
        return 2*np.pi-np.arccos(self[0]/nl.norm(self[:-1] if len(self)==3 else self))

def polard(self):
    '''
    Polar angle in degrees of an array-like vector.
    '''
    if self.shape[0]==3:
        return np.degrees(np.arccos(self[2]/nl.norm(self)))
    else:
        raise ValueError("polard error: the array-like vector must contain three elements.")

def polar(self):
    '''
    Polar angle in radians of an array-like vector.
    '''
    if self.shape[0]==3:
        return np.arccos(self[2]/nl.norm(self))
    else:
        raise ValueError("polar error: the array-like vector must contain three elements.")

def volume(O1,O2,O3):
    '''
    Volume spanned by three array-like vectors.
    '''
    if O1.shape[0] in [1,2] or O2.shape[0] in [1,2] or O3.shape[0] in [1,2]:
        return 0
    elif O1.shape[0] ==3 and O2.shape[0]==3 and O3.shape[0]==3:
        return np.inner(O1,np.cross(O2,O3))
    else:
        raise ValueError("volume error: the shape of the array-like vectors is not supported.")

def isparallel(O1,O2):
    '''
    Judge whether two array-like vectors are parallel to each other.

    Parameters
    ----------
    O1,O2 : 1d array-like
        The input vectors.

    Returns
    -------
    int
        *  0: not parallel
        *  1: parallel
        * -1: anti-parallel
    '''
    norm1=nl.norm(O1)
    norm2=nl.norm(O2)
    if norm1<RZERO or norm2<RZERO:
        return 1
    elif O1.shape[0]==O2.shape[0]:
        buff=np.inner(O1,O2)/(norm1*norm2)
        if np.abs(buff-1)<RZERO:
            return 1
        elif np.abs(buff+1)<RZERO:
            return -1
        else:
            return 0
    else:
        raise ValueError("isparallel error: the shape of the array-like vectors does not match.")

def isonline(p0,p1,p2,ends=(True,True),rtol=RZERO):
    '''
    Judge whether a point is on a line segment.

    Parameters
    ----------
    p0 : 1d ndarray
        The coordinates of the point.
    p1,p2 : 1d ndarray
        The coordinates of the ends of the line segment.
    ends : 2-tuple of logical, optional
        Define whether the line segment contains its ends. True for YES and False for NO.
    rtol : np.float64, optional
        The relative tolerance of the error.

    Returns
    -------
    logical
        True for the point being on the line segment.
    '''
    d1,d2,d=nl.norm(p0-p1),nl.norm(p0-p2),nl.norm(p1-p2)
    return (np.abs(d1)<d*rtol and ends[0]) or (np.abs(d2)<d*rtol and ends[0]) or (np.abs(d1+d2-d)<d*rtol)

def isintratriangle(p0,p1,p2,p3,vertexes=(True,True,True),edges=(True,True,True)):
    '''
    Judge whether a point belongs to the interior of a triangle.

    Parameters
    ----------
    p0 : 1d ndarray
        The coordinates of the point.
    p1,p2,p3 : 1d ndarray
        The coordinates of the vertexes of the triangle.
    vertexes : 3-tuple of logical, optional
        Define whether the "interior" contains the vertexes of the triangle. True for YES and False for NO.
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
    a,b,x,ndim=np.zeros((3,3)),np.zeros(3),np.zeros(3),p0.shape[0]
    a[0:ndim,0]=p2-p1
    a[0:ndim,1]=p3-p1
    a[(2 if ndim==2 else 0):3,2]=np.cross(p2-p1,p3-p1)
    b[0:ndim]=p0-p1
    x=np.dot(nl.inv(a),b)
    assert x[2]==0
    onvertexes=[x[0]==0 and x[1]==0,x[0]==1 and x[1]==0,x[0]==0 and x[1]==1]
    onedges=[x[1]==0 and 0<x[0]<1,x[0]==0 and 0<x[1]<1,x[0]+x[1]==1 and 0<x[0]<1]
    if any(onvertexes):
        return any([on and condition for on,condition in zip(onvertexes,vertexes)])
    elif any(onedges):
        return any([on and condition for on,condition in zip(onedges,edges)])
    elif 0<x[0]<1 and 0<x[1]<1 and x[0]+x[1]<1:
        return True
    else:
        return False

def issubordinate(rcoord,vectors):
    '''
    Judge whether or not a coordinate belongs to a lattice defined by vectors.

    Parameters
    ----------
    rcoord : 1d array-like
        The coordinate.
    vectors : list of 1d array-like
        The translation vectors of the lattice.

    Returns
    -------
    logical
        True for yes False for no.

    Notes
    -----
    Only 1,2 and 3 dimensional coordinates are supported.
    '''
    nvectors=len(vectors)
    ndim=len(vectors[0])
    a=np.zeros((3,3))
    for i in xrange(nvectors):
        a[0:ndim,i]=vectors[i]
    if nvectors==2:
        if ndim==2:
            buff=np.zeros(3)
            buff[2]=np.cross(vectors[0],vectors[1])
        else:
            buff=np.cross(vectors[0],vectors[1])
        a[:,2]=buff
    if nvectors==1:
        buff1,buff2=a[:,0],np.zeros(3)
        for i in xrange(3):
            buff2[...]=0.0
            buff2[i]=np.pi
            if not isparallel(buff1,buff2): break
        buff3=np.cross(buff1,buff2)
        a[:,1]=buff2
        a[:,2]=buff3
    b=np.zeros(3)
    b[0:len(rcoord)]=rcoord
    x=nl.inv(a).dot(b)
    if np.max(np.abs(x-np.around(x)))<RZERO:
        return True
    else:
        return False

def reciprocals(vectors):
    '''
    Return the corresponding reciprocals dual to the input vectors.

    Parameters
    ----------
    vectors : 2d array-like
        The input vectors.

    Returns
    -------
    2d array-like
        The reciprocals.
    '''
    result=[]
    nvectors=len(vectors)
    if nvectors==0:
        return []
    if nvectors==1:
        result.append(np.array(vectors[0]/(nl.norm(vectors[0]))**2*2*np.pi))
    elif nvectors in (2,3):
        ndim=vectors[0].shape[0]
        buff=np.zeros((3,3))
        buff[0:ndim,0]=vectors[0]
        buff[0:ndim,1]=vectors[1]
        if nvectors==2:
            buff[(2 if ndim==2 else 0):3,2]=np.cross(vectors[0],vectors[1])
        else:
            buff[0:ndim,2]=vectors[2]
        buff=nl.inv(buff)
        result.append(np.array(buff[0,0:ndim]*2*np.pi))
        result.append(np.array(buff[1,0:ndim]*2*np.pi))
        if nvectors==3:
            result.append(np.array(buff[2,0:ndim]*2*np.pi))
    else:
        raise ValueError('Reciprocals error: the number of translation vectors should not be greater than 3.')
    return result

def translation(cluster,vector):
    '''
    This function returns the translated cluster.

    Parameters
    ----------
    cluster : list of 1d array-like
        The original cluster.
    vector : 1d array-like
        The translation vector.

    Returns
    -------
    list of 1d ndarray
        The translated cluster.
    '''
    return [np.asarray(coord)+np.asarray(vector) for coord in cluster]

def rotation(cluster,angle=0,center=None):
    '''
    This function returns the rotated cluster.

    Parameters
    ----------
    cluster : list of 1d array-like
        The original cluster.
    angle : float
        The rotated angle. Clockwise for negative and anticlockwise for positive.
    center : 1d array-like, optional
        The center of the axis. Default the origin.

    Returns
    -------
    list of 1d ndarray
        The rotated coords.
    '''
    if center is None: center=0
    m11=np.cos(angle);m21=-np.sin(angle);m12=-m21;m22=m11
    m=np.array([[m11,m12],[m21,m22]])
    return [m.dot(np.asarray(coord)-np.asarray(center))+np.asarray(center) for coord in cluster]

def tiling(cluster,vectors=(),translations=()):
    '''
    Tile a supercluster by translations of the input cluster.

    Parameters
    ----------
    cluster : list of 1d ndarray
        The original cluster.
    vectors : list of 1d ndarray, optional
        The translation vectors.
    translations : iterator of tuple, optional
        The translations of the cluster.

    Returns
    -------
    list of 1d ndarray
        The supercluster tiled from the translations of the input cluster.
    '''
    supercluster=[]
    if len(vectors)==0: vectors,translations=0,(0,)
    for translation in translations:
        disp=np.dot(tuple(translation) if isinstance(translation,Iterable) else (translation,),vectors)
        for coord in cluster:
            supercluster.append(coord+disp)
    return supercluster

class PID(namedtuple('PID',['scope','site'])):
    '''
    The ID of a point.

    Attributes
    ----------
    scope : string
        The scope in which the point lives.
        Usually, it is same to the name of the cluster/sublattice/lattice the point belongs to.
    site : integer
        The site index of the point.
    '''

PID.__new__.__defaults__=(None,)*len(PID._fields)

class Point(object):
    '''
    Point class.

    Attributes
    ----------
    pid : PID
        The point id.
    rcoord : 1d ndarray
        The point rcoord.
    icoord : 1d ndarray
        The point icoord.
    '''

    def __init__(self,pid,rcoord,icoord=None):
        '''
        Constructor.

        Parameters
        ----------
        pid : PID
            The point id.
        rcoord : 1d array-like
            The point rcoord.
        icoord : 1d array-like, optional
            The point icoord.
        '''
        assert isinstance(pid,PID)
        self.pid=pid
        self.rcoord=np.asarray(rcoord)
        self.icoord=np.zeros(self.rcoord.shape) if icoord is None else np.asarray(icoord)

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return 'Point(pid=%s, rcoord=%s, icoord=%s)'%(self.pid,self.rcoord,self.icoord)

    __repr__=__str__

    def __eq__(self,other):
        '''
        Overloaded operator(==).
        '''
        return self.pid==other.pid and nl.norm(self.rcoord-other.rcoord)<RZERO and nl.norm(self.icoord-other.icoord)<RZERO
    
    def __ne__(self,other):
        '''
        Overloaded operator(!=).
        '''
        return not self==other

class Bond(object):
    '''
    This class describes a bond in a lattice.

    Attributes
    ----------
    neighbour : integer
        The rank of the neighbour of the bond.
    spoint : Point
        The start point of the bond.
    epoint : Point
        The end point of the bond.
    '''

    def __init__(self,neighbour,spoint,epoint):
        '''
        Constructor.

        Parameters
        ----------
        neighbour : integer
            The rank of the neighbour of the bond.
        spoint : Point
            The start point of the bond.
        epoint : Point
            The end point of the bond.
        '''
        self.neighbour=neighbour
        self.spoint=spoint
        self.epoint=epoint

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        return 'Bond(%s, %s, %s)'%(self.neighbour,self.spoint,self.epoint)
    
    @property
    def rcoord(self):
        '''
        The real coordinate of a bond.
        '''
        return self.epoint.rcoord-self.spoint.rcoord
    
    @property
    def icoord(self):
        '''
        The lattice coordinate of a bond.
        '''
        return self.epoint.icoord-self.spoint.icoord
    
    def isintracell(self):
        '''
        Judge whether a bond is intra the unit cell or not. 
        '''
        if nl.norm(self.icoord)< RZERO:
            return True
        else:
            return False

    @property
    def reversed(self):
        '''
        Return the reversed bond.
        '''
        return Bond(self.neighbour,self.epoint,self.spoint)

class Link(object):
    '''
    This class describes a link in a lattice.

    Attributes
    ----------
    neighbour : int
        The rank of the neighbour of the link.
    sindex : integer
        The start index of the link in the lattice.
    eindex : integer
        The end index of the link in the lattice.
    disp : 1d ndarray
        The displacement of the link.
    '''

    def __init__(self,neighbour,sindex,eindex,disp):
        '''
        Constructor.

        Parameters
        ----------
        neighbour : integer
            The rank of the neighbour of the link.
        sindex : integer
            The start index of the link in the lattice.
        eindex : integer
            The end index of the link in the lattice.
        disp : 1d ndarray
            The displacement of the link.
        '''
        self.neighbour=neighbour
        self.sindex=sindex
        self.eindex=eindex
        self.disp=disp

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        return 'Link(%s, %s, %s, %s)'%(self.neighbour,self.sindex,self.eindex,self.disp)

def minimumlengths(cluster,vectors=(),nneighbour=1,max_coordinate_number=8):
    '''
    This function searches the minimum bond lengths of a cluster.

    Parameters
    ----------
    cluster : list of 1d ndarray
        The coordinates of the cluster.
    vectors : list of 1d ndarray, optional
        The translation vectors of the cluster.
    nneighbour : int, optional
        The order of the minimum bond lengths.
    max_coordinate_number : int, optional
        The max coordinate number for every order of neighbour.

    Returns
    -------
    1d ndarray
        The minimum bond lengths.
    '''
    assert nneighbour>=0
    result=np.array([np.inf]*(nneighbour+1))
    if len(cluster)>0:
        translations=list(it.product(*([xrange(-nneighbour,nneighbour+1)]*len(vectors))))
        for translation in translations:
            if any(translation): translations.remove(tuple([-i for i in translation]))
        translations=sorted(translations,key=nl.norm)
        supercluster=tiling(cluster,vectors=vectors,translations=translations)
        for length in cKDTree(supercluster).query(cluster,k=nneighbour*max_coordinate_number if nneighbour>0 else 1)[0].flatten():
            for i,minlength in enumerate(result):
                if abs(length-minlength)<RZERO:
                    break
                elif length<minlength:
                    if nneighbour>0: result[i+1:nneighbour+1]=result[i:nneighbour]
                    result[i]=length
                    break
        if np.any(result==np.inf):
            warnings.warn('minimumlengths warning: np.inf remained in the result. Larger(>%s) max_coordinate_number may be needed.'%max_coordinate_number)
    return result

def intralinks(cluster,vectors=(),max_translations=None,neighbours=None):
    '''
    This function searches a certain set of neighbours intra a cluster.

    Parameters
    ----------
    cluster : list of 1d ndarray
        The coordinates of the cluster.
    vectors : list of 1d ndarray, optional
        The translation vectors of the cluster.
    max_translations: tuple of int, optional
        The maximum translations of the original cluster.
    neighbours : dict, optional
        The neighbour-length map of the bonds to be searched.

    Returns
    -------
    list of Link
        The searched links.

    Notes
    -----
        * When `vectors` **NOT** empty, periodic boundary condition is assumed and the links across the boundaries of the cluster are also searched.
    '''
    result=[]
    if len(cluster)>0:
        if max_translations is None: max_translations=[len(neighbours)-1]*len(vectors)
        if neighbours is None: neighbours={0:0.0}
        assert len(max_translations)==len(vectors)
        translations=list(it.product(*[xrange(-nnb,nnb+1) for nnb in max_translations]))
        for translation in translations:
            if any(translation): translations.remove(tuple([-i for i in translation]))
        translations=sorted(translations,key=nl.norm)
        supercluster=tiling(cluster,vectors=vectors,translations=translations)
        disps=tiling([np.zeros(len(next(iter(cluster))))]*len(cluster),vectors=vectors,translations=translations)
        smatrix=cKDTree(cluster).sparse_distance_matrix(cKDTree(supercluster),np.max(neighbours.values())+RZERO)
        for (i,j),dist in smatrix.items():
            if i<=j:
                for neighbour,length in neighbours.iteritems():
                    if abs(length-dist)<RZERO:
                        result.append(Link(neighbour,sindex=i,eindex=j%len(cluster),disp=disps[j]))
                        break
    return result

def interlinks(cluster1,cluster2,neighbours=None):
    '''
    This function searches a certain set of neighbours between two clusters.

    Parameters
    ----------
    cluster1, cluster2 : list of 1d ndarray
        The coordinates of the clusters.
    neighbours : dict, optional
        The neighbour-length map of the links to be searched.

    Returns
    -------
    list of Link
        The searched links.
    '''
    result=[]
    if len(cluster1)>0 and len(cluster2)>0:
        if neighbours is None: neighbours={0:0.0}
        smatrix=cKDTree(cluster1).sparse_distance_matrix(cKDTree(cluster2),np.max(neighbours.values())+RZERO)
        for (i,j),dist in smatrix.items():
            for neighbour,length in neighbours.iteritems():
                if abs(length-dist)<RZERO:
                    result.append(Link(neighbour,sindex=i,eindex=j,disp=0))
                    break
    return result

class Lattice(object):
    '''
    This class provides a unified description of 1d, quasi 1d, 2D, quasi 2D and 3D lattice systems.

    Attributes
    ----------
    name : string
        The lattice's name.
    pids : list of PID
        The pids of the lattice.
    rcoords : 2d ndarray
        The rcoords of the lattice.
    icoords : 2d ndarray
        The icoords of the lattice.
    vectors : list of 1d ndarray
        The translation vectors.
    reciprocals : list of 1d ndarray
        The dual translation vectors.
    neighbours : dict
        The neighbour-length map of the lattice.
    '''
    ZMAX=8

    def __init__(self,name,pids=None,rcoords=(),icoords=None,vectors=(),neighbours=1):
        '''
        Construct a lattice directly from its coordinates.

        Parameters
        ----------
        name : str
            The name of the lattice.
        pids : list of PID, optional
            The pids of the lattice.
        rcoords : 2d array-like, optional
            The rcoords of the lattice.
        icoords : 2d array-like, optional
            The icoords of the lattice.
        vectors : list of 1d ndarray, optional
            The translation vectors of the lattice.
        neighbours : dict, optional
            The neighbour-length map of the lattice.
        '''
        rcoords=np.asarray(rcoords)
        if pids is None: pids=[PID(scope=name,site=i) for i in xrange(len(rcoords))]
        if icoords is None: icoords=np.zeros(rcoords.shape)
        assert len(pids)==len(rcoords)==len(icoords)
        self.name=name
        self.pids=pids
        self.rcoords=rcoords
        self.icoords=icoords
        self.vectors=vectors
        self.reciprocals=reciprocals(vectors)
        self.neighbours=neighbours if isinstance(neighbours,dict) else {i:length for i,length in enumerate(minimumlengths(rcoords,vectors,neighbours,Lattice.ZMAX))}

    @staticmethod
    def compose(name,points=(),vectors=(),neighbours=1):
        '''
        Construct a lattice from its contained points.

        Parameters
        ----------
        name : str
            The name of the lattice.
        points : list of Point, optional
            The lattice points.
        vectors : list of 1d ndarray, optional
            The translation vectors of the lattice.
        neighbours : dict, optional
            The neighbour-length map of the lattice.

        Returns
        -------
        Lattice
            The composed lattice.
        '''
        return Lattice(
                    name=           name,
                    pids=           [point.pid for point in points],
                    rcoords=        [point.rcoord for point in points],
                    icoords=        [point.icoord for point in points],
                    vectors=        vectors,
                    neighbours=     neighbours
                    )

    @staticmethod
    def merge(name,sublattices,vectors=(),neighbours=1):
        '''
        Merge sublattices into a new lattice.

        Parameters
        ----------
        name : str
            The name of the new lattice.
        sublattices : list of Lattice
            The sublattices of the new lattice.
        vectors : list of 1d ndarray, optional
            The translation vectors of the new lattice.
        neighbours : dict, optional
            The neighbour-length map of the lattice.

        Returns
        -------
        Lattice
            The merged lattice.
        '''
        return Lattice(
                    name=           name,
                    pids=           [pid for lattice in sublattices for pid in lattice.pids],
                    rcoords=        np.concatenate([lattice.rcoords for lattice in sublattices]),
                    icoords=        np.concatenate([lattice.icoords for lattice in sublattices]),
                    vectors=        vectors,
                    neighbours=     neighbours
                    )

    def __len__(self):
        '''
        The number of points contained in this lattice.
        '''
        return len(self.pids)

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        return '%s(%s)'%(self.__class__.__name__,self.name)

    @property
    def nneighbour(self):
        '''
        The highest order of neighbours.
        '''
        return len(self.neighbours)-1

    @property
    def points(self):
        '''
        The points of the lattice.
        '''
        return [Point(pid=pid,rcoord=rcoord,icoord=icoord) for pid,rcoord,icoord in zip(self.pids,self.rcoords,self.icoords)]

    @property
    def bonds(self):
        '''
        The bonds of the lattice.
        '''
        result=[]
        for link in intralinks(self.rcoords,vectors=self.vectors,neighbours=self.neighbours):
            spoint=Point(pid=self.pids[link.sindex],rcoord=self.rcoords[link.sindex],icoord=self.icoords[link.sindex])
            epoint=Point(pid=self.pids[link.eindex],rcoord=self.rcoords[link.eindex]+link.disp,icoord=self.icoords[link.eindex]+link.disp)
            result.append(Bond(link.neighbour,spoint,epoint))
        return result

    def sublattice(self,name,subset):
        '''
        A sublattice of the original lattice.

        Parameters
        ----------
        name : str
            The name of the sublattice.
        subset : list of PID/int
            The sub-pids/sub-indices of the sublattice.

        Returns
        -------
        Lattice
            The sublattice.
        '''
        subset=[self.pids.index(index) if isinstance(index,PID) else index for index in subset]
        return Lattice(
                    name=           name,
                    pids=           [self.pids[index] for index in subset],
                    rcoords=        self.rcoords[subset],
                    icoords=        self.icoords[subset],
                    vectors=        self.vectors,
                    neighbours=     self.neighbours
                    )

    def point(self,pid):
        '''
        Return a specific point of the lattice.

        Parameters
        ----------
        pid : PID
            The pid of the point.

        Returns
        -------
        Point
            The point.
        '''
        return Point(pid=pid,rcoord=self.rcoord(pid),icoord=self.icoord(pid))

    def rcoord(self,pid):
        '''
        Return the rcoord of a point.

        Parameters
        ----------
        pid : PID
            The pid of the point.

        Returns
        -------
        1d ndarray
            The rcoord of the point.
        '''
        return self.rcoords[self.pids.index(pid)]

    def icoord(self,pid):
        '''
        Return the icoord of a point.

        Parameters
        ----------
        pid : PID
            The pid of the point.

        Returns
        -------
        1d ndarray
            The icoord of the point.
        '''
        return self.icoords[self.pids.index(pid)]

    def append(self,point):
        '''
        Append a point to the lattice.

        Parameters
        ----------
        point : Point or 2-tuple
            The inserted point.
        '''
        pid,rcoord=(point.pid,point.rcoord) if isinstance(point,Point) else point
        self.pids.append(pid)
        self.rcoords=np.append(self.rcoords,[rcoord],axis=0)

    def insert(self,position,point):
        '''
        Insert a point to the lattice.

        Parameters
        ----------
        position : PID or int
            The position before which to insert the point.
        point : Point or 2-tuple
            The inserted point.
        '''
        if isinstance(position,PID): position=self.pids.index(position)
        pid,rcoord=(point.pid,point.rcoord) if isinstance(point,Point) else point
        self.pids.insert(position,pid)
        self.rcoords=np.insert(self.rcoords,position,rcoord,axis=0)

    def plot(self,show=True,suspend=False,save=True,close=True,pidon=False):
        '''
        Plot the lattice points and bonds. Only 2D or quasi 1d systems are supported.
        '''
        ax=plt.subplots()[1]
        ax.axis('off')
        ax.axis('equal')
        ax.set_title(self.name)
        for bond in self.bonds:
            nb=bond.neighbour if 0<=bond.neighbour<np.inf else self.nneighbour+(2 if bond.neighbour<0 else 1)
            color='k' if nb==1 else 'r' if nb==2 else 3 if nb==3 else str(nb*1.0/(self.nneighbour+10))
            if nb==0:
                pid,x,y=bond.spoint.pid,bond.spoint.rcoord[0],bond.spoint.rcoord[1]
                ax.scatter(x,y)
                if pidon: ax.text(x,y,'%s%s'%('' if pid.scope is None else str(pid.scope)+'*',pid.site),fontsize=10,color='blue',ha='center',va='bottom')
            else:
                ax.plot([bond.spoint.rcoord[0],bond.epoint.rcoord[0]],[bond.spoint.rcoord[1],bond.epoint.rcoord[1]],color=color,ls='-' if bond.isintracell() else '--')
        if show and suspend: plt.show()
        if show and not suspend: plt.pause(1)
        if save: plt.savefig(self.name+'.png')
        if close:plt.close()

class SuperLattice(Lattice):
    '''
    This class is the union of sublattices.

    Attributes
    ---------
    sublattices : list of Lattice
        The sublattices of the superlattice.
    '''

    def __init__(self,name,sublattices,vectors=(),neighbours=1):
        '''
        Constructor.

        Parameters
        ----------
        name : str
            The name of the superlattice.
        sublattices : list of Lattice
            The sublattices of the superlattice.
        vectors : list of 1d ndarray, optional
            The translation vectors of the superlattice.
        neighbours : dict, optional
            The neighbour-length map of the lattice.
        '''
        self.name=name
        self.sublattices=sublattices
        self.pids=[pid for lattice in sublattices for pid in lattice.pids]
        self.rcoords=np.concatenate([lattice.rcoords for lattice in sublattices])
        self.icoords=np.concatenate([lattice.icoords for lattice in sublattices])
        self.vectors=vectors
        self.reciprocals=reciprocals(vectors)
        self.neighbours=neighbours if isinstance(neighbours,dict) else {i:length for i,length in enumerate(minimumlengths(self.rcoords,vectors,neighbours,Lattice.ZMAX))}

    @property
    def bonds(self):
        '''
        The bonds of the superlattice.
        '''
        result=[bond for lattice in self.sublattices for bond in lattice.bonds]
        for sub1,sub2 in it.combinations(self.sublattices,2):
            for link in interlinks(sub1.rcoords,sub2.rcoords,neighbours=self.neighbours):
                spoint=Point(pid=sub1.pids[link.sindex],rcoord=sub1.rcoords[link.sindex],icoord=sub1.icoords[link.sindex])
                epoint=Point(pid=sub2.pids[link.eindex],rcoord=sub2.rcoords[link.eindex]+link.disp,icoord=sub2.icoords[link.eindex]+link.disp)
                result.append(Bond(link.neighbour,spoint,epoint))
        return result

class Cylinder(Lattice):
    '''
    The cylinder geometry of a lattice.

    Attributes
    ----------
    block : list of 1d ndarray
        The building block of the cylinder.
    translation : 1d ndarray
        The translation vector of the building block to construct the cylinder.
    '''

    def __init__(self,block,translation,**karg):
        '''
        Constructor.

        Parameters
        ----------
        block : list of 1d ndarray
            The building block of the cylinder.
        translation : 1d ndarray
            The translation vector of the building block to construct the cylinder.
        '''
        super(Cylinder,self).__init__(**karg)
        self.block=block
        self.translation=translation

    def insert(self,A,B,news=None):
        '''
        Insert two blocks into the center of the cylinder.

        Parameters
        ----------
        A,B : any hashable object
            The scopes of the insert block points.
        news : list of any hashable object, optional
            The new scopes for the points of the cylinder before the insertion.
            If None, the old scopes remain unchanged.
        '''
        aspids,bspids,asrcoords,bsrcoords=[],[],[],[]
        for i,rcoord in enumerate(self.block):
            aspids.append(PID(scope=A,site=i))
            bspids.append(PID(scope=B,site=i))
            asrcoords.append(rcoord-self.translation/2)
            bsrcoords.append(rcoord+self.translation/2)
        if len(self)==0:
            self.pids=aspids+bspids
            self.rcoords=np.vstack([asrcoords,bsrcoords])
        else:
            if news is not None:
                assert len(news)*len(self.block)==len(self)
                self.pids=[PID(scope=scope,site=i) for scope in news for i in xrange(len(self.block))]
            apids,bpids=self.pids[:len(self)/2],self.pids[len(self)/2:]
            arcoords,brcoords=self.rcoords[:len(self)/2]-self.translation,self.rcoords[len(self)/2:]+self.translation
            self.pids=apids+aspids+bspids+bpids
            self.rcoords=np.vstack([arcoords,asrcoords,bsrcoords,brcoords])
        self.icoords=np.zeros(self.rcoords.shape)
        if np.any(np.asarray(self.neighbours.values())==np.inf):
            self.neighbours={i:length for i,length in enumerate(minimumlengths(self.rcoords,self.vectors,self.nneighbour,Lattice.ZMAX))}

    def __call__(self,scopes):
        '''
        Construct a cylinder with the assigned scopes.

        Parameters
        ----------
        scopes : list of hashable object
            The scopes of the cylinder.

        Returns
        -------
        Lattice
            The constructed cylinder.
        '''
        result=Lattice(
                    name=           self.name.replace('+',str(len(scopes))),
                    pids=           [PID(scope=scope,site=i) for scope in scopes for i in xrange(len(self.block))],
                    rcoords=        tiling(self.block,[self.translation],np.linspace(-(len(scopes)-1)/2.0,(len(scopes)-1)/2.0,len(scopes)) if len(scopes)>1 else xrange(1)),
                    icoords=        None,
                    vectors=        self.vectors,
                    neighbours=     self.neighbours
                    )
        if np.any(np.asarray(result.neighbours.values())==np.inf):
            result.neighbours={i:length for i,length in enumerate(minimumlengths(result.rcoords,result.vectors,result.nneighbour,Lattice.ZMAX))}
        return result
