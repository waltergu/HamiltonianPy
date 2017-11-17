'''
====================
Lattice construction
====================

This module provides enormous functions and classes to construct a lattice, including
    * functions: azimuthd, azimuth, polard, polar, volume, isparallel, isintratriangle, issubordinate, reciprocals, translation, rotation, tiling, intralinks, interlinks
    * classes: PID, Point, Bond, Link, Lattice, SuperLattice, Cylinder
'''

__all__=['azimuthd', 'azimuth', 'polard', 'polar', 'volume', 'isparallel', 'isintratriangle', 'issubordinate', 'reciprocals', 'translation', 'rotation', 'tiling', 'intralinks', 'interlinks', 'PID', 'Point', 'Bond', 'Link', 'Lattice', 'SuperLattice','Cylinder']

from Utilities import RZERO
from collections import namedtuple,Iterable
from scipy.spatial import cKDTree
import numpy as np
import numpy.linalg as nl
import matplotlib.pyplot as plt
import itertools as it

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
    if max(np.abs(x-np.around(x)))<RZERO:
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

class Point(np.ndarray):
    '''
    Point, which is a 2d ndarray with a shape of (2,N), with N being the dimension of the coordinates, and
        * Point[0,:]: The rcoord part of the Point.
        * Point[1,:]: The icoord part of the Point.

    Attributes
    ----------
    pid : PID
        The specific ID of a point.
    '''

    def __new__(cls,pid,rcoord=None,icoord=None):
        '''
        Constructor.

        Parameters
        ----------
        pid : PID
            The specific ID of a point
        rcoord : 1d array-like
            The coordinate in real space.
        icoord : 1d array-like,optional
            The coordinate in lattice space.
        '''
        assert isinstance(pid,PID)
        result=np.asarray([[] if rcoord is None else rcoord, [] if icoord is None else icoord]).view(cls)
        result.pid=pid
        return result

    def __array_finalize__(self,obj):
        '''
        Initialize an instance through both explicit and implicit constructions, i.e. constructor, view and slice.
        '''
        if obj is None:
            return
        else:
            self.pid=getattr(obj,'pid',None)

    def __reduce__(self):
        '''
        numpy.ndarray uses __reduce__ to pickle. Therefore this method needs overriding for subclasses.
        '''
        data=super(Point,self).__reduce__()
        return data[0],data[1],data[2]+(self.pid,)

    def __setstate__(self,state):
        '''
        Set the state of the Point for pickle and copy.
        '''
        self.pid=state[-1]
        super(Point,self).__setstate__(state[0:-1])

    @property
    def rcoord(self):
        '''
        The coordinate in real space.
        '''
        return np.asarray(self)[0,:]

    @rcoord.setter
    def rcoord(self,value):
        '''
        Set the rcoord of a point.
        '''
        self[0,:]=value

    @property
    def icoord(self):
        '''
        The coordinate in lattice space.
        '''
        return np.asarray(self)[1,:]

    @icoord.setter
    def icoord(self,value):
        '''
        Set the icoord of a point.
        '''
        self[1,:]=value

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
        return self.pid==other.pid and nl.norm(self-other)<RZERO
    
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
    neighbour : integer
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

def intralinks(mode='nb',cluster=(),indices=None,vectors=(),**options):
    '''
    This function searches the wanted links intra a cluster.

    Parameters
    ----------
    mode : 'nb' or 'dt'
        * When 'nb', the function searches the links within a certain order of nearest neighbour;
        * When 'dt', the function searches the links within a certain distance.
    cluster : list of 1d ndarray
        The cluster where the links are searched.
    vectors : list of 1d ndarray, optional
        The translation vectors of the cluster.
    indices : list of integer, optional
        The indices of the points of the cluster.
    options : dict
        When mode is 'nb', it contains:
            * `nneighbour`: integer, optional, default 1
                The highest order of neighbour to be searched.
            * `max_coordinate_number`: integer, optional, default 8
                The max coordinate number for every neighbour.
            * `return_mindists`: logical, optional, default False
                When True, the nneighbour minimum distances will also be returned.
        When mode is 'dt', it contains:
            * `r`: float64, optional, default 1.0
                The distance upper bound within which the links are searched.
            * `max_translations`: tuple, optional
                The maximum translations of the original cluster, which will be omitted if ``len(vectors)==0``.
                When `r` is properly set, its default value can make sure that all the required links will be searched and returned.
            * `mindists`: list of float64, optional, default empty list
                The distances of the lowest orders of nearest neighbours.
                If it doesn't contain the distance of the returned link, the attribute `neighbour` of the latter will be set to be inf.

    Returns
    -------
    result: list of Link
        The calculated links.
    mindists: list of float64, optional
        The `nneighbour`-th minimum distances within the cluster.
        It will be returned only when ``mode=='nb' and options['return_mindists']==True``.

    Notes
    -----
        * The zero-th neighbour links i.e. links with distances equal to zero are also included in `result`.
        * When `vectors` **NOT** empty, periodic boundary condition is assumed and the links across the boundaries of the cluster are also searched.
    '''
    assert mode in ('nb','dt')
    if mode=='nb':
        return __links_nb__(
            cluster=                cluster,
            indices=                indices,
            vectors=                vectors,
            nneighbour=             options.get('nneighbour',1),
            max_coordinate_number=  options.get('max_coordinate_number',8),
            return_mindists=        options.get('return_mindists',False)
            )
    elif mode=='dt':
        return __links_dt__(
            cluster=            cluster,
            indices=            indices,
            vectors=            vectors,
            r=                  options.get('r',1.0),
            max_translations=   options.get('max_translations',tuple([int(np.ceil(options.get('r',1.0)/nl.norm(vector))) for vector in vectors])),
            mindists=           options.get('mindists',[])
            )

def __links_nb__(cluster,indices,vectors,nneighbour,max_coordinate_number,return_mindists):
    '''
    For details, see intralinks.
    '''
    result,mindists=[],[]
    if len(cluster)>0:
        translations=list(it.product(*([xrange(-nneighbour,nneighbour+1)]*len(vectors))))
        for translation in translations:
            if any(translation): translations.remove(tuple([-i for i in translation]))
        translations=sorted(translations,key=nl.norm)
        supercluster=tiling(cluster,vectors=vectors,translations=translations)
        disps=tiling([np.zeros(len(next(iter(cluster))))]*len(cluster),vectors=vectors,translations=translations)
        tree=cKDTree(supercluster)
        distances,eseqses=tree.query(cluster,k=nneighbour*max_coordinate_number if nneighbour>0 else 2)
        mindists=[np.inf]*(nneighbour+1)
        for dist in np.concatenate(distances):
            for i,mindist in enumerate(mindists):
                if abs(dist-mindist)<RZERO:
                    break
                elif dist<mindist:
                    if nneighbour>0: mindists[i+1:nneighbour+1]=mindists[i:nneighbour]
                    mindists[i]=dist
                    break
        mindists=[mindist for mindist in mindists if mindist!=np.inf]
        for sseq,(dists,eseqs) in enumerate(zip(distances,eseqses)):
            if dists[-1]<mindists[-1] or abs(dists[-1]-mindists[-1])<RZERO:
                raise ValueError("Function _links_nb_ error: the max_coordinate_number(%s) should be larger."%max_coordinate_number)
            for dist,eseq in zip(dists,eseqs):
                if sseq<=eseq:
                    for neighbour,mindist in enumerate(mindists):
                        if abs(dist-mindist)<RZERO:
                            sindex=sseq if indices is None else indices[sseq]
                            eindex=eseq%len(cluster) if indices is None else indices[eseq%len(cluster)]
                            result.append(Link(neighbour,sindex=sindex,eindex=eindex,disp=disps[eseq]))
    if return_mindists:
        return result,mindists
    else:
        return result

def __links_dt__(cluster,indices,vectors,r,max_translations,mindists):
    '''
    For details, see intralinks.
    '''
    result=[]
    if len(cluster)>0:
        translations=list(it.product(*[xrange(-nnb,nnb+1) for nnb in max_translations]))
        for translation in translations:
            if any(translation):translations.remove(tuple([-i for i in translation]))
        translations=sorted(translations,key=nl.norm)
        supercluster=tiling(cluster,vectors=vectors,translations=translations)
        disps=tiling([np.zeros(len(next(iter(cluster))))]*len(cluster),vectors=vectors,translations=translations)
        tree,other=cKDTree(cluster),cKDTree(supercluster)
        smatrix=tree.sparse_distance_matrix(other,r)
        for (i,j),dist in smatrix.items():
            if i<=j:
                for k,mindist in enumerate(mindists):
                    if abs(mindist-dist)<RZERO: 
                        neighbour=k
                        break
                else:
                    neighbour=np.inf
                sindex=i if indices is None else indices[i]
                eindex=j%len(cluster) if indices is None else indices[j%len(cluster)]
                result.append(Link(neighbour,sindex=sindex,eindex=eindex,disp=disps[j]))
    return result

def interlinks(cluster1,cluster2,maxdist,indices1=None,indices2=None,mindists=()):
    '''
    This function searches the links between two clusters with the distances less than a certain value.

    Parameters
    ----------
    cluster1, cluster2 : list of 1d ndarray
        The clusters.
    maxdist : float64
        The maximum distance.
    indices1, indices2 : list of integer, optional
        The indices of the points of the clusters.
    mindists : list of float64, optional
        The values of the distances between minimum neighbours.

    Returns
    -------
    list of Link
        The wanted links.
    '''
    result=[]
    if len(cluster1)>0 and len(cluster2)>0:
        tree1,tree2=cKDTree(cluster1),cKDTree(cluster2)
        smatrix=tree1.sparse_distance_matrix(tree2,maxdist)
        for (i,j),dist in smatrix.items():
            for k,mindist in enumerate(mindists):
                if abs(mindist-dist)<RZERO:
                    neighbour=k
                    break
            else:
                neighbour=np.inf
            sindex=i if indices1 is None else indices1[i]
            eindex=j if indices2 is None else indices2[j]
            result.append(Link(neighbour,sindex=sindex,eindex=eindex,disp=0))
    return result

class Lattice(object):
    '''
    This class provides a unified description of 1d, quasi 1d, 2D, quasi 2D and 3D lattice systems.

    Attributes
    ----------
    name : string
        The lattice's name.
    points : list of Point
        The points of the lattice.
    vectors : list of 1d ndarray
        The translation vectors.
    reciprocals : list of 1d ndarray
        The dual translation vectors.
    nneighbour : integer
        The highest order of neighbours;
    links : list of Link
        The links of the lattice system.
    mindists : list of float
        The minimum distances within this lattice.
    max_coordinate_number : int
        The max coordinate number for every neighbour.
        This attribute is used in the search for links.
    '''
    max_coordinate_number=8

    def __init__(self,name=None,rcoords=(),icoords=None,vectors=(),nneighbour=1,max_coordinate_number=None):
        '''
        Construct a lattice directly from its coordinates.

        Parameters
        ----------
        name : string
            The name of the lattice.
        rcoords : list of 1d ndarray, optional
            The rcoords of the lattice.
        icoords : list of 1d ndarray, optional
            The icoords of the lattice.
        vectors : list of 1d ndarray, optional
            The translation vectors of the lattice.
        nneighbour : integer, optional
            The highest order of neighbours.
        max_coordinate_number : int, optional 
            The max coordinate number for every neighbour.
        '''
        assert icoords is None or len(icoords)==len(rcoords)
        if max_coordinate_number is not None: Lattice.max_coordinate_number=max_coordinate_number
        self.name=name
        rcoords=np.asarray(rcoords)
        icoords=np.zeros(rcoords.shape) if icoords is None else np.asarray(icoords)
        self.points=[Point(PID(scope=name,site=i),rcoord=rcoord,icoord=icoord) for i,(rcoord,icoord) in enumerate(zip(rcoords,icoords))]
        self.vectors=vectors
        self.reciprocals=reciprocals(vectors)
        self.nneighbour=nneighbour
        links,mindists=intralinks('nb',rcoords,None,vectors,nneighbour=nneighbour,max_coordinate_number=Lattice.max_coordinate_number,return_mindists=True)
        self.links=links
        self.mindists=mindists

    @classmethod
    def compose(cls,name=None,points=(),vectors=(),nneighbour=1,max_coordinate_number=None):
        '''
        Construct a lattice from its contained points.

        Parameters
        ----------
        name : string
            The name of the lattice.
        points : list of Point, optional
            The lattice points.
        vectors : list of 1d ndarray, optional
            The translation vectors of the lattice.
        nneighbour : integer, optional
            The highest order of neighbours.
        max_coordinate_number : int, optional
            The max coordinate number for every neighbour.
        '''
        if max_coordinate_number is not None: Lattice.max_coordinate_number=max_coordinate_number
        result=object.__new__(cls)
        result.name=name
        result.points=points
        result.vectors=vectors
        result.reciprocals=reciprocals(vectors)
        result.nneighbour=nneighbour
        rcoords=np.asarray([point.rcoord for point in points])
        links,mindists=intralinks('nb',rcoords,None,vectors,nneighbour=nneighbour,max_coordinate_number=Lattice.max_coordinate_number,return_mindists=True)
        result.links=links
        result.mindists=mindists
        return result

    def __len__(self):
        '''
        The number of points contained in this lattice.
        '''
        return len(self.points)

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return '\n'.join([str(link) for link in self.links])

    @property
    def pids(self):
        '''
        The pids of the lattice.
        '''
        return [point.pid for point in self.points]

    @property
    def rcoords(self):
        '''
        The rcoords of the lattice.
        '''
        return np.asarray([point.rcoord for point in self.points])

    @property
    def icoords(self):
        '''
        The icoords of the lattice.
        '''
        return np.asarray([point.icoord for point in self.points])

    @property
    def bonds(self):
        '''
        The bonds of the lattice.
        '''
        result=[]
        for link in self.links:
            spoint=self.points[link.sindex]
            epoint=self.points[link.eindex]+link.disp
            result.append(Bond(link.neighbour,spoint,epoint))
        return result

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
        return self.points[self.pids.index(pid)].rcoord

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
        return self.points[self.pids.index(pid)].icoord

    def translate(self,vector):
        '''
        Translate the whole lattice by a vector.

        Parameters
        ----------
        vector : 1d ndarray
            The translation vector.
        '''
        for point in self.points:
            point.rcoord+=vector

    def rename(self,name=None,pids=None):
        '''
        Rename the lattice and its points.

        Parameters
        ----------
        name : string, optional
            The new name of the lattice.
        pids : list of PID, optional
            The new pids of the points of the lattice.
        '''
        if name is not None: self.name=name
        if pids is not None:
            assert len(pids)==len(self)
            for pid,point in zip(pids,self.points):
                point.pid=pid

    def plot(self,fig=None,ax=None,show=True,suspend=False,save=True,close=True,pidon=False):
        '''
        Plot the lattice points and bonds. Only 2D or quasi 1d systems are supported.
        '''
        if fig is None or ax is None: fig,ax=plt.subplots()
        ax.axis('off')
        ax.axis('equal')
        ax.set_title(self.name)
        for bond in self.bonds:
            nb=bond.neighbour
            if nb<0: nb=self.nneighbour+2
            elif nb==np.inf: nb=self.nneighbour+1
            if nb==1: color='k'
            elif nb==2: color='r'
            elif nb==3: color='b'
            else: color=str(nb*1.0/(self.nneighbour+1))
            if nb==0:
                x,y=bond.spoint.rcoord[0],bond.spoint.rcoord[1]
                ax.scatter(x,y)
                if pidon:
                    pid=bond.spoint.pid
                    if pid.scope is None:
                        tag=str(pid.site)
                    else:
                        tag=str(pid.scope)+'*'+str(pid.site)
                    ax.text(x,y,tag,fontsize=10,color='blue',horizontalalignment='center')
            else:
                if bond.isintracell():
                    ax.plot([bond.spoint.rcoord[0],bond.epoint.rcoord[0]],[bond.spoint.rcoord[1],bond.epoint.rcoord[1]],color=color)
                else:
                    ax.plot([bond.spoint.rcoord[0],bond.epoint.rcoord[0]],[bond.spoint.rcoord[1],bond.epoint.rcoord[1]],color=color,ls='--')
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
    _merge_ : list of string
        The names of sublattices that are merged to form new links.
    _union_ : list of 2-tuple
        The pairs of names of sublattices that are united to form new links.
    '''

    def __init__(self,name,sublattices,vectors=(),nneighbour=1,merge=None,union=None,mindists=None,maxdist=None,max_coordinate_number=None):
        '''
        Constructor.

        Parameters
        ----------
        name : string
            The name of the superlattice.
        sublattices : list of Lattice
            The sublattices of the superlattice.
        vectors : list of 1d ndarray, optional
            The translation vectors of the superlattice.
        nneighbour : integer, optional
            The highest order of neighbours.
        merge : list of integer, optional
            The indices of the sublattices that are merged to form new links.
        union : list of 2-tuple of integer, optional
            The pairs of indices of the sublattices that are united to form new links.
        mindists : list of float64, optional
            The values of the distances between minimum neighbours.
        maxdist : float64, optional
            The maximum distance.
        max_coordinate_number : int, optional
            The max coordinate number for every neighbour.
        '''
        if max_coordinate_number is not None: Lattice.max_coordinate_number=max_coordinate_number
        self.name=name
        self.sublattices=sublattices
        self.points=[point for lattice in sublattices for point in lattice.points]
        self.vectors=vectors
        self.reciprocals=reciprocals(vectors)
        self._merge_=[] if merge is None else merge
        if len(self._merge_)>0:
            self.nneighbour=nneighbour
            links,mindists=intralinks(
                    mode=                   'nb',
                    cluster=                np.concatenate([self.sublattices[name].rcoords for name in self._merge_]),
                    vectors=                vectors,
                    nneighbour=             nneighbour,
                    max_coordinate_number=  Lattice.max_coordinate_number,
                    return_mindists=        True
                    )
            self.links=links
            self.mindists=mindists
        else:
            assert mindists is not None
            self.links=[]
            self.mindists=mindists
            self.nneighbour=len(mindists)-1
        if union is None:
            uns=sorted(set(xrange(len(self.sublattices)))-set(self._merge_))
            self._union_=[(n1,n2) for n1 in uns for n2 in uns if n1<n2]
        else:
            uns=sorted(set(np.concatenate(union)))
            self._union_=union
        indiceses={n:[self.pids.index(pid) for pid in self.sublattices[n].pids] for n in uns}
        for n1,n2 in self._union_:
            self.links.extend(interlinks(
                    cluster1=       self.sublattices[n1].rcoords,
                    cluster2=       self.sublattices[n2].rcoords,
                    maxdist=        mindists[-1]+RZERO if maxdist is None else maxdist,
                    indices1=       indiceses[n1],
                    indices2=       indiceses[n2],
                    mindists=       mindists
                    ))

    @staticmethod
    def merge(name,sublattices,vectors=(),nneighbour=1,max_coordinate_number=None):
        '''
        This is a simplified version of SuperLattice.__init__ by just merging sublattices to construct the superlattice.
        For details, see SuperLattice.__init__.
        '''
        return SuperLattice(
            name=                       name,
            sublattices=                sublattices,
            vectors=                    vectors,
            merge=                      range(len(sublattices)),
            nneighbour=                 nneighbour,
            max_coordinate_number=      max_coordinate_number
            )

    @staticmethod
    def union(name,sublattices,mindists,vectors=(),union=None,maxdist=None):
        '''
        This is a simplified version of SuperLattice.__init__ by just uniting sublattices to construct the superlattice.
        For details, see SuperLattice.__init__.
        '''
        return SuperLattice(
            name=               name,
            sublattices=        sublattices,
            vectors=            vectors,
            union=              union,
            nneighbour=         len(mindists)-1,
            maxdist=            maxdist,
            mindists=           mindists
            )

    @property
    def bonds(self):
        '''
        The bonds of the superlattice.
        '''
        result=super(SuperLattice,self).bonds
        for n in set(xrange(len(self.sublattices)))-set(self._merge_):
            result.extend(self.sublattices[n].bonds)
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
        if len(self)==0:
            aps=[Point(PID(scope=A,site=i),rcoord=rcoord-self.translation/2,icoord=np.zeros_like(rcoord)) for i,rcoord in enumerate(self.block)]
            bps=[Point(PID(scope=B,site=i),rcoord=rcoord+self.translation/2,icoord=np.zeros_like(rcoord)) for i,rcoord in enumerate(self.block)]
            ass,bss=[],[]
        else:
            if news is not None:
                assert len(news)*len(self.block)==len(self)
                for i,scope in enumerate(news):
                    for j in xrange(len(self.block)):
                        self.points[i*len(self.block)+j].pid=self.points[i*len(self.block)+j].pid._replace(scope=scope)
            aps,bps=self.points[:len(self)/2],self.points[len(self)/2:]
            for ap,bp in zip(aps,bps):
                ap.rcoord-=self.translation
                bp.rcoord+=self.translation
            ass=[Point(PID(scope=A,site=i),rcoord=rcoord-self.translation/2,icoord=np.zeros_like(rcoord)) for i,rcoord in enumerate(self.block)]
            bss=[Point(PID(scope=B,site=i),rcoord=rcoord+self.translation/2,icoord=np.zeros_like(rcoord)) for i,rcoord in enumerate(self.block)]
        self.points=aps+ass+bss+bps
        links,mindists=intralinks(
                mode=                   'nb',
                cluster=                np.asarray([p.rcoord for p in self.points]),
                indices=                None,
                vectors=                self.vectors,
                nneighbour=             self.nneighbour,
                max_coordinate_number=  self.max_coordinate_number,
                return_mindists=        True
                )
        self.links=links
        self.mindists=mindists

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
        points,num=[],len(scopes)
        for i,rcoord in enumerate(tiling(self.block,[self.translation],np.linspace(-(num-1)/2.0,(num-1)/2.0,num) if num>1 else xrange(1))):
            points.append(Point(PID(scope=scopes[i/len(self.block)],site=i%len(self.block)),rcoord=rcoord,icoord=np.zeros_like(rcoord)))
        return Lattice.compose(name=self.name.replace('+',str(num)),points=points,vectors=self.vectors,nneighbour=self.nneighbour)
