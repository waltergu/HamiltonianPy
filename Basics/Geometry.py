'''
Geometry, including
1) functions: azimuthd, azimuth, polard, polar, volume, is_parallel, reciprocals, tiling, translation, rotation, bonds, SuperLattice
2) classes: PID, Point, Bond, Lattice
'''

__all__=['azimuthd','azimuth','polard','polar','volume','is_parallel','reciprocals','tiling','translation','rotation','bonds','SuperLattice','PID','Point','Bond','Lattice']

from numpy import *
from numpy.linalg import norm,inv
from Constant import RZERO
from collections import namedtuple
from scipy.spatial import cKDTree
from copy import copy,deepcopy
import matplotlib.pyplot as plt
import itertools

def azimuthd(self):
    '''
    Azimuth in degrees of an array-like vector.
    '''
    if self[1]>=0:
        return degrees(arccos(self[0]/norm(self)))
    else:
        return 360-degrees(arccos(self[0]/norm(self)))

def azimuth(self):
    '''
    Azimuth in radians of an array-like vector.
    '''
    if self[1]>=0:
        return arccos(self[0]/norm(self))
    else:
        return 2*pi-arccos(self[0]/norm(self))

def polard(self):
    '''
    Polar angle in degrees of an array-like vector.
    '''
    if self.shape[0]==3:
        return degrees(arccos(self[2]/norm(self)))
    else:
        raise ValueError("PolarD error: the array-like vector must contain three elements.")

def polar(self):
    '''
    Polar angle in radians of an array-like vector.
    '''
    if self.shape[0]==3:
        return arccos(self[2]/norm(self))
    else:
        raise ValueError("Polar error: the array-like vector must contain three elements.")

def volume(O1,O2,O3):
    '''
    Volume spanned by three array-like vectors.
    '''
    if O1.shape[0] in [1,2] or O2.shape[0] in [1,2] or O3.shape[0] in [1,2]:
        return 0
    elif O1.shape[0] ==3 and O2.shape[0]==3 and O3.shape[0]==3:
        return inner(O1,cross(O2,O3))
    else:
        raise ValueError("Volume error: the shape of the array-like vectors is not supported.")

def is_parallel(O1,O2):
    '''
    Judge whether two array-like vectors are parallel to each other.
    Parameters:
        O1,O2: 1D array-like
    Returns: int
         0: not parallel,
         1: parallel, and 
        -1: anti-parallel.
    '''
    norm1=norm(O1)
    norm2=norm(O2)
    if norm1<RZERO or norm2<RZERO:
        return 1
    elif O1.shape[0]==O2.shape[0]:
        buff=inner(O1,O2)/(norm1*norm2)
        if abs(buff-1)<RZERO:
            return 1
        elif abs(buff+1)<RZERO:
            return -1
        else:
            return 0
    else:
        raise ValueError("Is_parallel error: the shape of the array-like vectors does not match.") 

def reciprocals(vectors):
    '''
    Return the corresponding reciprocals dual to the input vectors.
    Parameters:
        vectors: 2D array-like
    Returns: 2D array-like
        The reciprocals.
    '''
    result=[]
    nvectors=len(vectors)
    if nvectors==0:
        return
    if nvectors==1:
        result.append(array(vectors[0]/(norm(vectors[0]))**2*2*pi))
    elif nvectors in (2,3):
        ndim=vectors[0].shape[0]
        buff=zeros((3,3))
        buff[0:ndim,0]=vectors[0]
        buff[0:ndim,1]=vectors[1]
        if nvectors==2:
            buff[(2 if ndim==2 else 0):3,2]=cross(vectors[0],vectors[1])
        else:
            buff[0:ndim,2]=vectors[2]
        buff=inv(buff)
        result.append(array(buff[0,0:ndim]*2*pi))
        result.append(array(buff[1,0:ndim]*2*pi))
        if nvectors==3:
            result.append(array(buff[2,0:ndim]*2*pi))
    else:
        raise ValueError('Reciprocals error: the number of translation vectors should not be greater than 3.')
    return result

class PID(namedtuple('PID',['scope','site'])):
    '''
    The ID of a point.
    Attributes:
        scope: any hashable object, recommend string
            The scope in which the point lives.
            Usually, it is same to the name of the cluster/sublattice/lattice the point belongs to.
        site: any hashable object, recommend tuple of int or int
            The site index of the point.
    '''

PID.__new__.__defaults__=(None,)*len(PID._fields)

class Point:
    '''
    Point.
    Attributes:
        pid: PID
            The specific ID of a point.
        rcoord: 1D ndarray
            The coordinate in real space.
        icoord: 1D ndarray
            The coordinate in lattice space.
    '''

    def __init__(self,pid,rcoord=None,icoord=None):
        '''
        Constructor.
        Parameters:
            pid: PID
                The specific ID of a point
            rcoord: 1D array-like
                The coordinate in real space.
            icoord: 1D array-like,optional
                The coordinate in lattice space.
        '''
        if not isinstance(pid,PID):
            raise ValueError("Point constructor error: the 'pid' parameter must be an instance of PID.")
        self.pid=pid
        self.rcoord=array([]) if rcoord is None else array(rcoord)
        self.icoord=array([]) if icoord is None else array(icoord)

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        return 'Point(pid=%s, rcoord=%s, icoord=%s)'%(self.pid,self.rcoord,self.icoord)

    def __eq__(self,other):
        '''
        Overloaded operator(==).
        '''
        return self.pid==other.pid and norm(self.rcoord-other.rcoord)<RZERO and norm(self.icoord-other.icoord)<RZERO
    
    def __ne__(self,other):
        '''
        Overloaded operator(!=).
        '''
        return not self==other

def tiling(cluster,vectors=[],indices=[],flatten_site=True,return_map=False):
    '''
    Tile a supercluster by translations of the input cluster.
    Parameters:
        cluster: list of Point
            The original cluster.
        vectors: list of 1D ndarray, optional
            The translation vectors.
        indices: any iterable object of tuple, optional
            It iterates over the indices of the translated clusters in the tiled superlattice.
        flatten_site: logical, optional
            When it is True, the site attribute of the pid of the supercluster's points will be "flattened", i.e. it will be transformed form a tuple to an integer.
        return_map: logical, optional
            If it is set to be False, the tiling map will not be returned.
            Otherwise, the tiling map will be returned.
    Returns:
        supercluster: list of Point
            The supercluster tiled from the translations of the input cluster.
        map: dict,optional
            The tiling map, whose key is the translated point's pid and value the original point's pid.
            Only when return_map is set to be True, will it be returned.
    '''
    supercluster,map=[],{}
    if len(vectors)==0:
        for point in cluster:
            supercluster.append(point)
            map[point.pid]=point.pid
    else:
        for index in indices:
            for point in cluster:
                if isinstance(point.pid.site,int):
                    site=((index,) if isinstance(index,int) else tuple(index))+(point.pid.site,)
                else:
                    site=((index,) if isinstance(index,int) else tuple(index))+tuple(point.pid.site)
                new=point.pid._replace(site=site)
                map[new]=point.pid
                disp=dot(site[0:len(vectors)],vectors)
                supercluster.append(Point(pid=new,rcoord=point.rcoord+disp,icoord=point.icoord))
    if flatten_site:
        new_map={}
        supercluster.sort(key=lambda point: point.pid)
        for i,point in enumerate(supercluster):
            new=point.pid._replace(site=i)
            new_map[new]=map[point.pid]
            point.pid=new
        map=new_map
    if return_map:
        return supercluster,map
    else:
        return supercluster

def translation(cluster,vector):
    '''
    This function returns the translated cluster.
    Parameters:
        cluster: list of Point / list of 1D array-like
            The original cluster.
        vector: 1D ndarray
            The translation vector.
    Returns: list of Point / list of 1D array-like
        The translated cluster.
    '''
    if isinstance(cluster[0],Point):
        return [Point(pid=point.pid,rcoord=point.rcoord+vector,icoord=deepcopy(point.icoord)) for point in cluster]
    else:
        return [array(rcoord)+vector for rcoord in cluster]

def rotation(cluster,angle=0,axis=None,center=None):
    '''
    This function returns the rotated cluster.
    Parameters:
        cluster: list of Point / list of 1D array-like
            The original cluster.
        angle: float
            The rotated angle
        axis: 1D array-like, optional
            The rotation axis. Default the z-axis.
            Not supported yet.
        center: 1D array-like, optional
            The center of the axis. Defualt the origin.
    Returns: list of Point / list of 1D array-like
        The rotated points or coords.
    '''
    result=[]
    if center is None: center=0
    m11=cos(angle);m21=-sin(angle);m12=-m21;m22=m11
    m=array([[m11,m12],[m21,m22]])
    if isinstance(cluster[0],Point):
        return [Point(pid=point.pid,rcoord=dot(m,point.rcoord-center)+center,icoord=deepcopy(point.icoord)) for point in cluster]
    else:
        return [dot(m,rcoord-center)+center for rcoord in cluster]

class Bond:
    '''
    This class describes the bond in a lattice.
    Attributes:
        neighbour: integer
            The rank of the neighbour of the bond.
        spoint: Point
            The start point of the bond.
        epoint: Point
            The end point of the bond.
    '''
    
    def __init__(self,neighbour,spoint,epoint):
        '''
        Constructor.
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
    
    def is_intra_cell(self):
        '''
        Judge whether a bond is intra the unit cell or not. 
        '''
        if norm(self.icoord)< RZERO:
            return True
        else:
            return False

    @property
    def reversed(self):
        '''
        Return the reversed bond.
        '''
        return Bond(self.neighbour,self.epoint,self.spoint)

def bonds(cluster,vectors=[],nneighbour=1,max_coordinate_number=8):
    '''
    This function returns all the bonds up to the nneighbour-th order.
    Parameters:
        cluster: list of Point
            The cluster within which the bonds are looked for.
            If the parameter vectors is not None, the inter cluster bonds will also be searched.
        vectors: list of 1D ndarray, optional
            The translation vectors for the cluster.
        nneighbour: integer, optional
            The highest order of neighbour to be searched.
        max_coordinate_number: int, optional
            The max coordinate number for every neighbour.
    Returns:
        result: list of Bond
            All the bonds up to the nneighbour-th order.
            Note that the input points will be used to form the zero-th neighbour bonds, i.e. the start point and the end point is the same point.
    '''
    result=[]
    indices=[]
    for i in xrange(len(vectors)):
        indices.append(xrange(-nneighbour,nneighbour+1))
    indices=list(itertools.product(*indices))
    for index in indices:
        if any(index):indices.remove(tuple([-i for i in index]))
    supercluster,map=[],{}
    for index in indices:
        for point in cluster:
            if isinstance(point.pid.site,tuple):
                if any(index):
                    site=((index,) if isinstance(index,int) else tuple(index))+tuple(point.pid.site)
                else:
                    site=point.pid.site
            else:
                if any(index):
                    site=((index,) if isinstance(index,int) else tuple(index))+(point.pid.site,)
                else:
                    site=point.pid.site
            new=point.pid._replace(site=site)
            map[new]=point.pid
            disp=dot(index,vectors)
            supercluster.append(Point(pid=new,rcoord=point.rcoord+disp,icoord=point.icoord+disp))
    tree=cKDTree([point.rcoord for point in supercluster])
    distances,indices=tree.query([point.rcoord for point in cluster],k=nneighbour*max_coordinate_number)
    mdists=[inf for i in xrange(nneighbour+1)]
    for dist in concatenate(distances):
        for i,mdist in enumerate(mdists):
            if abs(dist-mdist)<RZERO:
                break
            elif dist<mdist:
                mdists[i+1:nneighbour+1]=mdists[i:nneighbour]
                mdists[i]=dist
                break
    max_mdists=mdists[nneighbour]
    for i,(dists,inds) in enumerate(zip(distances,indices)):
        max_dists=dists[nneighbour*max_coordinate_number-1]
        if max_dists<max_mdists or abs(max_dists-max_mdists)<RZERO:
            raise ValueError("Function bonds error: the max_coordinate_number(%s) should be larger."%max_coordinate_number)
        for dist,index in zip(dists,inds):
            for neighbour,mdist in enumerate(mdists):
                if abs(dist-mdist)<RZERO:
                    buff=supercluster[index]
                    if buff.pid!=map[buff.pid] or cluster[i].pid<=buff.pid:
                        result.append(Bond(neighbour,spoint=cluster[i],epoint=Point(pid=map[buff.pid],rcoord=buff.rcoord,icoord=buff.icoord)))
    return result

class Lattice(object):
    '''
    This class provides a unified description of 1D, quasi 1D, 2D, quasi 2D and 3D lattice systems.
    Attributes:
        name: string
            The lattice's name.
        points: dict of Point
            The lattice points in a unit cell.
        vectors: list of 1D ndarray
            The translation vectors.
        reciprocals: list of 1D ndarray
            The dual translation vectors.
        nneighbour: integer
            The highest order of neighbours;
        bonds: list of Bond
            The bonds of the lattice system.
        max_coordinate_number: int
            The max coordinate number for every neighbour.
    '''

    def __init__(self,name,points,vectors=[],nneighbour=1,max_coordinate_number=8):
        '''
        Constructor.
        Parameters:
            name: string
                The name of the lattice.
            points: list of Point
                The lattice points in a unit cell.
            vectors: list of 1D ndarray, optional
                The translation vectors of the lattice.
            nneighbour: integer, optional
                The highest order of neighbours.
            max_coordinate_number: int, optional
                The max coordinate number for every neighbour.
                This variable is used in the search for bonds.
        '''
        self.name=name
        self.points={}
        for point in copy(points):
            point.pid=point.pid._replace(scope=name)
            self.points[point.pid]=point
        self.vectors=vectors
        self.reciprocals=reciprocals(self.vectors)
        self.nneighbour=nneighbour
        self.max_coordinate_number=max_coordinate_number
        self.bonds=bonds(points,vectors,nneighbour,max_coordinate_number)

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return '\n'.join([str(bond) for bond in self.bonds])

    def plot(self,show=True,pid_on=False):
        '''
        Plot the lattice points and bonds. Only 2D or quasi 1D systems are supported.
        '''
        plt.axes(frameon=0)
        plt.axis('equal')
        plt.title(self.name)
        for bond in self.bonds:
            nb=bond.neighbour
            if nb==1: color='k'
            elif nb==2: color='r'
            elif nb==3: color='b'
            else: color=str(nb*1.0/self.nneighbour)
            if nb==0:
                x,y=bond.spoint.rcoord[0],bond.spoint.rcoord[1]
                plt.scatter(x,y)
                if pid_on:
                    pid=bond.spoint.pid
                    if pid.scope==None:
                        tag=str(pid.site)
                    else:
                        tag=str(pid.scope)+'*'+str(pid.site)
                    plt.text(x-0.2,y+0.1,tag,fontsize=10,color='blue')
            else:
                if bond.is_intra_cell():
                    plt.plot([bond.spoint.rcoord[0],bond.epoint.rcoord[0]],[bond.spoint.rcoord[1],bond.epoint.rcoord[1]],color=color)
                else:
                    plt.plot([bond.spoint.rcoord[0],bond.epoint.rcoord[0]],[bond.spoint.rcoord[1],bond.epoint.rcoord[1]],color=color,ls='--')
        frame=plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        if show:
            plt.show()
        else:
            plt.savefig(self.name+'.png')
        plt.close()

def SuperLattice(name,sublattices,vectors=[],nneighbour=1):
    '''
    This function returns the union of sublattices.
    Parameters:
        name: string
            The name of the super-lattice.
        sublattices: list of Lattice
            The sub-lattices of the super-lattice.
        vectors: list of 1D ndarray, optional
            The translation vectors of the super-lattice.
        nneighbour: integer,optional
            The highest order of neighbours.
    Returns: Lattice
        The super-lattice.
    '''
    result=object.__new__(Lattice)
    result.name=name
    result.points={}
    for lattice in sublattices:
        result.points.update(lattice.points)
    result.vectors=vectors
    result.reciprocals=reciprocals(vectors)
    result.nneighbour=nneighbour
    result.bonds=bonds(cluster=result.points.values(),vectors=vectors,nneighbour=nneighbour)
    result.sublattices=sublattices
    return result
