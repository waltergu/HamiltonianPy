'''
Geometry, including
1) functions: azimuthd, azimuth, polard, polar, volume, is_parallel, reciprocals, tiling, translation, rotation, bonds, bonds_between_clusters
2) classes: PID, Point, Bond, Lattice, SuperLattice
'''

__all__=['azimuthd','azimuth','polard','polar','volume','is_parallel','reciprocals','tiling','translation','rotation','bonds','bonds_between_clusters','SuperLattice','PID','Point','Bond','Lattice']

from numpy import *
from numpy.linalg import norm,inv
from Constant import RZERO
from collections import namedtuple,OrderedDict
from scipy.spatial import cKDTree
from copy import copy,deepcopy
import matplotlib.pyplot as plt
import itertools
import warnings

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
        scope: string
            The scope in which the point lives.
            Usually, it is same to the name of the cluster/sublattice/lattice the point belongs to.
        site: integer
            The site index of the point.
    '''

PID.__new__.__defaults__=(None,)*len(PID._fields)

class Point(ndarray):
    '''
    Point.
    Attributes:
        pid: PID
            The specific ID of a point.
    '''

    def __new__(cls,pid,rcoord=None,icoord=None):
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
        result=asarray([[] if rcoord is None else rcoord, [] if icoord is None else icoord]).view(cls)
        result.pid=pid
        return result

    def __array_finalize__(self,obj):
        '''
        Initialize an instance through both explicit and implicit constructions, i.e. construtor, view and slice.
        '''
        if obj is None:
            return
        else:
            self.pid=getattr(obj,'pid',None)

    @property
    def rcoord(self):
        '''
        The coordinate in real space.
        '''
        return asarray(self)[0,:]

    @property
    def icoord(self):
        '''
        The coordinate in lattice space.
        '''
        return asarray(self)[1,:]

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return 'Point(pid=%s, rcoord=%s, icoord=%s)'%(self.pid,self.rcoord,self.icoord)

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        return 'Point(pid=%s, rcoord=%s, icoord=%s)'%(self.pid,self.rcoord,self.icoord)

    def __eq__(self,other):
        '''
        Overloaded operator(==).
        '''
        return self.pid==other.pid and norm(self-other)<RZERO
    
    def __ne__(self,other):
        '''
        Overloaded operator(!=).
        '''
        return not self==other

def tiling(cluster,vectors=[],indices=[],return_map=False):
    '''
    Tile a supercluster by translations of the input cluster.
    Parameters:
        cluster: list of Point
            The original cluster.
        vectors: list of 1D ndarray, optional
            The translation vectors.
        indices: any iterable object of tuple, optional
            It iterates over the indices of the translated clusters in the tiled superlattice.
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
        cmax=max([point.pid.site for point in cluster])+1
        for i,index in enumerate(indices):
            for point in cluster:
                new=point.pid._replace(site=point.pid.site+i*cmax)
                map[new]=point.pid
                disp=dot((index,) if (isinstance(index,int) or isinstance(index,long)) else tuple(index),vectors)
                supercluster.append(Point(pid=new,rcoord=point.rcoord+disp,icoord=point.icoord))
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

def bonds(cluster,vectors=[],mode='nb',options={}):
    '''
    This function calculates the bonds within a cluster.
    When vector is not empty, periodic boundary condition is assumed and the bonds across the boundaries of the cluster are also included.
    Parameters:
        cluster: list of Point
            The cluster within which the bonds are looked for.
        vectors: list of 1D ndarray, optional
            The translation vectors of the cluster.
        mode: 'nb' or 'dt'
            When 'nb', the function calculates all the bonds within a certain order of nearest neighbour;
            When 'dt', the function calculates all the bonds within a certain distance.
        options: dict
            The extra controlling parameters for either mode.
            When mode is 'nb', it contains:
                'nneighbour': integer, optional, default 1
                    The highest order of neighbour to be searched.
                'max_coordinate_number': integer, optional, default 8
                    The max coordinate number for every neighbour.
                'return_mdists': logical, optional, default False
                    When it is True, the nneighbour minimum distances will alse be returned.
            When mode is 'dt', it contains:
                'r': float64, optional, default 1.0
                    The distance upper bound within which the bonds are searched.
                'mdists': list of float64, optional, default empty list
                    The distances of the lowest orders of nearest neighbours.
                    If it doesn't contain the distance of the returned bond, the attribute 'neighbour' of the latter will be set to be inf.
    Returns:
        For both modes:
            result: list of Bond
                The calculated bonds.
                <NOTE> The zero-th neighbour bonds i.e. bonds with distances equal to zero are also included.
        For mode 'nb' only:
            mdists: list of float64, optional
                The nneighbour-th minimum distances within the cluster.
                It will be returned only when options['return_mdists']==True.
    '''
    if mode=='nb':
        return _bonds_nb_(
            cluster=                cluster,
            vectors=                vectors,
            nneighbour=             options.get('nneighbour',1),
            max_coordinate_number=  options.get('max_coordinate_number',8),
            return_mdists=          options.get('return_mdists',False)
            )
    elif mode=='dt':
        return _bonds_dt_(
            cluster=    cluster,
            vectors=    vectors,
            r=          options.get('r',1.0),
            mdists=     options.get('mdists',[])
            )
    else:
        raise ValueError("Function bonds error: mode(%s) not supported."%(mode))

def _bonds_nb_(cluster,vectors=[],nneighbour=1,max_coordinate_number=8,return_mdists=False):
    '''
    For details, see bonds.
    '''
    result=[]
    indices=[]
    for i in xrange(len(vectors)):
        indices.append(xrange(-nneighbour,nneighbour+1))
    indices=list(itertools.product(*indices))
    for index in indices:
        if any(index):indices.remove(tuple([-i for i in index]))
    supercluster,map=[],{}
    cmax=max([point.pid.site for point in cluster])+1
    for i,index in enumerate(indices):
        for point in cluster:
            new=point.pid._replace(site=point.pid.site+i*cmax if any(index) else point.pid.site)
            map[new]=point.pid
            disp=dot(index,vectors)
            supercluster.append(Point(pid=new,rcoord=point.rcoord+disp,icoord=point.icoord+disp))
    tree=cKDTree([point.rcoord for point in supercluster])
    distances,indices=tree.query([point.rcoord for point in cluster],k=nneighbour*max_coordinate_number)
    mdists=[inf for i in xrange(nneighbour+1)]
    for dist in concatenate(distances):
        for i,mdist in enumerate(mdists):
            if dist==mdist or abs(dist-mdist)<RZERO:
                break
            elif dist<mdist:
                mdists[i+1:nneighbour+1]=mdists[i:nneighbour]
                mdists[i]=dist
                break
    mdists=[mdist for mdist in mdists if mdist!=inf]
    max_mdist=max(mdists)
    for i,(dists,inds) in enumerate(zip(distances,indices)):
        max_dist=dists[nneighbour*max_coordinate_number-1]
        if max_dist<max_mdist or abs(max_dist-max_mdist)<RZERO:
            raise ValueError("Function _bonds_nb_ error: the max_coordinate_number(%s) should be larger."%max_coordinate_number)
        for dist,index in zip(dists,inds):
            for neighbour,mdist in enumerate(mdists):
                if abs(dist-mdist)<RZERO:
                    buff=supercluster[index]
                    if buff.pid!=map[buff.pid] or cluster[i].pid<=buff.pid:
                        result.append(Bond(neighbour,spoint=cluster[i],epoint=Point(pid=map[buff.pid],rcoord=buff.rcoord,icoord=buff.icoord)))
    if return_mdists:
        return result,mdists
    else:
        return result

def _bonds_dt_(cluster,vectors=[],r=1.0,mdists=[]):
    '''
    For details, see bonds.
    '''
    pass

def bonds_between_clusters(cluster1,cluster2,max_distance,mdists=[]):
    '''
    This function returns all the bonds between two clusters with the distances less than max_distance.
    Parameters:
        cluster1,cluster2: list of Point
            The clusters.
        max_distance: float64
            The maximum distance.
        mdist: list of float64, optional
            The values of the distances between minimum neighbours.
    Returns: list of Bond
        The bonds between cluster1 and cluster2 with the distances less than max_distance.
    '''
    tree1=cKDTree([point.rcoord for point in cluster1])
    tree2=cKDTree([point.rcoord for point in cluster2])
    smatrix=tree1.sparse_distance_matrix(tree2,max_distance)
    mdists=[] if mdists is None else mdists
    for i,j,dist in enumerate(smatrix):
        for k,mdist in enumerate(mdists):
            if abs(mdist-dist)<RZERO: 
                neighbour=k
                break
        else:
            neighbour=inf
        result.append(Bond(neighbour,spoint=cluster1[i],epoint=cluster2[j]))
    return result

class Lattice(dict):
    '''
    This class provides a unified description of 1D, quasi 1D, 2D, quasi 2D and 3D lattice systems.
    Attributes:
        name: string
            The lattice's name.
        vectors: list of 1D ndarray
            The translation vectors.
        reciprocals: list of 1D ndarray
            The dual translation vectors.
        nneighbour: integer
            The highest order of neighbours;
        bonds: list of Bond
            The bonds of the lattice system.
        mdists: list of float
            The minimum distances within this lattice.
        max_coordinate_number: int
            The max coordinate number for every neighbour.
    '''
    max_coordinate_number=8

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
        Lattice.max_coordinate_number=max_coordinate_number
        self.name=name
        for point in copy(points):
            point.pid=point.pid._replace(scope=name)
            self[point.pid]=point
        self.vectors=vectors
        self.reciprocals=reciprocals(self.vectors)
        self.nneighbour=nneighbour
        self.bonds,self.mdists=bonds(
                cluster=    points,
                vectors=    vectors,
                options={   'nneighbour':nneighbour,
                            'max_coordinate_number':max_coordinate_number,
                            'return_mdists':True
                        }
        )

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return '\n'.join([str(bond) for bond in self.bonds])

    def plot(self,fig=None,ax=None,show=True,save=False,close=True,pid_on=False):
        '''
        Plot the lattice points and bonds. Only 2D or quasi 1D systems are supported.
        '''
        if fig is None or ax is None: fig,ax=plt.subplots()
        ax.axis('off')
        ax.axis('equal')
        ax.set_title(self.name)
        for bond in self.bonds:
            nb=bond.neighbour
            if nb<0: nb=self.nneighbour+1
            if nb==1: color='k'
            elif nb==2: color='r'
            elif nb==3: color='b'
            else: color=str(nb*1.0/self.nneighbour)
            if nb==0:
                x,y=bond.spoint.rcoord[0],bond.spoint.rcoord[1]
                ax.scatter(x,y)
                if pid_on:
                    pid=bond.spoint.pid
                    if pid.scope==None:
                        tag=str(pid.site)
                    else:
                        tag=str(pid.scope)+'*'+str(pid.site)
                    ax.text(x-0.2,y+0.1,tag,fontsize=10,color='blue')
            else:
                if bond.is_intra_cell():
                    ax.plot([bond.spoint.rcoord[0],bond.epoint.rcoord[0]],[bond.spoint.rcoord[1],bond.epoint.rcoord[1]],color=color)
                else:
                    ax.plot([bond.spoint.rcoord[0],bond.epoint.rcoord[0]],[bond.spoint.rcoord[1],bond.epoint.rcoord[1]],color=color,ls='--')
        if show: plt.show()
        if save: plt.savefig(self.name+'.png')
        if close:plt.close()

class SuperLattice(Lattice):
    '''
    This class is the union of sublattices.
    Attributes:
        sublattices: list of Lattice
            The sublattices of the superlattice.
    '''

    def __init__(self,name,sublattices,vectors=[],nneighbour=1):
        '''
        Constructor.
        Parameters:
            name: string
                The name of the super-lattice.
            sublattices: list of Lattice
                The sublattices of the superlattice.
            vectors: list of 1D ndarray, optional
                The translation vectors of the superlattice.
            nneighbour: integer,optional
                The highest order of neighbours.
        '''
        super(Lattice,self).__init__()
        self.name=name
        self.sublattices=sublattices
        for lattice in sublattices:
            self.update({k:p for k,p in lattice.items() if k.scope==lattice.name})
        self.vectors=vectors
        self.reciprocals=reciprocals(vectors)
        self.nneighbour=nneighbour
        self.bonds,self.mdists=bonds(cluster=self.values(),vectors=vectors,nneighbour=nneighbour,return_mdists=True)
        for lattice in sublattices:
            self.update(lattice.attached_points())
            self.bonds.extend(lattice.attached_bonds())
