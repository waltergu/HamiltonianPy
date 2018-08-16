'''
------------
Lattice pack
------------

Lattice pack, including:
    * classes: Pieces, Cluster, Line, Square, Hexagon, Triangle, Kagome
'''

__all__=['Pieces','Cluster','Line','Square','Hexagon','Triangle','Kagome']

from ..Geometry import PID,Lattice,Cylinder,SuperLattice,tiling,minimumlengths
import numpy as np
import itertools as it
import re

class Pieces(object):
    '''
    The tiling pieces of a cluster.

    Attributes
    ----------
    rcoords : list of 1d ndarray
        The rcoord indices of the components of the pieces.
    baths : list of 1d ndarray
        The bath indices of the components of the pieces.
    '''

    def __init__(self,*pieces):
        '''
        Constructor.

        Parameters
        ----------
        pieces : list of 2-tuple
            The rcoord indices and bath indices of the components of the pieces.
        '''
        self.rcoords,self.baths=[],[]
        for rcoords,baths in pieces:
            self.rcoords.append(None if rcoords is None else np.asarray(rcoords))
            self.baths.append(None if baths is None else np.asarray(baths))

    def __len__(self):
        '''
        The number of the components of the pieces.
        '''
        return len(self.rcoords)

    def tiling(self,ts):
        '''
        New pieces by tiling.

        Parameters
        ----------
        ts : tuple of int
            The tiling information.

        Returns
        -------
        Pieces
            The new pieces.
        '''
        result=Pieces()
        rnum,bnum=self.rcoordnum,self.bathnum
        for i in range(np.product(ts)):
            for rcoords,baths in zip(self.rcoords,self.baths):
                if rcoords is not None: result.rcoords.append(rcoords+i*rnum)
                if baths is not None: result.baths.append(baths+i*bnum)
        return result

    @property
    def rcoordnum(self):
        '''
        The total number of rcoords of the pieces.
        '''
        return sum((0 if rcoords is None else len(rcoords)) for rcoords in self.rcoords)

    @property
    def bathnum(self):
        '''
        The total number of baths of the pieces.
        '''
        return sum((0 if baths is None else len(baths)) for baths in self.baths)

class Cluster(object):
    '''
    Cluster, the building block of a lattice.

    Attributes
    ----------
    name : str
        The name of the cluster.
    rcoords : 2d ndarray
        The rcoords of the cluster.
    vectors : 2d ndarray
        The translation vectors of the cluster.
    tiles : list of str
        The tiling information of the cluster.
    baths : 2d ndarray
        The baths of the cluster.
    pieces : Pieces
        The tiling pieces of the cluster.
    '''
    BATHRATIO=0.3

    def __init__(self,name,rcoords,vectors,tiles=None,baths=None,pieces=None):
        '''
        Constructor.

        Parameters
        ----------
        name : str
            The name of the cluster.
        rcoords : 2d ndarray
            The rcoords of the cluster.
        vectors : 2d ndarray
            The translation vectors of the cluster.
        tiles : list of str, optional
            The tiling information of the cluster.
        baths : 2d ndarray, optional
            The baths of the cluster.
        pieces : Pieces, optional
            The tiling pieces of the cluster.
        '''
        if pieces is not None: assert pieces.rcoordnum==len(rcoords) and (baths is None or pieces.bathnum==len(baths))
        self.name=name
        self.rcoords=np.asarray(rcoords)
        self.vectors=np.asarray(vectors)
        self.tiles=tiles
        self.baths=None if baths is None else np.asarray(baths)
        self.pieces=pieces

    @property
    def piecenum(self):
        '''
        The number of pieces of the cluster.
        '''
        return 0 if self.pieces is None else len(self.pieces)

    def tiling(self,ts):
        '''
        Construct a new cluster by tiling.

        Parameters
        ----------
        ts : tuple of int
            The tiling information.

        Returns
        -------
        Cluster
            The new cluster after the tiling.
        '''
        assert len(ts)==len(self.vectors)
        result=Cluster.__new__(self.__class__)
        result.name=self.name
        result.rcoords=np.asarray(tiling(cluster=self.rcoords,vectors=self.vectors,translations=it.product(*[range(t) for t in ts])))
        result.vectors=np.asarray([self.vectors[i]*t for i,t in enumerate(ts)])
        result.tiles=['%s%s'%('' if self.tiles is None else self.tiles[i]+'^',str(t)) for t in ts]
        result.baths=None if self.baths is None else np.asarray(tiling(cluster=self.baths,vectors=self.vectors,translations=it.product(*[range(t) for t in ts])))
        result.pieces=None if self.pieces is None else self.pieces.tiling(ts)
        return result

    def __call__(self,tbs=None,nneighbour=1):
        '''
        Construct a lattice according the translation and boundary conditions.

        Parameters
        ----------
        tbs : str, optional
            The translation and boundary conditions.
        nneighbour : int, optional
            The highest order of the neighbours.

        Returns
        -------
        Lattice
            The constructed lattice.
        '''
        ts,bcs=([1]*len(self.vectors),['P']*len(self.vectors)) if tbs is None else ([int(t) for t in re.findall('\d+',tbs)],re.findall('[P,p,O,o]',tbs))
        assert len(ts)==len(bcs)==len(self.vectors)
        cluster=self.tiling(ts) if np.any(ts)>0 else self
        tiles,neighbours=['']*len(self.vectors) if self.tiles is None else ['%s^'%tile for tile in self.tiles],cluster.neighbours(nneighbour,bcs)
        if cluster.baths is None:
            return Lattice(
                    name=           '%s(%s)'%(self.name,'-'.join('%s%s%s'%(tile,t,bc.upper()) for tile,t,bc in zip(tiles,ts,bcs))),
                    pids=           [PID(cluster.name,i) for i in range(len(cluster.rcoords))] if cluster.pieces is None else
                                    [PID('%s-%s'%(cluster.name,i),j) for i in range(len(cluster.pieces)) for j in range(len(cluster.pieces.rcoords[i]))],
                    rcoords=        cluster.rcoords,
                    vectors=        [cluster.vectors[i] for i,bc in enumerate(bcs) if bc.lower()=='p'],
                    neighbours=     neighbours
                    )
        else:
            lattice=Lattice(
                    name=           '%s-L'%self.name,
                    pids=           [PID('%s-L'%cluster.name,i) for i in range(len(cluster.rcoords))] if cluster.pieces is None else
                                    [PID('%s-L%s'%(cluster.name,i),j) for i in range(len(cluster.pieces)) for j in range(len(cluster.pieces.rcoords[i]))],
                    rcoords=        cluster.rcoords,
                    vectors=        [cluster.vectors[i] for i,bc in enumerate(bcs) if bc.lower()=='p'],
                    neighbours=     neighbours
                    )
            bath=Lattice(
                    name=           '%s-BATH'%self.name,
                    pids=           [PID('%s-BATH'%cluster.name,i) for i in range(len(cluster.baths))] if cluster.pieces is None else
                                    [PID('%s-BATH%s'%(cluster.name,i),j) for i in range(len(cluster.pieces)) for j in range(len(cluster.pieces.baths[i]))],
                    rcoords=        cluster.baths,
                    neighbours=     0
                    )
            return SuperLattice(
                    name=           '%s(%s)'%(self.name,'-'.join('%s%s%s'%(tile,t,bc.upper()) for tile,t,bc in zip(tiles,ts,bcs))),
                    sublattices=    [lattice,bath],
                    vectors=        [cluster.vectors[i] for i,bc in enumerate(bcs) if bc.lower()=='p'],
                    neighbours=     neighbours
                    )

    def sublattice(self,index,nneighbour=1):
        '''
        Construct a sublattice from a piece of the cluster.

        Parameters
        ----------
        index : int
            The index of the piece.
        nneighbour : int, optional
            The highest order of the neighbours.

        Returns
        -------
        Lattice
            The constructed sublattice.
        '''
        assert self.pieces is not None
        neighbours=self.neighbours(nneighbour,['O']*len(self.vectors))
        if self.baths is None:
            return Lattice(
                    name=           '%s%s-%s'%(self.name,'' if self.tiles is None else '(%s)'%('-'.join(str(tile) for tile in self.tiles)),index),
                    pids=           [PID('%s-%s'%(self.name,index),i) for i in range(len(self.pieces.rcoords[index]))],
                    rcoords=        self.rcoords[self.pieces.rcoords[index]],
                    neighbours=     neighbours
                    )
        else:
            lattice=Lattice(
                    name=           '%s%s-L%s'%(self.name,'' if self.tiles is None else '(%s)'%('-'.join(str(tile) for tile in self.tiles)),index),
                    pids=           [PID('%s-L%s'%(self.name,index),i) for i in range(len(self.pieces.rcoords[index]))],
                    rcoords=        self.rcoords[self.pieces.rcoords[index]],
                    neighbours=     neighbours
                    )
            bath=Lattice(
                    name=           '%s%s-BATH%s'%(self.name,'' if self.tiles is None else '(%s)'%('-'.join(str(tile) for tile in self.tiles)),index),
                    pids=           [PID('%s-BATH%s'%(self.name,index),i) for i in range(len(self.pieces.baths[index]))],
                    rcoords=        self.baths[self.pieces.baths[index]],
                    neighbours= 0
                    )
            return SuperLattice(
                    name=           '%s%s-%s'%(self.name,'' if self.tiles is None else '(%s)'%('-'.join(str(tile) for tile in self.tiles)),index),
                    sublattices=    [lattice,bath],
                    neighbours=     neighbours
                    )

    def cylinder(self,dt,tbs=None,nneighbour=1):
        '''
        Construct a cylinder according to the extension direction and the translation-and-boundary conditions.

        Parameters
        ----------
        dt : int
            The direction along which the cylinder extends.
        tbs : str, optional
            The translation and boundary conditions.
        nneighbour : int, optional
            The highest order of the neighbours.

        Returns
        -------
        Cylinder
            The constructed cylinder.

        Notes
        -----
        The condition of the boundary along the cylinder must be open.
        '''
        tbs='-'.join(['1O']*len(self.vectors)) if tbs is None else tbs
        ts,bcs=re.findall('\d+',tbs),re.findall('[P,p,O,o]',tbs)
        assert self.baths is None and self.pieces is None and len(ts)==len(bcs)==len(self.vectors) and bcs[dt].upper()=='O'
        tiles=['']*len(self.vectors) if self.tiles is None else ['%s^'%tile for tile in self.tiles]
        return Cylinder(
                    name=           '%s(%s)'%(self.name,'-'.join('%s%s^%s'%(tile,t,('+' if i==dt else '')+bc.upper()) for i,(tile,t,bc) in enumerate(zip(tiles,ts,bcs)))),
                    block=          tiling(self.rcoords,vectors=self.vectors,translations=it.product(*[range(int(t)) for t in ts])),
                    translation=    self.vectors[dt]*int(ts[dt]),
                    vectors=        [self.vectors[i]*int(t) for i,(t,bc) in enumerate(zip(ts,bcs)) if bc.upper()=='P'],
                    neighbours=     nneighbour
                    )

    @property
    def unit(self):
        '''
        The least unit cell of the cluster.

        Returns
        -------
        rcoords : 2d ndarray-like
            The coordinates of the least unit cell.
        vectors : 2d ndarray-like
            The translation vectors of the least unit cell.
        '''
        raise NotImplementedError('%s unit error: not implemented.'%self.__class__.__name__)

    def neighbours(self,nneighbour,bcs=None):
        '''
        The neighbour-length map of the cluster.

        Parameters
        ----------
        nneighbour : int
            The highest order of the neighbours.
        bcs : iterable of 'O','o','P' and 'p', optional
            The boundary conditions of the cluster.

        Returns
        -------
        dict
            The requested neighbour-length map.
        '''
        bcs='P'*len(self.vectors) if bcs is None else ''.join(bcs).upper()
        assert len(bcs)==len(self.vectors)
        rcoords,vectors=self.unit if set(bcs)=={'P'} else (self.rcoords,[vector for i,vector in enumerate(self.vectors) if bcs[i]=='P'])
        result={order:length for order,length in enumerate(minimumlengths(rcoords,vectors,nneighbour))}
        if self.baths is not None: result[-1]=result[1]*Cluster.BATHRATIO
        return result

class Line(Cluster):
    '''
    Cluster of one dimensional lattices.
    '''
    unit=[[0.0,0.0]],[[1.0,0.0]]

    def __init__(self,name,pieces=False):
        '''
        Constructor.

        Parameters
        ----------
        name : 'L1','L2'
            The name of the cluster.
        pieces : logical, optional
            True for generating pieces for the cluster and False for not.
        '''
        if name not in ['L1','L2']:
            raise ValueError('Line __init__ error: unexpected name(%s).'%name)
        if name=='L1':
            rcoords=[[0.0,0.0]]
            vectors=[[1.0,0.0]]
            baths=None
            pieces=Pieces((list(range(len(rcoords))),None)) if pieces else None
        elif name=='L2':
            rcoords=[[0.0,0.0],[1.0,0.0]]
            vectors=[[2.0,0.0]]
            baths=None
            pieces=Pieces((list(range(len(rcoords))),None)) if pieces else None
        super(Line,self).__init__(name,rcoords,vectors,baths=baths,pieces=pieces)

class Square(Cluster):
    '''
    Cluster of square lattices.
    '''
    unit=[[0.0,0.0]],[[1.0,0.0],[0.0,1.0]]
    BNBS=0.3

    def __init__(self,name,pieces=False):
        '''
        Constructor.

        Parameters
        ----------
        name : 'S1','S2x','S2y','S2xxy','S2yxy','S4','S4B4','S4B8','S8','S10','S12','S13'
            The name of the cluster.
        pieces : logical, optional
            True for generating pieces for the cluster and False for not.
        '''
        if name not in ['S1','S2x','S2y','S2xxy','S2yxy','S4','S4B4','S4B8','S8','S10','S12','S13']:
            raise ValueError('Square __init__ error: unexpected name(%s).'%name)
        if name=='S1':
            rcoords=[[0.0,0.0]]
            vectors=[[1.0,0.0],[0.0,1.0]]
            baths=None
            pieces=Pieces((list(range(len(rcoords))),None)) if pieces else None
        elif name=='S2x':
            rcoords=[[0.0,0.0],[1.0,0.0]]
            vectors=[[2.0,0.0],[0.0,1.0]]
            baths=None
            pieces=Pieces((list(range(len(rcoords))),None)) if pieces else None
        elif name=='S2y':
            rcoords=[[0.0,0.0],[0.0,1.0]]
            vectors=[[1.0,0.0],[0.0,2.0]]
            baths=None
            pieces=Pieces((list(range(len(rcoords))),None)) if pieces else None
        elif name=='S2xxy':
            rcoords=[[0.0,0.0],[1.0,0.0]]
            vectors=[[1.0,1.0],[1.0,-1.0]]
            baths=None
            pieces=Pieces((list(range(len(rcoords))),None)) if pieces else None
        elif name=='S2yxy':
            rcoords=[[0.0,0.0],[0.0,1.0]]
            vectors=[[1.0,1.0],[1.0,-1.0]]
            baths=None
            pieces=Pieces((list(range(len(rcoords))),None)) if pieces else None
        elif name=='S4' or name=='S4B4' or name=='S4B8':
            rcoords=[[0.0,0.0],[1.0,0.0],[0.0,1.0],[1.0,1.0]]
            vectors=[[2.0,0.0],[0.0,2.0]]
            if name=='S4':
                baths=None
                pieces=Pieces((list(range(len(rcoords))),None)) if pieces else None
            elif name=='S4B4':
                baths=[[-np.sqrt(2)*3/20, -np.sqrt(2)*3/20],[np.sqrt(2)*3/20+1, -np.sqrt(2)*3/20],
                       [-np.sqrt(2)*3/20,np.sqrt(2)*3/20+1],[np.sqrt(2)*3/20+1,np.sqrt(2)*3/20+1]
                       ]
                pieces=Pieces((list(range(len(rcoords))),list(range(len(baths))))) if pieces else None
            else:
                baths=[[-0.3,0.0],[0.0,-0.3],[1.0,-0.3],[ 1.3,0.0],
                       [ 1.3,1.0],[1.0, 1.3],[0.0, 1.3],[-0.3,1.0]
                       ]
                pieces=Pieces((list(range(len(rcoords))),list(range(len(baths))))) if pieces else None
        elif name=='S8':
            rcoords=[[0.0,0.0],[1.0,0.0],[2.0,0.0],[1.0,-1.0],
                     [0.0,1.0],[1.0,1.0],[2.0,1.0],[1.0, 2.0]
                     ]
            vectors=[[2.0,2.0],[2.0,-2.0]]
            baths=None
            pieces=Pieces((list(range(len(rcoords))),None)) if pieces else None
        elif name=='S10':
            rcoords=[[0.0,0.0],[1.0,0.0],[0.0,1.0],[1.0,1.0],
                     [0.0,2.0],[2.0,1.0],[1.0,2.0],[2.0,2.0],
                     [1.0,3.0],[2.0,3.0]
                     ]
            vectors=[[3.0,1.0],[-1.0,3.0]]
            baths=None
            pieces=Pieces((list(range(len(rcoords))),None)) if pieces else None
        elif name=='S12':
            rcoords=[[ 0.0,0.0],[1.0,0.0],[0.0,1.0],[1.0,1.0],
                     [-1.0,1.0],[2.0,1.0],[0.0,2.0],[1.0,2.0],
                     [-1.0,2.0],[2.0,2.0],[0.0,3.0],[1.0,3.0]
                     ]
            vectors=[[2.0,3.0],[-2.0,3.0]]
            baths=None
            pieces=Pieces((list(range(len(rcoords))),None)) if pieces else None
        elif name=='S13':
            rcoords=[[ 0.0, 0.0],[1.0,0.0],[-1.0,0.0],[0.0, 1.0],
                     [ 0.0,-1.0],[1.0,1.0],[-1.0,1.0],[1.0,-1.0],
                     [-1.0,-1.0],[2.0,0.0],[-2.0,0.0],[0.0, 2.0],
                     [ 0.0,-2.0]
                     ]
            vectors=[[3.0,2.0],[-2.0,3.0]]
            baths=None
            pieces=Pieces((list(range(len(rcoords))),None)) if pieces else None
        super(Square,self).__init__(name,rcoords,vectors,baths=baths,pieces=pieces)

class Hexagon(Cluster):
    '''
    Cluster of hexagonal lattices.
    '''
    unit=[[0.0,0.0],[0.0,np.sqrt(3)/3]],[[1.0,0.0],[0.5,np.sqrt(3)/2]]
    BNBS=np.sqrt(3)/10

    def __init__(self,name,pieces=False):
        '''
        Constructor.

        Parameters
        ----------
        name : 'H2','H2B4','H4','H6','H6B6','H8O','H8P','H10','H24','H4C','H4CB6C'
            The name of the cluster.
        pieces : logical, optional
            True for generating pieces for the cluster and False for not.
        '''
        if name not in ['H2','H2B4','H4','H6','H6B6','H8O','H8P','H10','H24','H4C','H4CB6C']:
            raise ValueError('Hexagon __init__ error: unexpected name(%s).'%name)
        if name=='H2' or name=='H2B4':
            rcoords=[[0.0,0.0],[0.0,np.sqrt(3)/3]]
            vectors=[[1.0,0.0],[0.5,np.sqrt(3)/2]]
            if name=='H2':
                baths=None
                pieces=Pieces((list(range(len(rcoords))),None)) if pieces else None
            else:
                baths=[[-0.15,-np.sqrt(3)/20],[0.15,-np.sqrt(3)/20],[0.15,np.sqrt(3)*23/60],[-0.15,np.sqrt(3)*23/60]]
                pieces=Pieces((list(range(len(rcoords))),list(range(len(baths))))) if pieces else None
        elif name=='H4':
            rcoords=[[0.0,0.0],[0.0,np.sqrt(3)/3],[0.5,np.sqrt(3)/2],[0.5,-np.sqrt(3)/6]]
            vectors=[[1.0,0.0],[0.0,np.sqrt(3)]]
            baths=None
            pieces=Pieces((list(range(len(rcoords))),None)) if pieces else None
        elif name=='H6' or name=='H6B6':
            rcoords=[[0.0,0.0],[0.0,np.sqrt(3)/3],[0.5,np.sqrt(3)/2],[0.5,-np.sqrt(3)/6],
                     [1.0,0.0],[1.0,np.sqrt(3)/3]
                     ]
            vectors=[[1.5,np.sqrt(3)/2],[1.5,-np.sqrt(3)/2]]
            if name=='H6':
                baths=None
                pieces=Pieces((list(range(len(rcoords))),None)) if pieces else None
            else:
                baths=[[-0.15,-np.sqrt(3)/20],[-0.15,np.sqrt(3)*23/60],[0.5,np.sqrt(3)*3/5],[0.5,-np.sqrt(3)*4/15],
                       [ 1.15,-np.sqrt(3)/20],[ 1.15,np.sqrt(3)*23/60]
                       ]
                pieces=Pieces((list(range(len(rcoords))),list(range(len(baths))))) if pieces else None
        elif name=='H8P':
            rcoords=[[0.0,0.0],[0.0,np.sqrt(3)/3],[0.5, np.sqrt(3)/2],[0.5, -np.sqrt(3)/6],
                     [1.0,0.0],[1.0,np.sqrt(3)/3],[0.5,-np.sqrt(3)/2],[0.5,np.sqrt(3)*5/6]
                     ]
            vectors=[[1.0,np.sqrt(3)],[1.5,-np.sqrt(3)/2]]
            baths=None
            pieces=Pieces((list(range(len(rcoords))),None)) if pieces else None
        elif name=='H8O':
            rcoords=[[0.0,0.0],[0.0,np.sqrt(3)/3],[0.5,np.sqrt(3)/2],[0.5,-np.sqrt(3)/6],
                     [1.0,0.0],[1.0,np.sqrt(3)/3],[1.5,np.sqrt(3)/2],[1.5,-np.sqrt(3)/6]
                     ]
            vectors=[[2.0,0.0],[0.0,np.sqrt(3)]]
            baths=None
            pieces=Pieces((list(range(len(rcoords))),None)) if pieces else None
        elif name=='H10':
            rcoords=[[0.0,0.0],[0.0,np.sqrt(3)/3],[0.5,np.sqrt(3)/2],[0.5,-np.sqrt(3)/6],
                     [1.0,0.0],[1.0,np.sqrt(3)/3],[1.5,np.sqrt(3)/2],[1.5,-np.sqrt(3)/6],
                     [2.0,0.0],[2.0,np.sqrt(3)/3]
                     ]
            vectors=[[2.5,np.sqrt(3)/2],[0.0,np.sqrt(3)]]
            baths=None
            pieces=Pieces((list(range(len(rcoords))),None)) if pieces else None
        elif name=='H24':
            rcoords=[[0.0,          0.0],[0.0,   np.sqrt(3)/3],[0.5, np.sqrt(3)/2],[0.5,  -np.sqrt(3)/6],
                     [1.0,          0.0],[1.0,   np.sqrt(3)/3],[1.5, np.sqrt(3)/2],[1.5,  -np.sqrt(3)/6],
                     [2.0,          0.0],[2.0,   np.sqrt(3)/3],[2.5, np.sqrt(3)/2],[2.5,  -np.sqrt(3)/6],
                     [3.0,          0.0],[3.0,   np.sqrt(3)/3],[0.5,-np.sqrt(3)/2],[1.0,-np.sqrt(3)*2/3],
                     [1.5,-np.sqrt(3)/2],[2.0,-np.sqrt(3)*2/3],[2.5,-np.sqrt(3)/2],[0.5, np.sqrt(3)*5/6],
                     [1.0,   np.sqrt(3)],[1.5, np.sqrt(3)*5/6],[2.0,   np.sqrt(3)],[2.5, np.sqrt(3)*5/6]
                     ]
            vectors=[[3.0,np.sqrt(3)],[3.0,-np.sqrt(3)]]
            baths=None
            pieces=Pieces((list(range(len(rcoords))),None)) if pieces else None
        elif name=='H4C' or name=='H4CB6C':
            rcoords=[[0.5,-np.sqrt(3)/6],[0.0,         0.0],[0.5,-np.sqrt(3)/2],[1.0,           0.0],
                     [0.5, np.sqrt(3)/2],[0.0,np.sqrt(3)/3],[1.0, np.sqrt(3)/3],[0.5,np.sqrt(3)*5/6]
                     ]
            vectors=[[1.0,np.sqrt(3)],[1.5,-np.sqrt(3)/2]]
            if name=='H4C':
                baths=None
                pieces=Pieces(([0,1,2,3],None),([4,5,6,7],None))
            else:
                baths=[[ 0.0,  np.sqrt(3)/10],[-0.15,  -np.sqrt(3)/20],[ 0.35,-np.sqrt(3)*11/20],[0.65,-np.sqrt(3)*11/20],
                       [1.15, -np.sqrt(3)/20],[  1.0,   np.sqrt(3)/10],[-0.15, np.sqrt(3)*23/60],[ 0.0,  np.sqrt(3)*7/30],
                       [ 1.0,np.sqrt(3)*7/30],[ 1.15,np.sqrt(3)*23/60],[ 0.65, np.sqrt(3)*53/60],[0.35, np.sqrt(3)*53/60]
                       ]
                pieces=Pieces(([0,1,2,3],[0,1,2,3,4,5]),([4,5,6,7],[6,7,8,9,10,11]))
        super(Hexagon,self).__init__(name,rcoords,vectors,baths=baths,pieces=pieces)

class Triangle(Cluster):
    '''
    Cluster of triangular lattices.
    '''
    unit=[[0.0,0.0]],[[1.0,0.0],[0.5,np.sqrt(3)/2]]

    def __init__(self,name,pieces=False):
        '''
        Constructor.

        Parameters
        ----------
        name : 'T1','T3','T12'
            The name of the cluster.
        pieces : logical, optional
            True for generating pieces for the cluster and False for not.
        '''
        if name not in ['T1','T3','T12']:
            raise ValueError('Triangle __init__ error: unexpected name(%s).'%name)
        if name=='T1':
            rcoords=[[0.0,0.0]]
            vectors=[[1.0,0.0],[0.5,np.sqrt(3)/2]]
            baths=None
            pieces=Pieces((list(range(len(rcoords))),None)) if pieces else None
        elif name=='T3':
            rcoords=[[0.0,0.0],[1.0,0.0],[0.5,np.sqrt(3)/2]]
            vectors=[[1.5,np.sqrt(3)/2],[1.5,-np.sqrt(3)/2]]
            baths=None
            pieces=Pieces((list(range(len(rcoords))),None)) if pieces else None
        elif name=='T12':
            rcoords=[[0.0,          0.0],[1.0,          0.0],[2.0,          0.0],[3.0,         0.0],
                     [0.5,-np.sqrt(3)/2],[1.5,-np.sqrt(3)/2],[2.5,-np.sqrt(3)/2],[1.0, -np.sqrt(3)],
                     [2.0,  -np.sqrt(3)],[0.5, np.sqrt(3)/2],[1.5, np.sqrt(3)/2],[2.5,np.sqrt(3)/2]
                     ]
            vectors=[[0.0,2*np.sqrt(3)],[3.0,np.sqrt(3)]]
            baths=None
            pieces=Pieces((list(range(len(rcoords))),None)) if pieces else None
        super(Triangle,self).__init__(name,rcoords,vectors,baths=baths,pieces=pieces)

class Kagome(Cluster):
    '''
    Cluster of Kagome lattices.
    '''
    unit=[[0.0,0.0],[0.5,0.0],[0.25,np.sqrt(3.0)/4]],[[1.0,0.0],[0.5,np.sqrt(3)/2]]

    def __init__(self,name,pieces=False):
        '''
        Constructor.

        Parameters
        ----------
        name : 'K3','K9','K12'
            The name of the cluster.
        pieces : logical, optional
            True for generating pieces for the cluster and False for not.
        '''
        if name not in ['K3','K9','K12']:
            raise ValueError('Kagome __init__ error: unexpected name(%s).'%name)
        if name=='K3':
            rcoords=[[0.0,0.0],[0.5,0.0],[0.25,np.sqrt(3.0)/4]]
            vectors=[[1.0,0.0],[0.5,np.sqrt(3.0)/2]]
            baths=None
            pieces=Pieces((list(range(len(rcoords))),None)) if pieces else None
        elif name=='K9':
            rcoords=[[ 0.0,             0.0],[ 0.5,           0.0],[0.25,np.sqrt(3.0)/4],[1.0,           0.0],
                     [ 1.5,             0.0],[1.25,np.sqrt(3.0)/4],[ 0.5,np.sqrt(3.0)/2],[1.0,np.sqrt(3.0)/2],
                     [0.75,np.sqrt(3.0)*3/4]
                     ]
            vectors=[[1.5,np.sqrt(3.0)/2],[1.5,-np.sqrt(3.0)/2]]
            baths=None
            pieces=Pieces((list(range(len(rcoords))),None)) if pieces else None
        elif name=='K12':
            rcoords=[[ 0.0,             0.0],[ 0.5,           0.0],[0.25,np.sqrt(3.0)/4],[ 1.0,            0.0],
                     [ 1.5,             0.0],[1.25,np.sqrt(3.0)/4],[ 0.5,np.sqrt(3.0)/2],[ 1.0, np.sqrt(3.0)/2],
                     [0.75,np.sqrt(3.0)*3/4],[ 1.5,np.sqrt(3.0)/2],[ 0.0,np.sqrt(3.0)/2],[0.75,-np.sqrt(3.0)/4]
                     ]
            vectors=[[2.0,0.0],[1.0,np.sqrt(3.0)]]
            baths=None
            pieces=Pieces((list(range(len(rcoords))),None)) if pieces else None
        super(Kagome,self).__init__(name,rcoords,vectors,baths=baths,pieces=pieces)
