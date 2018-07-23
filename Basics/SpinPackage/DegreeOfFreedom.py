'''
----------------------
Spin degree of freedom
----------------------

Spin degree of freedom package, including:
    * constants: DEFAULT_SPIN_PRIORITY
    * classes: SID, Spin, SpinMatrix, SpinPack
    * functions: Heisenberg, Ising, S
'''

__all__=['DEFAULT_SPIN_PRIORITY','SID','Spin','SpinMatrix','SpinPack','Heisenberg','Ising','S']

from ..Utilities import decimaltostr,RZERO
from ..Geometry import PID
from ..DegreeOfFreedom import *
from collections import namedtuple
import numpy as np
import numpy.linalg as nl
import copy

DEFAULT_SPIN_PRIORITY=('scope','site','orbital','S')

class SID(namedtuple('SID',['orbital','S'])):
    '''
    Internal spin ID.

    Attributes
    ----------
    orbital : int
        The orbital index, start with 0, default value 0.
    S : int or half int
        The total spin, default value 0.5.
    '''

    @property
    def dagger(self):
        '''
        THe dagger of the spin ID.
        '''
        return self

SID.__new__.__defaults__=(0,0.5)

class Spin(Internal):
    '''
    This class defines the internal spin degrees of freedom in a single point.

    Attributes
    ----------
    norbital : int
        The number of orbitals.
    S : int or half int
        The total spin.
    '''

    def __init__(self,norbital=1,S=0.5):
        '''
        Constructor.

        Parameters
        ----------
        norbital : int, optional
            The number of orbitals.
        S : int or half int, optional
            The total spin.
        '''
        if S is not None and abs(2*S-int(2*S))>10**-6:
            raise ValueError('Spin construction error: S(%s) is not an integer or half integer.')
        self.norbital=norbital
        self.S=S

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        return 'Spin(norbital=%s,S=%s)'%(self.norbital,self.S)

    def __eq__(self,other):
        '''
        Overloaded operator(==).
        '''
        return self.norbital==other.norbital and self.S==other.S

    def indices(self,pid,mask=()):
        '''
        Return a list of all the masked indices within this internal degrees of freedom combined with an extra spatial part.

        Parameters
        ----------
        pid : PID
            The extra spatial part of the indices.
        mask : list of str, optional
            The attributes that will be masked to None.

        Returns
        -------
        list of Index
            The indices.
        '''
        pid=pid._replace(**{key:None for key in set(mask)&set(PID._fields)})
        result=[]
        for orbital in (None,) if 'orbital' in mask else xrange(self.norbital):
            result.append(Index(pid=pid,iid=SID(orbital=orbital,S=None if 'S' in mask else self.S)))
        return result

class SpinMatrix(np.ndarray):
    '''
    The matrix representation of spin operators.

    Attributes
    ----------
    S : int or half int
        The total spin.
    tag : any hashable object
        The tag of the matrix.
    '''

    def __new__(cls,S,tag,matrix=None,dtype=np.complex128,**kargs):
        '''
        Constructor.

        Parameters
        ----------
        S : int or half int
            The total spin.
        tag : 'I','i','X','x','Y','y','Z','z','P','p','M','m', and any other hashable object
            The tag of the matrix.
        matrix : 2d ndarray, optional
            The matrix of the SpinMatrix beyond the predefined set, which will be omitted when ``tag`` is in ('I','i','X','x','Y','y','Z','z','P','p','M','m').
        dtype : np.float64 or np.complex128, optional
            The data type of the matrix.
        '''
        delta=lambda i,j: 1 if i==j else 0
        result=np.zeros((int(S*2)+1,int(S*2)+1),dtype=dtype).view(cls)
        if tag in ('I','i','X','x','Y','y','Z','z','P','p','M','m'):
            tag=tag.lower()
            for i in xrange(int(S*2)+1):
                row,m=int(S*2)-i,S-i
                for j in xrange(int(S*2)+1):
                    col,n=int(S*2)-j,S-j
                    if tag=='i':
                        result[row,col]=delta(i,j)
                    elif tag=='x':
                        result[row,col]=(delta(i+1,j)+delta(i,j+1))*np.sqrt(S*(S+1)-m*n)/2
                    elif tag=='y':
                        result[row,col]=(delta(i+1,j)-delta(i,j+1))*np.sqrt(S*(S+1)-m*n)/2j
                    elif tag=='z':
                        result[row,col]=delta(i,j)*m
                    elif tag=='p':
                        result[row,col]=delta(i+1,j)*np.sqrt(S*(S+1)-m*n)
                    elif tag=='m':
                        result[row,col]=delta(i,j+1)*np.sqrt(S*(S+1)-m*n)
        elif matrix is not None:
            assert matrix.shape==result.shape
            result[...]=matrix[...]
        result.S=S
        result.tag=tag
        return result

    def __array_finalize__(self,obj):
        '''
        Initialize an instance through both explicit and implicit constructions, i.e. constructor, view and slice.
        '''
        if obj is None:
            return
        else:
            self.S=getattr(obj,'S',None)
            self.tag=getattr(obj,'tag',None)

    def __reduce__(self):
        '''
        numpy.ndarray uses __reduce__ to pickle. Therefore this method needs overriding for subclasses.
        '''
        pickle=super(SpinMatrix,self).__reduce__()
        return pickle[0],pickle[1],pickle[2]+(self.S,self.tag)

    def __setstate__(self,state):
        '''
        Set the state of the SpinMatrix for pickle and copy.
        '''
        super(SpinMatrix,self).__setstate__(state[0:-2])
        self.S=state[-2]
        self.tag=state[-1]

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return "SpinMatrix(S=%s,tag=%s,matrix=\n%s\n)"%(self.S,self.tag,super(SpinMatrix,self).__str__())

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        return "%s%s"%(self.S,self.tag)

class SpinPack(IndexPack):
    '''
    The pack of spin degrees of freedom.

    Attributes
    ----------
    tags : tuple of str
        Each element is the tag of a SpinMatrix.
    orbitals : tuple of int
        The orbital indices for the spin term.
    matrices : tuple of 2d ndarray/None
        Each element is the matrix of a SpinMatrix, which should be None when the corresponding tag is in ('I','i','X','x','Y','y','Z','z','P','p','M','m').
    '''

    def __init__(self,value,tags,orbitals=None,matrices=None):
        '''
        Constructor.

        Parameters
        ----------
        value : float or complex
            The overall coefficient of the spin pack.
        tags : tuple of str
            Each element is the tag of a SpinMatrix.
        orbitals : tuple of int, optional
            The orbital indices for the spin term.
        matrices : tuple of 2d ndarray/None, optional
            Each element is the matrix of a SpinMatrix, which should be None when the corresponding tag is in ('I','i','X','x','Y','y','Z','z','P','p','M','m').

        '''
        if matrices is not None: assert len(tags)==len(matrices)
        if orbitals is not None: assert len(tags)==len(orbitals)
        super(SpinPack,self).__init__(value)
        self.tags=tuple(tags)
        self.orbitals=(None,)*len(tags) if orbitals is None else tuple(orbitals)
        self.matrices=(None,)*len(tags) if matrices is None else tuple(matrices)

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        condition=isinstance(self.value,complex) and abs(self.value.real)>5*10**-6 and abs(self.value.imag)>5*10**-6
        temp=['(%s)'%decimaltostr(self.value)] if condition else [decimaltostr(self.value)]
        temp.append(('%s'%('%s'*len(self.tags)))%self.tags)
        if self.orbitals!=(None,)*len(self.orbitals): temp.append(('ob%s'%('%s'*len(self.orbitals)))%self.orbitals)
        return '*'.join(temp)

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return 'SpinPack(value=%s, tags=%s, orbitals=%s, matrices=%s)'%(self.value,self.tags,self.orbitals,self.matrices)

    def __mul__(self,other):
        '''
        Overloaded operator(*), which supports the multiplication of an SpinPack instance with a scalar.
        '''
        result=copy.copy(self)
        result.value*=other
        return result

    def __eq__(self,other):
        '''
        Overloaded operator(==).
        '''
        return (    self.value==other.value and
                    self.tags==other.tags and
                    self.orbitals==other.orbitals and
                    all(nl.norm(m1-m2)<RZERO if isinstance(m1,np.ndarray) and isinstance(m2,np.ndarray) else m1==m2 for m1,m2 in zip(self.matrices,other.matrices))
                    )

def Heisenberg(orbitals=None):
    '''
    The Heisenberg spin packs.
    '''
    result=IndexPacks()
    result.append(SpinPack(0.5,('p','m'),orbitals=orbitals))
    result.append(SpinPack(0.5,('m','p'),orbitals=orbitals))
    result.append(SpinPack(1.0,('z','z'),orbitals=orbitals))
    return result

def Ising(tag,orbitals=None):
    '''
    The Ising spin packs.
    '''
    assert tag in ('x','y','z','X','Y','Z')
    result=IndexPacks()
    result.append(SpinPack(1.0,(tag,tag),orbitals=orbitals))
    return result

def S(tag,matrix=None,orbital=None):
    '''
    Single spin packs.

    Parameters
    ----------
    tag : 'x','X','y','Y','z','Z', or any other hashable object.
        The tag of a SpinMatrix.
    orbital : int, optional
        The orbital index of the spin term.
    matrix : 2d ndarray, optional
        The matrix of a SpinMatrix, which should be None when `tag` is in ('x','X','y','Y','z','Z').

    Returns
    -------
    IndexPacks
        The single spin pack.
    '''
    result=IndexPacks()
    result.append(SpinPack(1.0,(tag,),orbitals=None if orbital is None else (orbital,),matrices=None if matrix is None else (matrix,)))
    return result
