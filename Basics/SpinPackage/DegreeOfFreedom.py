'''
Spin degree of freedom package, including:
1) constants: DEFAULT_SPIN_PRIORITY, DEFAULT_SPIN_LAYERS
2) classes: SID, Spin, SpinMatrix, SpinPack
3) functions: Heisenberg, S
'''

__all__=['DEFAULT_SPIN_PRIORITY','DEFAULT_SPIN_LAYERS','SID','Spin','SpinMatrix','SpinPack','Heisenberg','S']

from ..DegreeOfFreedom import *
from numpy import *
from collections import namedtuple
import copy

DEFAULT_SPIN_PRIORITY=['socpe','site','S']
DEFAULT_SPIN_LAYERS=['scope','site','S']

class SID(namedtuple('SID',['S'])):
    '''
    Internal spin ID.
    Attributes:
        S: integer or half integer
            The total spin.
    '''

    @property
    def dagger(self):
        '''
        THe dagger of the spin ID.
        '''
        return self

class Spin(Internal):
    '''
    This class defines the internal spin degrees of freedom in a single point.
    Attributes:
        S: integer or half integer
            The total spin.
    '''
    def __init__(self,S):
        '''
        Constructor.
            S: integer or half integer
                The total spin.
        '''
        if abs(2*S-int(2*S))>10**-6:
            raise ValueError('Spin construction error: S(%s) is not an integer or half integer.')
        self.S=S

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        return 'Spin(S=%s)'%(self.S)

    def __eq__(self,other):
        '''
        Overloaded operator(==).
        '''
        return self.S==other.S

    def ndegfre(self,mask=None):
        '''
        Return the number of the interanl degrees of freedom modified by mask.
        Parameters:
            mask: list of string, optional
                Only the indices in mask can be varied in the counting of the number of the degrees of freedom.
                When None, all the allowed indices can be varied and thus the total number of the interanl degrees of freedom is returned.
        Returns: number
            The requested number of the interanl degrees of freedom.
        '''
        return int(2*self.S)+1

    def indices(self,pid):
        '''
        Return a list of all the allowed indices within this internal degrees of freedom combined with an extra spatial part.
        Parameters:
            pid: PID
                The extra spatial part of the indices.
        Returns: list of Index
            The allowed indices.
        '''
        return [Index(pid=pid,iid=SID(S=self.S))]

class SpinMatrix(ndarray):
    '''
    The matrix representation of spin operators.
    Attributes:
        id: 2-tuple
            id[0]: integer or half integer
                The total spin.
            id[1]: any hashable object
                The tag of the matrix.
    '''
    def __new__(cls,id,dtype=complex128,matrix=None,**kargs):
        '''
        Constructor.
        Parameters:
            id: 2-tuple
                id[0]: integer or half integer
                    The total spin.
                id[1]: 'X','x','Y','y','Z','z','+','-', and any other hashable object
                    This parameter specifies the matrix.
            dtype: float64 or complex128, optional
                The data type of the matirx.
            matrix: 2D ndarray, optional
                This parameter only takes on effect when id[1] is not in ('X','x','Y','y','Z','z','+','-').
                It is then used as the matrix of this SpinMatrix.
        '''
        if isinstance(id,tuple):
            delta=lambda i,j: 1 if i==j else 0
            temp=(id[0])*(id[0]+1)
            result=zeros((int(id[0]*2)+1,int(id[0]*2)+1),dtype=dtype).view(cls)
            if id[1] in ('X','x','Y','y','Z','z','+','-'):
                for i in xrange(int(id[0]*2)+1):
                    m=id[0]-i
                    for j in xrange(int(id[0]*2)+1):
                        n=id[0]-j
                        if id[1] in ('X','x'):
                            result[i,j]=(delta(i+1,j)+delta(i,j+1))*sqrt(temp-m*n)/2
                        elif id[1] in ('Y','y'):
                            result[i,j]=(delta(i+1,j)-delta(i,j+1))*sqrt(temp-m*n)/(2j)
                        elif id[1] in ('Z','z'):
                            result[i,j]=delta(i,j)*m
                        elif id[1] in ('+'):
                            result[i,j]=delta(i+1,j)*sqrt(temp-m*n)
                        elif id[1] in ('-'):
                            result[i,j]=delta(i,j+1)*sqrt(temp-m*n)
            elif matrix is not None:
                if matrix.shape!=result.shape:
                    raise ValueError("SpinMatrix construction error: id[0](%s) and the matrix's shape(%s) do not match."%(id[0],matrix.shape))
                result[...]=matrix[...]
            result.id=id
        else:
            raise ValueError('SpinMatrix construction error: id must be a tuple.')
        return result

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return "SpinMatrix(id=%s,\nmatrix=\n%s\n)"%(self.id,super(SpinMatrix,self).__str__())

class SpinPack(IndexPack):
    '''
    The pack of spin degrees of freedom.
    Attributes:
        pack: tuple of characters
            Each character specifies a SpinMatrix.
            Each character must be in ('x','X','y','Y','z','Z','+','-')
    '''

    def __init__(self,value,pack):
        '''
        Constructor.
        Parameters:
            value: float64 or complex128
                The overall coefficient of the spin pack.
            pack: tuple of characters or tuple of tuples
                when it is tuple of characters:
                    Each character specifies a SpinMatrix.
                    Each character must be in ('x','X','y','Y','z','Z','+','-')
                when it is tuple of tuples, for each tuple:
                    pack[.][0]: any hashable object
                        The second part of the id of a SpinMatrix.
                    pack[.][1]: 2D ndarray
                        The matrix of a SpinMatrix.
        '''
        super(SpinPack,self).__init__(value)
        self.pack=pack

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        return ''.join(['SpinPack(','value=%s, ','pack=%s',')'])%(self.value,self.pack)

    def __mul__(self,other):
        '''
        Overloaded operator(*), which supports the multiplication of an SpinPack instance with a scalar.
        '''
        result=copy.copy(self)
        result.value*=other
        return result

def Heisenberg():
    '''
    The Heisenberg spin packs.
    '''
    result=IndexPackList()
    result.append(SpinPack(0.5,('+','-')))
    result.append(SpinPack(0.5,('-','+')))
    result.append(SpinPack(1.0,('z','z')))
    return result

def S(id,matrix=None):
    '''
    Single spin packs.
    Parameters:
        id: 'x','X','y','Y','z','Z', or any other hashable object.
            It specifies the single spin pack.
        matrix: 2D ndarray, optional
            It specifies the matrix of the spin pack and takes on effect only when id is not in ('x','X','y','Y','z','Z').
    Returns: IndexPackList
        The single spin pack.
    '''
    result=IndexPackList()
    if id in ('x','X','y','Y','z','Z'):
        result.append(SpinPack(1.0,(id,)))
    else:
        result.append(SpinPack(1.0,((id,matrix),)))
    return result
