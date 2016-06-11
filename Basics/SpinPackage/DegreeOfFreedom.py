'''
Spin degree of freedom package, including:
1) constants: DEFAULT_SPIN_PRIORITY
2) classes: SID, Spin, SpinMatrix, SpinPack
3) functions: Heisenberg, S
'''

__all__=['DEFAULT_SPIN_PRIORITY','SID','Spin','SpinMatrix','SpinPack','Heisenberg','S']

from ..DegreeOfFreedom import *
from numpy import *
from collections import namedtuple
import copy

DEFAULT_SPIN_PRIORITY=['socpe','site','S']

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

    @property
    def ns(self):
        '''
        The number of spin states.
        '''
        return int(2*self.S)

    def table(self,pid):
        '''
        This method returns a Table instance that contains all the allowed indices constructed from an input pid and the internal degrees of freedom.
        Parameters:
            pid: PID
                The spatial part of the indices.
        Returns: Table
            The index-sequence table.
        '''
        return Table([Index(pid=pid,iid=SID(S=self.S))])

class SpinMatrix(ndarray):
    '''
    The matrix representation of spin operators.
    '''
    def __new__(cls,id,dtype=complex128,**kargs):
        '''
        Constructor.
        Parameters:
            id: 2-tuple
                id[0]: integer or half integer
                    The total spin.
                id[1]: 'X','x','Y','y','Z','z','+','-'
                    This parameter specifies the matrix.
            dtype: float64 or complex128, optional
                The data type of the matirx.
        '''
        if isinstance(id,tuple):
            delta=lambda i,j: 1 if i==j else 0
            temp=(id[0])*(id[0]+1)
            result=zeros((int(id[0]*2)+1,int(id[0]*2)+1),dtype=dtype).view(cls)
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
                    else:
                        raise ValueError('SpinMatrix construction error: id=%s not supported.'%(id,))
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
            pack: tuple of characters
                Each character specifies a SpinMatrix.
                Each character must be in ('x','X','y','Y','z','Z','+','-')
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

def S(id):
    '''
    Single spin packs.
    Parameters:
        id: 'x','X','y','Y','z','Z'
            It specifies the single spin pack.
    Returns: IndexPackList
        The single spin pack.
    '''
    if id not in ('x','X','y','Y','z','Z'):
        raise ValueError('S error: %s not supported.'%id)
    result=IndexPackList()
    result.append(SpinPack(1.0,(id)))
    return result
