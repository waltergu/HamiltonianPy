'''
Quantum number, including:
1) classes: QuantumNumber, QuantumNumbers
'''

from collections import namedtuple
from copy import copy,deepcopy
import numpy as np

__all__=['QuantumNumber','QuantumNumbers']

class QuantumNumber(np.ndarray):
    '''
    Quantum number, which itself is a 1d ndarray with the elements being the values of the quantum number.
    Attributes:
        names: tuple of string
            The names of the quantum number.
        periods: tuple of integer
            The periods of the quantum number.
    '''
    names=()
    periods=()

    def __new__(cls,values):
        '''
        Constructor.
        Parameters:
            values: iterable of numbers
                The values of the quantum number.
        '''
        self=np.asarray(values).reshape(-1).view(cls)
        assert len(self)==len(cls.names)
        return self

    @classmethod
    def __set_names_and_periods__(cls,names,periods):
        '''
        Set the names and periods of the quantum number.
        Parameters:
            names: list of str
                The names of the quantum number.
            periods: list of None/posotive integer
                The periods of the quantum number.
        '''
        assert len(names)==len(periods)
        for name,period in zip(names,periods):
            assert isinstance(name,str)
            assert (period is None) or (period in (long,int) and period>0)
        cls.names=tuple(names)
        cls.periods=tuple(periods)

    def __len__(self):
        '''
        The length of the quantum number.
        '''
        return len(np.asarray(self))

    def __getattr__(self,key):
        '''
        Overloaded dot(.) operator.
        '''
        try:
            return self[self.__class__.names.index(key)]
        except ValueError:
            raise AttributeError("%s has no attribute '%s'."%(self.__class__.__name__,key))

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        temp=[]
        for name,value,period in zip(self.__class__.names,self,self.__class__.periods):
            temp.append(name)
            temp.append(value)
            temp.append('U1' if period is None else 'Z%s'%(period))
        return ''.join(['QN','(',','.join(['%s=%r(%s)']*len(self)),')'])%tuple(temp)

    __str__=__repr__

    def __neg__(self):
        '''
        Overloaded negative(-) operator.
        '''
        values=-np.asarray(self)
        for i,t in enumerate(self.__class__.periods):
            if t is not None: values[i]%=t
        return self.__class__(values)

    def __add__(self,other):
        '''
        Overloaded addition(+) operator, which supports the addition of two quantum numbers.
        '''
        if isinstance(other,QuantumNumbers):
            pass
        else:
            assert self.__class__ is other.__class__
            values=np.asarray(self)+np.asarray(other)
            for i,t in enumerate(self.__class__.periods):
                if t is not None: values[i]%=t
            return self.__class__(values)

    def __sub__(self,other):
        '''
        Overloaded subtraction(-) operator, which supports the subtraction of two quantum numbers.
        '''
        if isinstance(other,QuantumNumbers):
            pass
        else:
            assert self.__class__ is other.__class__
            values=np.asarray(self)-np.asarray(other)
            for i,t in enumerate(self.__class__.periods):
                if t is not None: values[i]%=t
            return self.__class__(values)

    def __mul__(self):
        '''
        Overloaded multiplication(*) operator.
        '''
        raise NotImplementedError()

    def __rmul__(self,other):
        '''
        Overloaded multiplication(*) operator.
        '''
        return self.__mul__(other)

    def replace(self,**karg):
        '''
        Return a new QuantumNumber object with specified fields replaced with new values.
        '''
        result=self.__class__(map(karg.pop,self.__class__.names,self))
        if karg:
            raise ValueError('%s replace error: it got unexpected field names: %r'%(self.__class__.__name__,karg.keys()))
        return result

    @property
    def zeros(self):
        '''
        Return a new quantum number with all the values equal to zero.
        '''
        return self.__class__([0 for i in xrange(len(self))])

class QuantumNumbers(object):
    '''
    A collection of quantum numbers in a stroage format similiar to that of compressed-sparse-row vectors.
    Attributes:
        type: class
            The class of the quantum numbers contained in the collection.
        contents: 2d ndarray
            The ndarray representation of the collection with the columns representing the set of its quantum numbers.
        indptr: 1d ndarray of integers
            The index pointer array of the set of the quantum numbers.
        history: dict with each of its (key,value) pairs
            key: integer
                The id of the collection of quantum numbers whose history is recorded.
            value: 1d ndarray of integer
                The permutation to make the quantum numbers in order.
    '''
    history={}
    INDPTR,COUNT,FULL=0,1,2

    def __init__(self,qns,counts):
        '''
        Constructor.
        Parameters:
            qns: list of QuantumNumber
                The quantum numbers contained in the collection.
            counts: list of integer
                The counts of the duplicates of the quantum numbers.
        '''
        assert len(qns)==len(counts)
        self.type=next(iter(qns)).__class__
        self.contents=np.array(qns).T
        self.indptr=np.zeros(len(counts)+1,dtype=np.int64)
        for i,count in enumerate(counts):
            assert count>0
            self.indptr[i+1]=self.indptr[i]+count

    @staticmethod
    def load(type,data,protocal=INDPTR,history=False):
        '''
        Constructor.
        Parameters:
            type: class
                The class of the quantum numbers contained in the collection.
            data: three cases
            1) when protocal==QuantumNumbers.INDPTR: 2-tuple with
                data[0]: 2d ndarray
                    The contents of the collection.
                data[1]: 1d ndarray of integers
                    The indptr of the collection.
            2) when protocal==QuantumNumbers.COUNT: 2-tuple with
                data[0]: 2d ndarray
                    The contents of the collection.
                data[1]: 1d ndarray
                    The counts of the duplicates of the quantum numbers.
            3) when protocal==QuantumNumbers.FULL: ndarray
                The full ndarray representation of the collection, with the duplicates explicitly expanded.
            history: logical, optional
                It only takes effects when protocal is 2. When True, the permutation to make the quantum numbers in order will be recorded.
        '''
        self=object.__new__(QuantumNumbers)
        self.type=type
        if protocal==QuantumNumbers.INDPTR:
            contents,indptr=data
            assert contents.ndim==2 and indptr.ndim==1
            self.contents=contents
            self.indptr=indptr
        elif protocal==QuantumNumbers.COUNT:
            contents,counts=data
            assert contents.ndim==2 and counts.ndim==1
            self.contents=contents
            self.indptr=np.zeros(len(counts)+1,dtype=np.int64)
            for i,count in enumerate(counts):
                assert count>0
                self.indptr[i+1]=self.indptr[i]+count
        else:
            assert data.ndim==2
            permutation=np.lexsort(data[::-1])
            if history:
                QuantumNumbers.history[id(self)]=permutation
            mask=np.concatenate(([True],np.any(data[:,permutation[1:]]!=data[:,permutation[:-1]],axis=0)))
            self.contents=data[:,permutation][:,mask]
            self.indptr=np.concatenate((np.argwhere(mask).reshape((-1)),[data.shape[1]]))
        return self

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        return ''.join(['QNS(',','.join(['%s:(%s:%s)'%(qn,self.indptr[i],self.indptr[i+1]) for i,qn in enumerate(self)]),')'])

    def __len__(self):
        '''
        The total number of quantum numbers when duplicates are counted duplicately.
        '''
        return self.indptr[-1]

    @property
    def num(self):
        '''
        The number of unduplicate quantum numbers of the collection.
        '''
        return self.contents.shape[1]

    @staticmethod
    def clear_history(*arg):
        '''
        Clear the historical information of the collections of quantum numbers.
        '''
        if len(arg)==0:
            QuantumNumbers.history.clear()
        else:
            for qns in arg:
                if isinstance(qns,QuantumNumbers):
                    QuantumNumbers.history.pop(id(qns),None)

    def __iter__(self):
        '''
        Return an iterator over the quantum number it contains.
        '''
        for values in self.contents.T:
            yield self.type(values)

    def index(self,qn):
        '''
        Find the index of the quantum number.
        Parameters:
            qn: QuantumNumber
                The quantum number whose index is inquired.
        Returns: integer
            The corresponding index of the quantum number.
        '''
        L,R,pos=0,self.num,self.num/2
        qn,current=tuple(qn),tuple(self.contents[:,pos])
        while current!=qn:
            if current>qn:
                R=pos
            else:
                L=pos
            pos,last=(L+R)/2,pos
            if pos==last:
                raise ValueError('QuantumNumbers index error: %s not in QuantumNumbers.'%(qn))
            current=tuple(self.contents[:,pos])
        else:
            return pos

    def __contains__(self,qn):
        '''
        Judge whether a quantum number is contained in the collection.
        '''
        try:
            self.index(qn)
            return True
        except ValueError:
            return False

    def __neg__(self):
        '''
        Overloaded negative(-) operator.
        '''
        contents=-self.contents
        for i,t in enumerate(self.type.periods):
            if t is not None: contents[i,:]%=t
        result=QuantumNumbers.load(self.type,(contents,self.indptr),protocal=QuantumNumbers.INDPTR)
        return result

    def __add__(self,other):
        '''
        Overloaded addition(+) operator.
        '''
        assert self.type is other.__class__ or not issubclass(other.__class__,QuantumNumber)
        contents=self.contents+np.asarray(other)[:,np.newaxis]
        for i,t in enumerate(self.type.periods):
            if t is not None: contents[i,:]%=t
        result=QuantumNumbers.load(self.type,(contents,self.indptr),protocal=QuantumNumbers.INDPTR)
        return result

    def __sub__(self,other):
        '''
        Overloaded subtraction(-) operator.
        '''
        assert self.type is other.__class__ or not issubclass(other.__class__,QuantumNumber)
        contents=self.contents-np.asarray(other)[:,np.newaxis]
        for i,t in enumerate(self.type.periods):
            if t is not None: contents[i,:]%=t
        result=QuantumNumbers.load(self.type,(contents,self.indptr),protocal=QuantumNumbers.INDPTR)
        return result

    def __rsub__(self,other):
        '''
        Overloaded subtraction(-) operator.
        '''
        assert self.type is other.__class__ or not issubclass(other.__class__,QuantumNumber)
        contents=np.asarray(other)[:,np.newaxis]-self.contents
        for i,t in enumerate(self.type.periods):
            if t is not None: contents[i,:]%=t
        result=QuantumNumbers.load(self.type,(contents,self.indptr),protocal=QuantumNumbers.INDPTR)
        return result

    def subset(self,targets=()):
        '''
        A subsets of the collection of the quantum numbers.
        Parameters:
            targets: list of QuantumNumber, optional
                The quantum numbers of the subset.
        Returns: QuantumNumbers
            The subset.
        '''
        indices=np.array([self.index(target) for target in targets])
        return QuantumNumbers.load(self.type,(self.contents[:,indices],self.indptr[indices+1]-self.indptr[indices]),protocal=QuantumNumbers.COUNT)

    def subslice(self,targets=()):
        '''
        The subslice with the corresponding quantum numbers in targets.
        Parameters:
            targets: list of QuantumNumber, optional
                The quantum numbers whose slices wanted to be extracted.
        Returns: list of integer
            The subslice.
        '''
        result=[]
        for target in targets:
            index=self.index(target)
            result.append(xrange(self.indptr[index],self.indptr[index+1]))
        return np.concatenate(result)

    def expansion(self,targets=None):
        '''
        The expansion of the quantum number collection.
        Parameters:
            targets: list of QuantumNumber, optional
                The quantum numbers to be expanded.
        Returns: list of QuantumNumber
            The expanded quantum numbers.
        '''
        if targets is None:
            return np.repeat(self.contents,self.indptr[1:]-self.indptr[:-1],axis=1)
        else:
            indices=np.array([self.index(target) for target in targets])
            return np.repeat(self.contents[:,indices],self.indptr[indices+1]-self.indptr[indices])

    def permutation(self,targets=None):
        '''
        The permutation of the current order (ordered quantum numbers) with respect to the ordinary numpy.kron order (disordered quantum numbers).
        Parameters:
            targets: list of QuantumNumber, optional
                The quantum numbers whose permutation wanted to be extracted.
        Returns: list of integer
            The permutation array.
        '''
        if id(self) in QuantumNumbers.history:
            if targets is None:
                result=QuantumNumbers.history[id(self)]
            else:
                result=QuantumNumbers.history[id(self)][self.subslice(targets)]
        else:
            if targets is None:
                result=np.array(xrange(self.len))
            else:
                indices=[self.index(target) for target in targets]
                result=np.concatenate([xrange(self.indptr[index],self.indptr[index+1]) for index in indices])
        return result

    def kron(self,other,action='+',history=False):
        '''
        Tensor dot of two quantum number collections.
        Parameters:
            self,other: QuantumNumbers
                The quantum number collections to be tensor dotted.
            action: '+' or '-'
                When '+', the elements from self and other are added;
                Wehn '-', the elements from self and other are subtracted.
            history: logical, optional
                When True, the historical information of the tensor dot process will be recorded.
                Otherwise not.
        '''
        assert action in ('+-')
        if len(other)==0:
            return copy(self)
        elif len(self)==0:
            if action=='+':
                return copy(other)
            else:
                return -other
        else:
            A,B=self.expansion(),other.expansion()
            if action=='+':
                data=np.repeat(A,len(other),axis=1)+np.tile(B,len(self))
            else:
                data=np.repeat(A,len(other),axis=1)-np.tile(B,len(self))
            return QuantumNumbers.load(self.type,data,protocal=QuantumNumbers.FULL,history=history)

    def reorder(self,array,axes=None,targets=None):
        '''
        Recorder the axes of an array from the ordinary numpy.kron order to the correct quantum number collection order.
        Parameters:
            array: ndarray-like
                The original array in the ordinary numpy.kron order.
            axes: list of integer, optional
                The axes of the array to be reordered.
            targets: list of QuantumNumber, optional
                When its length is nonzero, some sub slices of the array tagged by the quantum numbers in it will be extracted.
        Returns: ndarray-like
            The axes-reordered array.
        '''
        result=array
        permutation=self.permutation(targets)
        axes=xrange(array.ndim) if axes is None else axes
        for axis in axes:
            assert self.n==result.shape[axis]
            temp=[slice(None,None,None)]*array.ndim
            temp[axis]=permutation
            result=result[tuple(temp)]
        return result
