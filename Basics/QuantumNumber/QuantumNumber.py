'''
Quantum number, including:
1) classes: QuantumNumber, QuantumNumbers
'''

from collections import namedtuple
from copy import copy,deepcopy
from fpermutation import fpermutation
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

    @classmethod
    def regularization(cls,array):
        '''
        Regularize the elements of an array so that it can represent quantum numbers.
        Parameters:
            array: 1d/2d ndarray
                When 1d array, it is a potential representation of a quantum number;
                When 2d array, it is a potential representation of a collection of quantum numbers with the rows being the individual ones.
        Returns: 1d/2d ndarray
            The input array after the regularization.
        '''
        assert len(array)==len(cls.periods)
        if array.ndim==1:
            for i,t in enumerate(cls.periods):
                if t is not None: array[i]%=t
        elif array.ndim==2:
            for i,t in enumerate(cls.periods):
                if t is not None: array[:,i]%=t
        else:
            raise ValueError('%s regularization error: array should be 1d or 2d.'%(cls.__name__))
        return array

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
        return self.__class__(self.__class__.regularization(-np.asarray(self)))

    def __add__(self,other):
        '''
        Overloaded addition(+) operator, which supports the addition of two quantum numbers.
        '''
        if isinstance(other,QuantumNumbers):
            pass
        else:
            assert self.__class__ is other.__class__
            return self.__class__(self.__class__.regularization(np.asarray(self)+np.asarray(other)))

    def __sub__(self,other):
        '''
        Overloaded subtraction(-) operator, which supports the subtraction of two quantum numbers.
        '''
        if isinstance(other,QuantumNumbers):
            pass
        else:
            assert self.__class__ is other.__class__
            return self.__class__(self.__class__.regularization(np.asarray(self)-np.asarray(other)))

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
        form: QuantumNumbers.G or QuantumNumbers.C
            1) QuantumNumbers.G: general form
                No restriction for the contents of the collection
            2) QuantumNumbers.C: canonical form
                The contents of the collection must be arranged in a accending order with no duplicates by the rows.
        type: class
            The class of the quantum numbers contained in the collection.
        contents: 2d ndarray
            The ndarray representation of the collection with the rows representing the set of its quantum numbers.
        indptr: 1d ndarray of integers
            The index pointer array of the set of the quantum numbers.
        history: dict with each of its (key,value) pairs
            key: integer
                The id of the collection of quantum numbers whose history is recorded.
            value: 1d ndarray of integer
                The permutation to make the quantum numbers in order.
    '''
    G,C=0,1
    COUNTS,INDPTR=0,1
    history={}

    def __init__(self,form,data,protocal=COUNTS):
        '''
        Constructor, supporting the following usages:
        1) QuantumNumbers(form,(qns,counts),QuantumNumbers.COUNTS), with
            form: QuantumNumbers.G or QuantumNumbers.C
                QuantumNumbers.G for general form and QuantumNumbers.C for canonical form.
            qns: list of QuantumNumber
                The quantum numbers contained in the collection.
            counts: list of integer
                The counts of the duplicates of the quantum numbers.
        2) QuantumNumbers(form,(qns,indptr),QuantumNumbers.INDPTR), with
            form: QuantumNumbers.G or QuantumNumbers.C
                QuantumNumbers.G for general form and QuantumNumbers.C for canonical form.
            qns: list of QuantumNumber
                The quantum numbers contained in the collection.
            indptr: list of integer
                The indptr of the collection.
        3) QuantumNumbers(form,(type,contents,counts),QuantumNumbers.COUNTS), with
            form: QuantumNumbers.G or QuantumNumbers.C
                QuantumNumbers.G for general form and QuantumNumbers.C for canonical form.
            type: class
                The class of the quantum numbers contained in the collection.
            contents: 2d ndarray
                The contents of the collection.
            counts: list of integer
                The counts of the duplicates of the quantum numbers.
        4) QuantumNumbers(form,(type,contents,indptr),QuantumNumbers.INDPTR), with
            form: QuantumNumbers.G or QuantumNumbers.C
                QuantumNumbers.G for general form and QuantumNumbers.C for canonical form.
            type: class
                The class of the quantum numbers contained in the collection.
            contents: 2d ndarray
                The contents of the collection.
            indptr: list of integer
                The indptr of the collection.
        '''
        assert form in (QuantumNumbers.G,QuantumNumbers.C) and len(data) in (2,3) and protocal in (QuantumNumbers.COUNTS,QuantumNumbers.INDPTR)
        self.form=form
        if protocal==QuantumNumbers.COUNTS:
            if len(data)==2:
                self.type=next(iter(data[0])).__class__
                self.contents=np.array(data[0])
                counts=data[1]
            else:
                self.type=data[0]
                self.contents=data[1]
                counts=data[2]
            assert np.all(counts>=0)
            self.indptr=np.concatenate(([0],np.cumsum(counts)))
            #self.indptr=np.zeros(len(counts)+1,dtype=np.int64)
            #for i,count in enumerate(counts):
            #    assert count>0
            #    self.indptr[i+1]=self.indptr[i]+count
        else:
            if len(data)==2:
                self.type=next(iter(data[0])).__class__
                self.contents=np.array(data[0])
                self.indptr=data[1]
            else:
                self.type=data[0]
                self.contents=data[1]
                self.indptr=data[2]
        assert self.contents.ndim==2 and self.indptr.ndim==1

    def sort(self,history=False):
        '''
        Sort the contents of the collection and convert it to the canonical form.
        Parameters:
            history: logical, optional
                When True, the permutation to make the quantum numbers in order will be recorded.
        '''
        #import time
        #t1=time.time()
        self.form=QuantumNumbers.C
        contents,counts=self.contents,self.indptr[1:]-self.indptr[:-1]
        permutation=np.lexsort(contents.T[::-1])
        mask=np.concatenate(([True],np.any(contents[permutation[1:]]!=contents[permutation[:-1]],axis=1)))
        self.contents=contents[permutation][mask]
        #t2=time.time()
        counts=counts[permutation]
        if history:
            QuantumNumbers.history[id(self)]=fpermutation(self.indptr[:-1][permutation],counts,len(self))
        #t3=time.time()
        indices=np.concatenate((np.argwhere(mask).reshape((-1)),[len(mask)]))
        self.indptr=np.zeros(self.contents.shape[0]+1,dtype=np.int64)
        for i in xrange(len(indices)-1):
            self.indptr[i+1]=self.indptr[i]+counts[indices[i]:indices[i+1]].sum()
        #t4=time.time()
        #def ptime(*arg):
        #    n=len(arg)
        #    result=['%s(%s)'%(arg[i+1]-arg[i],(arg[i+1]-arg[i])*100/(arg[-1]-arg[0])) for i in xrange(n-1)]
        #    return ','.join(result)
        #print ptime(t1,t2,t3,t4)
        return self

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        return 'QNS(%s,%s)'%(self.num,len(self))

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return '\n'.join(['QNS(','\n'.join(['%s:(%s:%s)'%(qn,self.indptr[i],self.indptr[i+1]) for i,qn in enumerate(self)]),')'])

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
        return self.contents.shape[0]

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
        for values in self.contents:
            yield self.type(values)

    def indices(self,qn):
        '''
        Find the indices of a quantum number.
        Parameters:
            qn: QuantumNumber
                The quantum number whose index is inquired.
        Returns: 1d ndarray of integers
            The corresponding indices of the quantum number.
        '''
        if self.form:
            L,R,pos=0,self.num,self.num/2
            qn,current=tuple(qn),tuple(self.contents[pos])
            while current!=qn:
                if current>qn:
                    R=pos
                else:
                    L=pos
                pos,last=(L+R)/2,pos
                if pos==last:
                    raise ValueError('QuantumNumbers index error: %s not in QuantumNumbers.'%(qn))
                current=tuple(self.contents[pos])
            else:
                return pos
        else:
            return np.argwhere(np.all(self.contents==np.asarray(qn),axis=1)).reshape((-1))

    def __contains__(self,qn):
        '''
        Judge whether a quantum number is contained in the collection.
        '''
        try:
            self.indices(qn)
            return True
        except ValueError:
            return False

    def __neg__(self):
        '''
        Overloaded negative(-) operator.
        '''
        contents=self.type.regularization(-self.contents)[::-1]
        indptr=np.concatenate(([0],np.cumsum((self.indptr[:-1]-self.indptr[1:])[::-1])))
        return QuantumNumbers(self.form,(self.type,contents,indptr),protocal=QuantumNumbers.INDPTR)

    def __add__(self,other):
        '''
        Overloaded addition(+) operator.
        '''
        assert self.type is other.__class__ or not issubclass(other.__class__,QuantumNumber)
        contents=self.type.regularization(self.contents+np.asarray(other)[np.newaxis,:])
        return QuantumNumbers(self.form,(self.type,contents,self.indptr),protocal=QuantumNumbers.INDPTR)

    def __sub__(self,other):
        '''
        Overloaded subtraction(-) operator.
        '''
        assert self.type is other.__class__ or not issubclass(other.__class__,QuantumNumber)
        contents=self.type.regularization(self.contents-np.asarray(other)[np.newaxis,:])
        return QuantumNumbers(self.form,(self.type,contents,self.indptr),protocal=QuantumNumbers.INDPTR)

    def subset(self,targets=()):
        '''
        A subsets of the collection of the quantum numbers.
        Parameters:
            targets: list of QuantumNumber, optional
                The quantum numbers of the subset.
        Returns: QuantumNumbers
            The subset.
        '''
        indices=np.concatenate([self.indices(target) for target in targets])
        return QuantumNumbers('g',(self.type,self.contents[indices],self.indptr[indices+1]-self.indptr[indices]),protocal=QuantumNumbers.COUNTS)

    def subslice(self,targets=()):
        '''
        The subslice with the corresponding quantum numbers in targets.
        Parameters:
            targets: list of QuantumNumber, optional
                The quantum numbers whose slices wanted to be extracted.
        Returns: list of integer
            The subslice.
        '''
        indices=np.concatenate([self.indices(target) for target in targets])
        return np.concatenate([xrange(self.indptr[index],self.indptr[index+1]) for index in indices])

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
            return np.repeat(self.contents,self.indptr[1:]-self.indptr[:-1],axis=0)
        else:
            indices=np.concatenate([self.indices(target) for target in targets])
            return np.repeat(self.contents[indices],self.indptr[indices+1]-self.indptr[indices],axis=0)

    def permutation(self,targets=None):
        '''
        The permutation information.
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
                result=np.array(xrange(len(self)))
            else:
                indices=np.concatenate([self.indices(target) for target in targets])
                result=np.concatenate([xrange(self.indptr[index],self.indptr[index+1]) for index in indices])
        return result

    def kron(self,other,action='+'):
        '''
        Tensor dot of two quantum number collections.
        Parameters:
            self,other: QuantumNumbers
                The quantum number collections to be tensor dotted.
            action: '+' or '-'
                When '+', the elements from self and other are added;
                Wehn '-', the elements from self and other are subtracted.
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
            import time
            A,B=self.expansion(),other.contents
            if action=='+':
                contents=np.repeat(A,len(B),axis=0)+np.tile(B.T,len(A)).T
            else:
                contents=np.repeat(A,len(B),axis=0)-np.tile(B.T,len(A)).T
            counts=np.repeat(other.indptr[1:]-other.indptr[:-1],len(A))
            return QuantumNumbers(QuantumNumbers.G,(self.type,contents,counts),protocal=QuantumNumbers.COUNTS)

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
