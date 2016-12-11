'''
Quantum number, including:
1) classes: QuantumNumber, QuantumNumberHistory, QuantumNumberCollection
'''

from collections import namedtuple,OrderedDict
from copy import copy,deepcopy
import numpy as np
import scipy.sparse as sp

__all__=['QuantumNumber','QuantumNumberHistory','QuantumNumberCollection']

class QuantumNumber(tuple):
    '''
    Quantum number.
    It is a generalization of namedtuple.
    Attributes:
        names: list of string
            The names of the quantum number.
        types: list of integer
            The types of the quantum number.
    '''
    repr_forms=['FULL','SIMPLE','ORIGINAL']
    repr_form='FULL'

    def __new__(cls,para):
        '''
        Constructor.
        Parameters:
            para: list of 3-tuple.
                For each tuple, it should be in the form of (name,value,type):
                    name: string
                        The name of the quantum number.
                    value: int
                        The value of the quantum number.
                    type: 'U1', 'Z2' or any integer
                        The type of the quantum number.
        '''
        names,values,types=[],[],[]
        for name,value,type in para:
            names.append(name)
            values.append(value)
            if type=='U1':
                types.append('U1')
            elif type=='Z2':
                types.append(2)
            else:
                assert (isinstance(type,long) or isinstance(type,int)) and type>0
                types.append(type)
        self=super(QuantumNumber,cls).__new__(cls,values)
        self.names=tuple(names)
        self.types=tuple(types)
        return self

    def __getnewargs__(self):
        '''
        Return the arguments for QuantumNumber.__new__, required by copy and pickle.
        '''
        return (zip(self.names,self,self.types),)

    def __getstate__(self):
        '''
        Since QuantumNumber.__new__ constructs everything, self.__dict__ can be omitted for copy and pickle.
        '''
        pass

    def __getattr__(self,key):
        '''
        Overloaded dot(.) operator.
        '''
        try:
            return self[self.names.index(key)]
        except ValueError:
            raise AttributeError()

    @classmethod
    def set_repr_form(cls,para):
        '''
        Set the form __repr__ will use.
        '''
        if para in cls.repr_forms:
            cls.repr_form=para
        else:
            raise ValueError('QuantumNumber set_repr_form error: %s is not one of the supported forms (%s).'%(para,', '.join(cls.repr_forms)))

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        if self.repr_form==self.repr_forms[0]:
            temp=[]
            for name,value,type in zip(self.names,self,self.types):
                temp.append(name)
                temp.append(value)
                temp.append('U1' if type=='U1' else 'Z%s'%(type))
            return ''.join(['QN','(',','.join(['%s=%r(%s)']*len(self)),')'])%tuple(temp)
        elif self.repr_form==self.repr_forms[1]:
            return ''.join(['QN','(',','.join(['%s']*len(self)),')'])%self
        else:
            return tuple.__repr__(self)

    def __neg__(self):
        '''
        Overloaded negative(-) operator.
        '''
        values=[]
        for n,v,t in zip(self.names,self,self.types):
            if t=='U1':
                values.append(-v)
            else:
                values.append((t-v)%t)
        result=tuple.__new__(self.__class__,values)
        result.names=self.names
        result.types=self.types
        return result

    def __add__(self,other):
        '''
        Overloaded addition(+) operator, which supports the addition of two quantum numbers.
        '''
        if other==0:
            return self
        else:
            values=[]
            for n1,v1,t1,n2,v2,t2 in zip(self.names,self,self.types,other.names,other,other.types):
                assert n1==n2
                assert t1==t2
                if t1=='U1':
                    values.append(v1+v2)
                else:
                    values.append((v1+v2)%t1)
            result=tuple.__new__(self.__class__,values)
            result.names=self.names
            result.types=self.types
            return result

    def __radd__(self,other):
        '''
        Overloaded addition(+) operator, which supports the addition of two quantum numbers.
        '''
        return self.__add__(other)

    def __sub__(self,other):
        '''
        Overloaded addition(-) operator, which supports the subtraction of two quantum numbers.
        '''
        values=[]
        for n1,v1,t1,n2,v2,t2 in zip(self.names,self,self.types,other.names,other,other.types):
            assert n1==n2
            assert t1==t2
            if t1=='U1':
                values.append(v1-v2)
            else:
                values.append((v1-v2)%t1)
        result=tuple.__new__(self.__class__,values)
        result.names=self.names
        result.types=self.types
        return result

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
        result=tuple.__new__(QuantumNumber,map(karg.pop,self.names,self))
        if karg:
            raise ValueError('QuantumNumber replace error: it got unexpected field names: %r'%karg.keys())
        result.names=self.names
        result.types=self.types
        return result

    def direct_sum(self,other):
        '''
        Direct sum of two quantum numbers.
        '''
        result=tuple.__new__(self.__class__,tuple.__add__(self,other))
        result.names=self.names+other.names
        result.types=self.types+other.types
        return result

    @property
    def zeros(self):
        '''
        Return a new quantum number with all the values equal to zero.
        '''
        return QuantumNumber([(name,0,type) for name,type in zip(self.names,self.types)])

class QuantumNumberHistory(namedtuple('QuantumNumberHistory',['pairs','slices'])):
    '''
    The historical information of a quantum number.
    Attribues:
        pairs: list of 2-tuples of QuantumNumber
            All the pairs of the addends of the quantum number.
        slices: list of slice
            The unpermutated slices of the quantum number.
    '''

QuantumNumberHistory.__new__.__defaults__=(None,)*len(QuantumNumberHistory._fields)

class QuantumNumberCollection(OrderedDict):
    '''
    Quantum number collection.
    For its (key,value) pair:
        key: QuantumNumber
            The quantum number.
        value: slice
            The corresponding slice of the quantum number.
    Attributes:
        n: integer
            The total number of quantum numbers when duplicates are counted duplicately.
        history: dict with each of its (key,value) pairs
            key: integer
                The id of the quantum number collection whose history is recorded.
            value: OrderedDict, with each of its (key,value) pairs
                key: QuantumNumber
                    The quantum number contained in the quantum number collection whose history is recorded.
                value: QuantumNumberHistory
                    The historical information.
    '''
    history={}

    def __init__(self,para=[],history=False):
        '''
        Constructor.
        Parameters:
            para: list of QuantumNumber or 2-tuple
            1) QuantumNumber:
                The quantum numbers of the quantum number collection.
            2) 2-tuple:
                tuple[0]: QuantumNumber
                    The quantum number.
                tuple[1]: integer or slice
                    1) integer:
                        The number of the duplicates of the quantum number
                    2) slice:
                        The corresponding slice of the quantum number.
            history: logical, optional
                When para is a list of QuantumNumber and history is True, the permutation information will be recorded.
        '''
        OrderedDict.__init__(self)
        temp=np.array([isinstance(qn,QuantumNumber) for qn in para])
        if all(temp):
            contents=OrderedDict()
            if history:
                record=OrderedDict()
                for i,qn in enumerate(para):
                    if qn not in contents:
                        contents[qn]=0
                        record[qn]=QuantumNumberHistory(pairs=[],slices=[])
                    contents[qn]+=1
                    record[qn].slices.append(slice(i,i+1))
                QuantumNumberCollection.history[id(self)]=record
            else:
                for qn in para:
                    contents[qn]=contents.get(qn,0)+1
            para=contents.items()
        count=0
        if all(temp) or all(~temp):
            for (key,value) in para:
                if isinstance(value,int) or isinstance(value,long):
                    assert value>=0
                    if value>0:
                        self[key]=slice(count,count+value)
                        count+=value
                elif isinstance(value,slice):
                    assert value.start<=value.stop and value.step is None
                    if value.start<value.stop:
                        self[key]=value
                        count+=value.stop-value.start
                else:
                    raise ValueError('QuantumNumberCollection construction error: improper parameter(%s).'%(value.__class__.__name__))
        else:
            raise ValueError('QuantumNumberCollection construction error: improper parameter type.')
        self.n=count

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        return ''.join(['QNC(',','.join(['%s:(%s:%s)'%(qn,value.start,value.stop) for qn,value in self.items()]),')'])

    def __neg__(self):
        '''
        Overloaded negative(-) operator.
        '''
        return QuantumNumberCollection([(-key,value) for key,value in self.iteritems()])

    def subslice(self,targets=()):
        '''
        The subslice with the corresponding quantum numbers in targets.
        Parameters:
            targets: list of QuantumNumber, optional
                The quantum numbers whose slices wanted to be extracted.
        Returns: list of integer
            The subslice.
        '''
        return [i for target in targets for i in xrange(self[target].start,self[target].stop)]

    def expansion(self,targets=None):
        '''
        The expansion of the quantum number collection.
        Parameters:
            targets: list of QuantumNumber, optional
                The quantum numbers to be expanded.
        Returns: list of QuantumNumber
            The expanded quantum numbers.
        '''
        targets=self.keys() if targets is None else targets
        return [key for key in targets for i in xrange(self[key].start,self[key].stop)]

    def permutation(self,targets=None):
        '''
        The permutation of the current order (ordered quantum numbers) with respect to the ordinary numpy.kron order (disordered quantum numbers).
        Parameters:
            targets: list of QuantumNumber, optional
                The quantum numbers whose permutation wanted to be extracted.
        Returns: list of integer
            The permutation array.
        '''
        if id(self) in self.history:
            histories=self.history[id(self)].values() if targets is None else [self.history[id(self)][target] for target in targets]
            result=[i for history in histories for slice in history.slices for i in xrange(slice.start,slice.stop)]
        else:
            result=range(self.n) if targets is None else [i for i in xrange(self[target].start,self[target].stop) for target in targets]
        return result

    def kron(self,other,action='+',history=False):
        '''
        Tensor dot of two quantum number collections.
        Parameters:
            self,other: QuantumNumberCollection
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
                return QuantumNumberCollection([(-key,value) for key,value in other.items()])
        else:
            contents=OrderedDict()
            if history:
                record=OrderedDict()
                for qn1,v1 in self.items():
                    for qn2,v2 in other.items():
                        sum=qn1+qn2 if action=='+' else qn1-qn2
                        if sum not in contents:
                            contents[sum]=0
                            record[sum]=QuantumNumberHistory(pairs=[],slices=[])
                        contents[sum]+=(v2.stop-v2.start)*(v1.stop-v1.start)
                        record[sum].pairs.append((qn1,qn2))
                        for i in xrange(v1.start,v1.stop):
                            record[sum].slices.append(slice(i*other.n+v2.start,i*other.n+v2.stop))
                result=QuantumNumberCollection(contents.items())
                QuantumNumberCollection.history[id(result)]=record
            else:
                for qn1,v1 in self.items():
                    for qn2,v2 in other.items():
                        sum=qn1+qn2 if action=='+' else qn1-qn2
                        contents[sum]=contents.get(sum,0)+(v2.stop-v2.start)*(v1.stop-v1.start)
                result=QuantumNumberCollection(contents.items())
            return result

    def subset(self,targets=()):
        '''
        A subsets of the quantum number collection.
        Parameters:
            targets: list of QuantumNumber, optional
                The quantum numbers of the subset.
        Returns: QuantumNumberCollection
            The subset.
        '''
        return QuantumNumberCollection([(target,self[target].stop-self[target].start) for target in targets])

    def pairs(self,qn):
        '''
        The historical pairs of the addends of a quantum number in the quantum number collection.
        Parameters:
            qn: QuantumNumber
                The quantum number whose historical pairs of the addends are requested.
        Returns: list of 2-tuple of QuantumNumber
            The historical pairs of the addends of the quantum number.
        '''
        return self.history[id(self)][qn].pairs

    @staticmethod
    def clear_history(*arg):
        '''
        Clear the historical information of the quantum number collection.
        '''
        if len(arg)==0:
            QuantumNumberCollection.history.clear()
        else:
            for qnc in arg:
                if isinstance(qnc,QuantumNumberCollection):
                    QuantumNumberCollection.history.pop(id(qnc),None)

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
