'''
Quantum number, including:
1) classes: QuantumNumber, QuantumNumberCollection
'''

from collections import namedtuple,OrderedDict
from numpy import concatenate
from copy import copy,deepcopy

__all__=['QuantumNumber','QuantumNumberCollection']

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
                types.append(type)
        self=super(QuantumNumber,cls).__new__(cls,values)
        self.names=tuple(names)
        self.types=tuple(types)
        return self

    def __getattr__(self,key):
        '''
        Overloaded dot(.) operator.
        '''
        return self[self.names.index(key)]

    def __copy__(self):
        '''
        Copy.
        '''
        return self.replace(**{key:value for key,value in zip(self.names,self)})

    def __deepcopy__(self,memo):
        '''
        Deep copy.
        '''
        return self.replace(**{key:value for key,value in zip(self.names,self)})

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

    def __add__(self,other):
        '''
        Overloaded addition(+) operator, which supports the addition of two quantum numbers.
        '''
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
        history: dict of QuantumNumberHistory
            The historical information of the tensordot of two quantum number collections.
    '''
    history={}

    def __init__(self,para=None):
        '''
        Constructor.
        Parameters:
            para: list of 2-tuple
                tuple[0]: QuantumNumber
                    The quantum number.
                tuple[1]: integer or slice
                    1) integer:
                        The number of the duplicates of the quantum number
                    2) slice:
                        The corresponding slice of the quantum number.
        '''
        OrderedDict.__init__(self)
        self.id=id(self)
        count=0
        if para is not None:
            for (key,value) in para:
                if isinstance(value,int) or isinstance(value,long):
                    self[key]=slice(count,count+value)
                    count+=value
                elif isinstance(value,slice):
                    self[key]=value
                    count+=value.stop-value.start
                else:
                    raise ValueError('QuantumNumberCollection construction error: improper parameter(%s).'%(value.__class__.__name__))
        self.n=count

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        return ''.join(['QNC(',','.join(['%s:(%s:%s)'%(qn,value.start,value.stop) for qn,value in self.items()]),')'])

    @property
    def permutation(self):
        '''
        The permutation of the current order with respect to the direct product order.
        '''
        if self.id in self.history:
            return [i for value in self.history[self.id].values() for slice in value.slices for i in xrange(slice.start,slice.stop)]
        else:
            return list(xrange(self.n))

    def tensordot(self,other,history=False):
        '''
        Tensor dot of two quantum number collections.
        Parameters:
            self,other: QuantumNumberCollection
                The quantum number collections to be tensor dotted.
            history: logical
                When True, the historical information of the tensor dot process will be recorded.
                Otherwise not.
        '''
        if len(self)==0:
            return copy(other)
        elif len(other)==0:
            return copy(self)
        else:
            contents=OrderedDict()
            if history:
                record=OrderedDict()
                for qn1,v1 in self.items():
                    for qn2,v2 in other.items():
                        sum=qn1+qn2
                        if sum not in contents:
                            contents[sum]=0
                            record[sum]=QuantumNumberHistory(pairs=[],slices=[])
                        contents[sum]+=(v2.stop-v2.start)*(v1.stop-v1.start)
                        record[sum].pairs.append((qn1,qn2))
                        for i in xrange(v1.start,v1.stop):
                            record[sum].slices.append(slice(i*other.n+v2.start,i*other.n+v2.stop))
                result=QuantumNumberCollection(contents.iteritems())
                QuantumNumberCollection.history[result.id]=record
            else:
                for qn1,v1 in self.items():
                    for qn2,v2 in other.items():
                        sum=qn1+qn2
                        contents[sum]=contents.get(sum,0)+(v2.stop-v2.start)*(v1.stop-v1.start)
                result=QuantumNumberCollection(contents.iteritems())
            return result

    def clear_history(self):
        '''
        Clear the historical information of the quantum number collection.
        '''
        QuantumNumberCollection.history.pop(self.id,None)
