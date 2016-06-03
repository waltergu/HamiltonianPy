'''
Quantum number, including:
1) classes: QuantumNumber, QuantumNumberCollection
'''

from collections import OrderedDict

__all__=['QuantumNumber','QuantumNumberCollection']

class QuantumNumber(tuple):
    '''
    Quantum number.
    It is a generalization of namedtuple.
    Attributes:
        values: OrderedDict
            The values of the quantum number.
            The keys correspond to the names.
        types: OrderedDict
            The types of the quantum number.
            The keys correspond to the names.
    '''
    U1,Z2=('U1','Z2')

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
                    type: 'U1' or 'Z2'
                        The type of the quantum number.
        '''
        values,types=OrderedDict(),OrderedDict()
        for name,value,type in para:
            values[name]=value
            types[name]=type
        self=super(QuantumNumber,cls).__new__(cls,values.values())
        self.values=values
        self.types=types
        return self

    def __getattr__(self,key):
        '''
        Overloaded dot(.) operator.
        '''
        return self.values[key]

    def __copy__(self):
        '''
        Copy.
        '''
        return self.replace(**self.values)

    def __deepcopy__(self,memo):
        '''
        Deep copy.
        '''
        return self.replace(**self.values)

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        temp=[]
        for key in self.values:
            temp.append(key)
            temp.append(self.values[key])
            temp.append(self.types[key])
        return ''.join(['%s('%self.__class__.__name__,','.join(['%s=%r(%s)']*len(self)),')'])%tuple(temp)

    def __add__(self,other):
        '''
        Overloaded addition(+) operator, which supports the addition of two quantum numbers.
        '''
        temp=[]
        for key in self.values:
            if self.types[key]!=other.types[key]:
                raise ValueError("QuantumNumber '+' error: different types of quantum numbers cannot be added.")
            type=self.types[key]
            if type==self.U1:
                value=getattr(self,key)+getattr(other,key)
            elif type==self.Z2:
                if getattr(self,key)!=getattr(other,key):
                    raise ValueError("QuantumNumber '+' error: only equal Z2 quantum numbers can be added.")
                value=getattr(self,key)
            temp.append((key,value,type))
        return QuantumNumber(temp)

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
        result=tuple.__new__(QuantumNumber,map(karg.pop,self.values.keys(),self))
        if karg:
            raise ValueError('QuantumNumber replace error: it got unexpected field names: %r'%karg.keys())
        result.values=OrderedDict()
        for key,value in zip(self.values.keys(),result):
            result.values[key]=value
        result.types=self.types
        return result

    def direct_sum(self,other):
        '''
        Direct sum of two quantum numbers.
        '''
        result=tuple.__new__(self.__class__,tuple.__add__(self,other))
        result.values=OrderedDict()
        result.values.update(self.values)
        result.values.update(other.values)
        result.types=OrderedDict()
        result.types.update(self.types)
        result.types.update(other.types)
        return result

class QuantumNumberCollection(set):
    '''
    Quantum number collection.
    Attributes:
        map: dict
            The history information of quantum numbers.
    '''
    trace_history=True

    def __new__(cls,para=[],map={}):
        '''
        Constructor.
        Parameters:
            para: list of QuantumNumber
                The quantum numbers.
            map: dict
                The history information of quantum numbers.
        '''
        self=set.__new__(cls,para)
        self.map=map if QuantumNumberCollection.trace_history else {}
        return self

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        pass

    def __add__(self,other):
        '''
        Overloaded addition(+) operator.
        '''
        temp,buff=[],{}
        other=[other] if isinstance(other,QuantumNumer) else other
        for qn1 in self:
            for qn2 in other:
                sum=qn1+qn2
                if sum not in buff:
                    temp.append(sum)
                    buff[sum]=[]
                buff[sum].append((qn1,qn2))
        return QuantumNumberCollection(temp,buff)

    def __radd__(self,other):
        '''
        Overloaded addition(+) operator.
        '''
        return self.__add__(other)
