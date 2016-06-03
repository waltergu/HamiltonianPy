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
    repr_forms=['FULL','SIMPLE','ORIGINAL']
    repr_form=repr_forms[0]

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
            for key in self.values:
                temp.append(key)
                temp.append(self.values[key])
                temp.append(self.types[key])
            return ''.join([self.__class__.__name__,'(',','.join(['%s=%r(%s)']*len(self)),')'])%tuple(temp)
        elif self.repr_form==self.repr_forms[1]:
            return ''.join([self.__class__.__name__,'(',','.join(['%s']*len(self)),')'])%self
        else:
            return tuple.__repr__(self)

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
                value=(getattr(self,key)+getattr(other,key))%2
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

class QuantumNumberCollection(OrderedDict):
    '''
    Quantum number collection.
    Attributes:
        map: dict
            The history information of quantum numbers.
    '''
    trace_history=True

    def __init__(self,para=[],map={}):
        '''
        Constructor.
        Parameters:
            para: list of QuantumNumber
                The quantum numbers.
            map: dict
                The history information of quantum numbers.
        '''
        OrderedDict.__init__(self)
        for key in para:
            self[key]=None
        self.map=map if QuantumNumberCollection.trace_history else {}        

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        return ''.join(['QuantumNumberCollection(',','.join([str(qn) for qn in self]),')'])

    def __add__(self,other):
        '''
        Overloaded addition(+) operator.
        '''
        temp,buff=[],OrderedDict()
        other=[other] if isinstance(other,QuantumNumber) else other
        if len(self)==0:
            temp=other
        elif len(other)==0:
            temp=self
        else:
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
