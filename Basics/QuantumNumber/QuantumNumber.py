'''
Quantum number, including:
1) classes: QuantumNumber, U1, QuantumNumberSet
'''

from collections import OrderedDict

__all__=['QuantumNumber','U1','QuantumNumberSet']

class QuantumNumber(tuple):
    '''
    '''
    def __new__(cls,para):
        '''
        '''
        if not isinstance(para,OrderedDict):
            raise ValueError('QuantumNumber __new__ error: the parameter must be an OrderedDict.')
        self=super(QuantumNumber,cls).__new__(cls,para.values())
        self.__dict__=para
        return self

    def __copy__(self):
        '''
        Copy.
        '''
        return self.replace(**self.__dict__)

    def __deepcopy__(self,memo):
        '''
        Deep copy.
        '''
        return self.replace(**self.__dict__)

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        return ''.join(['%s('%self.__class__.__name__,'=%r, '.join(self.__dict__.keys()),'=%r)'])%self

    def __add__(self):
        '''
        '''
        raise NotImplementedError()

    def __mul__(self):
        '''
        '''
        raise NotImplementedError()

    def __rmul__(self,other):
        '''
        '''
        return self.__mul__(other)

    def replace(self,**karg):
        '''
        Return a new QuantumNumber object with specified fields replaced with new values.
        '''
        result=tuple.__new__(QuantumNumber,map(karg.pop,self.__dict__.keys(),self))
        if karg:
            raise ValueError('QuantumNumber replace error: it got unexpected field names: %r'%karg.keys())
        result.__dict__=OrderedDict()
        for key,value in zip(self.__dict__.keys(),result):
            result.__dict__[key]=value
        return result

    def direct_sum(self,other):
        '''
        '''
        result=tuple.__new__(self.__class__,tuple.__add__(self,other))
        result.__dict__=OrderedDict()
        result.__dict__.update(self.__dict__)
        result.__dict__.update(other.__dict__)
        return result

class U1(QuantumNumber):
    '''
    '''
    def __add__(self,other):
        '''
        '''
        temp=OrderedDict()
        for key in self.__dict__:
            temp[key]=getattr(self,key)+getattr(other,key)
        return U1(temp)

    def __mul__(self,other):
        '''
        '''
        temp=OrderedDict()
        for key in self.__dict__:
            temp[key]=getattr(self,key)*other
        return U1(temp)

class QuantumNumberSet(set):
    '''
    '''
    def __init__(self):
        '''
        '''
        self.edges={}
        

    def __add__(self,other):
        '''
        '''
        pass

    def __radd__(self,other):
        pass
        '''
        '''
        pass

    def __iadd__(self,other):
        '''
        '''
        pass

    def __mul__(self,other):
        '''
        '''
        pass

    def __rmul__(self,other):
        '''
        '''
        return self.__mul__(other)

    def __imul__(self,other):
        '''
        '''
        pass

class QuantumNumberFlow(dict):
    pass
