'''
'''

from collections import OrderedDict

__all__=[]

class QuantumNumber(tuple):
    '''
    '''
    def __new__(cls,para):
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
        return ''.join(['QuantumNumber(','=%r, '.join(self.__dict__.keys()),'=%r)'])%self

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

    #def direct_sum(self,other):
    #    result=super(QuantumNumber,self).__add__(other)
    #    result.__dict__.update(self.__dict__)
    #    result.__dict__.update(other.__dict__)

class U1(QuantumNumber):
    pass
