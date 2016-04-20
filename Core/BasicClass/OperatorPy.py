'''
Operator and OperatorCollection.
'''

__all__=['Operator','OperatorCollection']

from ConstantPy import *
from copy import copy

class Operator(object):
    '''
    This class is the base class of all types of operators.
    Attributes:
        id: anyhashable object
            The unique id this operator has.
            Two operators with the same id can be combined.
        value: number
            The overall factor of the operator.
    '''

    def set_id(self):
        '''
        Set the unique id of this operator.
        Note: this method must be overridden by its child class if it is to be used.
        '''
        raise ValueError("Operator set_id error: this method must be overridden by child classes.")

    def __add__(self,other):
        '''
        Overloaded operator(+), which supports the addition of an Operator instance with an Operator/OperatorCollection instance.
        '''
        if isinstance(other,Operator):
            result=OperatorCollection()
            if self.id==other.id:
                value=self.value+other.value
                if abs(value)>RZERO:
                    temp=copy(self)
                    temp.value=value
                    result[self.id]=temp
            else:
                result[self.id]=self
                result[other.id]=other
        elif isinstance(other,OperatorCollection):
            result=other.__add__(self)
        else:
            raise ValueError("Operator '+' error: the 'other' parameter must be of class Operator or OperatorCollection.")
        return result

    def __mul__(self,other):
        '''
        Overloaded operator(*), which supports the multiplication of an Operator instance with a scalar.
        '''
        result=copy(self)
        result.value=self.value*other
        return result
        
    def __rmul__(self,other):
        '''
        Overloaded operator(*), which supports the multiplication of an Operator instance with a scalar.
        '''
        return self.__mul__(other)

    def __sub__(self,other):
        '''
        Overloaded operator(-), which supports the subtraction of an Operator instance with an Operator/OperatorCollection instance.
        '''
        return self+other*(-1)

    def __eq__(self,other):
        '''
        Overloaded operator(==).
        '''
        return self.id==other.id and abs(self.value-other.value)<RZERO

class OperatorCollection(dict):
    '''
    This class packs several operators as a whole for convenience.
    '''
    
    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return '\n'.join(['[%s]:%s'%(i,obj) for i,obj in enumerate(self.values())])

    def __add__(self,other):
        '''
        Overloaded operator(+), which supports the addition of an OperatorCollection instance with an Operator/OperatorCollection instance.
        '''
        return copy(self).__iadd__(other)

    def __mul__(self,other):
        '''
        Overloaded operator(*), which supports the multiplication of an OperatorCollection instance with a scalar.
        '''
        result=OperatorCollection()
        for id,obj in self.iteritems():
            result[id]=obj*other
        return result

    def __rmul__(self,other):
        '''
        Overloaded operator(*), which supports the multiplication of an OperatorCollection instance with a scalar.
        '''
        return self.__mul__(other)

    def __sub__(self,other):
        '''
        Overloaded operator(-), which supports the subtraction of an OperatorCollection instance with an Operator/OperatorCollection instance.
        '''
        return self+other*(-1)

    def __iadd__(self,other):
        '''
        Overloaded operator(+=).
        '''
        if isinstance(other,Operator):
            id=other.id
            if id in self:
                value=self[id].value+other.value
                if abs(value)>RZERO:
                    temp=copy(other)
                    temp.value=value
                    self[id]=temp
                else:
                    del self[id]                
            else:
                self[other.id]=other
        elif isinstance(other,OperatorCollection):
            for obj in other.values():
                self.__iadd__(obj)
        else:
            raise ValueError("OperatorCollection '+=' error: the 'other' parameter must be of class Operator or OperatorCollection.")
        return self

    def __isub__(self,other):
        '''
        Overloaded operator(-=).
        '''
        if isinstance(other,Operator):
            id=other.id
            if id in self:
                value=self[id].value-other.value
                if abs(value)>RZERO:
                    temp=copy(other)
                    temp.value=value
                    self[id]=temp
                else:
                    del self[id]                
            else:
                temp=copy(other)
                temp.value*=-1
                self[other.id]=temp
        elif isinstance(other,OperatorCollection):
            for obj in other.values():
                self.__isub__(obj)
        else:
            raise ValueError("OperatorCollection '-=' error: the 'other' parameter must be of class Operator or OperatorCollection.")
        return self
