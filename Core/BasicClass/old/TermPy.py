'''
Term.
'''
from numpy import *
from copy import deepcopy
class Term(object):
    '''
    This class is the base class for all kinds of terms contained in a Hamiltonian.
    Attributes:
        mode: string
            The type of the term.
        tag: string
            The tag specifying the term used for dictionary lookup.
        value: scalar of 1D array-like of float, complex
            The overall coefficient(s) of the term.
        modulate: function
            A function used to alter the value of the term.
    '''
    def __init__(self,mode,tag,value,modulate=None):
        self.mode=mode
        self.tag=tag
        self.value=value
        if not modulate is None:
            self.modulate=modulate

    def __mul__(self,other):
        '''
        Overloaded operator(*) which supports the left multiplication with a scalar.
        '''
        result=deepcopy(self)
        if isinstance(result.value,list):
            result.value=[value*other for value in result.value]
        else:
            result.value*=other
            if hasattr(self,'modulate'):
                result.modulate=lambda *arg,**karg: self.modulate(*arg,**karg)*other if self.modulate(*arg,**karg) is not None else None
        return result
 
    def __rmul__(self,other):
        '''
        Overloaded operator(*) which supports the right multiplication with a scalar.
        '''
        return self.__mul__(other)
