'''
Term and TermList
'''

__all__=['Term','TermList']

from copy import copy

class Term(object):
    '''
    This class is the base class for all kinds of terms contained in a Hamiltonian.
    Attributes:
        id: string
            The specific id of the term.
        mode: string
            The type of the term.
        value: scalar of 1D array-like of float, complex
            The overall coefficient(s) of the term.
        modulate: function
            A function used to alter the value of the term.
    '''
    def __init__(self,id,mode,value,modulate=None):
        self.id=id
        self.mode=mode
        self.value=value
        if not modulate is None:
            self.modulate=modulate

    def __mul__(self,other):
        '''
        Overloaded operator(*) which supports the left multiplication with a scalar.
        '''
        result=copy(self)
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

class TermList(list):
    '''
    This class packs several instances of Term's subclasses as a whole for convenience.
    '''
    def __new__(cls,*arg):
        '''
        Constructor.
        '''
        self=list.__new__(cls)
        for obj in arg:
            if issubclass(obj.__class__,Term):
                self.append(obj)
            else:
                raise ValueError("%s init error: the input argument should be instances of Term's subclasses."%self.__class__.__name__)
        return self

    def __init__(self,*arg):
        '''
        Constructor. It is needed to change the interface of list construction.
        '''
        pass

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return '\n'.join(['Node[%s]:%s'%(i,obj) for i,obj in enumerate(self)])

    def __add__(self,other):
        '''
        Overloaded operator(+), which supports the left addition of a TermList instance with a Term/TermList instance.
        '''
        result=copy(self)
        if issubclass(other.__class__,Term):
            result.append(other)
        elif issubclass(other.__class__,TermList):
            result.extend(other)
        else:
            raise ValueError("%s '+' error: the other parameter must be an instance of Term's or TermList's subclasses."%self.__class__.__name__)
        return result

    def __radd__(self,other):
        '''
        Overloaded operator(+), which supports the right addition of a TermList instance with a Term/TermList instance.
        '''
        return self.__add__(other)

    def __mul__(self,other):
        '''
        Overloaded operator(*), which supports the left multiplication with a scalar.
        '''
        result=list.__new__(self.__class__)
        for obj in self:
            result.append(obj*other)
        return result

    def __rmul__(self,other):
        '''
        Overloaded operator(*), which supports the right multiplication with a scalar.
        '''
        return self.__mul__(other)

    def operators(self,*arg,**karg):
        '''
        This method returns all the desired operators which are described those terms in self.
        Note: To use this method, the subclass must override it.
        '''
        raise ValueError('%s operators error: it is not implemented.'%self.__class__.__name__)
