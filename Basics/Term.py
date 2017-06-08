'''
=================
Term descriptions
=================

This module defines the way to desribe a term and a set of terms of the Hamiltonian, including:
    * classes: Term and Terms
'''

__all__=['Term','Terms']

from numpy import complex128
from ..Misc import Arithmetic
from copy import copy

class Term(Arithmetic,object):
    '''
    This class is the base class for all kinds of terms contained in a Hamiltonian.

    Attributes
    ----------
    id : string
        The specific id of the term.
    mode : string
        The type of the term.
    value : scalar of 1D array-like of float, complex
        The overall coefficient(s) of the term.
    modulate : function
        A function used to alter the value of the term.
    '''

    def __init__(self,id,mode,value,modulate=None):
        '''
        Constructor.

        Parameters
        ----------
        id : string
            The specific id of the term.
        mode : string
            The type of the term.
        value : scalar of 1D array-like of float, complex
            The overall coefficient(s) of the term.
        modulate : function, optional
            A function used to alter the value of the term.
        '''
        self.id=id
        self.mode=mode
        self.value=value
        if modulate is not None:
            self.modulate=modulate if callable(modulate) else lambda **karg: karg.get(self.id,'None')

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

    def replace(self,**karg):
        '''
        Replace 
        '''
        result=copy(self)
        assert all(key in self.__dict__ for key in karg)
        result.__dict__.update(**karg)
        return result

class Terms(Arithmetic,list):
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
        Overloaded operator(+), which supports the left addition of a Terms instance with a Term/Terms instance.
        '''
        result=copy(self)
        if issubclass(other.__class__,Term):
            result.append(other)
        elif issubclass(other.__class__,Terms):
            result.extend(other)
        else:
            raise ValueError("%s '+' error: the other parameter must be an instance of Term's or Terms's subclasses."%self.__class__.__name__)
        return result

    __iadd__=__add__

    def __mul__(self,other):
        '''
        Overloaded operator(*), which supports the left multiplication with a scalar.
        '''
        result=list.__new__(self.__class__)
        for obj in self:
            result.append(obj*other)
        return result

    __imul__=__mul__

    def operators(self,bond,config,table=None,dtype=complex128):
        '''
        This method returns all the desired operators which are described those terms in self.

        Parameters
        ----------
        bond : Bond
            The bond on which the terms are defined.
        config : IDFConfig
            The configuration of degrees of freedom.
        table : Table, optional
            The index-sequence table.
        dtype : dtype,optional
            The data type of the coefficient of the returned operators.

        Returns
        -------
        Operators
            All the desired operators with non-zero coeffcients.

        Notes
        -----
        To use this method, the subclass must override it.
        '''
        raise ValueError('%s operators error: it is not implemented.'%self.__class__.__name__)
