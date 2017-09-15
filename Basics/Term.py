'''
=================
Term descriptions
=================

This module defines the way to describe a term of the Hamiltonian, including:
    * classes: Term
'''

__all__=['Term']

from numpy import ndarray,complex128
from Utilities import Arithmetic
from collections import Iterable
from copy import copy

class Term(Arithmetic):
    '''
    This class is the base class for all kinds of terms contained in a Hamiltonian.

    Attributes
    ----------
    id : string
        The specific id of the term.
    value : scalar of 1d array-like of float/complex
        The overall coefficient(s) of the term.
    modulate : callable
        A function used to alter the value of the term.
    '''

    def __init__(self,id,value,modulate=None):
        '''
        Constructor.

        Parameters
        ----------
        id : string
            The specific id of the term.
        value : scalar of 1D array-like of float, complex
            The overall coefficient(s) of the term.
        modulate : callable, optional
            A function used to alter the value of the term.
        '''
        self.id=id
        self.value=value
        self.modulate=modulate if callable(modulate) or modulate is None else lambda **karg: karg.get(self.id,None)

    def __mul__(self,other):
        '''
        Overloaded operator(*) which supports the left multiplication with a scalar.
        '''
        result=copy(self)
        if isinstance(result.value,ndarray) or not isinstance(result.value,Iterable):
            result.value*=other
        else:
            result.value=type(result.value)([value*other for value in result.value])
        if result.modulate is not None:
            result.modulate=lambda *arg,**karg: self.modulate(*arg,**karg)*other if self.modulate(*arg,**karg) is not None else None
        return result

    def replace(self,**karg):
        '''
        Replace some attributes of the term.
        '''
        result=copy(self)
        assert all(key in self.__dict__ for key in karg)
        result.__dict__.update(**karg)
        return result

    def operators(self,bond,config,table=None,dtype=complex128,**karg):
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
        dtype : np.complex128, np.float64, optional
            The data type of the coefficient of the returned operators.

        Returns
        -------
        Operators
            All the desired operators with non-zero coefficients.

        Notes
        -----
        To use this method, the subclass must override it.
        '''
        raise ValueError('%s operators error: it is not implemented.'%self.__class__.__name__)

    def strrep(self,bond,config):
        '''
        The string representation of the term on a bond.

        Parameters
        ----------
        bond : Bond
            The bond where the term is to be represented.
        config : IDFConfig
            The configuration of internal degrees of freedom.

        Returns
        -------
        str
            The string representation of the term on the bond.
        '''
        raise NotImplementedError('%s strreps error: it is not implemented.'%self.__class__.__name__)
