'''
==========
Arithmetic
==========

This module defines the base class for those that support arithmetic operations, including:
    * classes: Arithmetic
'''

__all__=['Arithmetic']

class Arithmetic(object):
    '''
    This class defines the base class for those that support arithmetic operations(+,-,*,/,==,!=).
    To realize the full basic arithmetics, the following methods must be overloaded by its subclasses:
        * __iadd__
        * __add__
        * __imul__
        * __mul__
        * __eq__

    Notes
    -----
        * The addition('+') and multiplication('*') operations are assumed to be commutable.
        * The minus sign ('-' in the negative operator and subtraction operator) are interpreted as the multiplication by -1.0
        * The division operation is interpreted as the multiplication by the inverse of the second argument, which should be a scalar.
    '''

    def __pos__(self):
        '''
        Overloaded positive(+) operator.
        '''
        return self

    def __neg__(self):
        '''
        Overloaded negative(-) operator.
        '''
        return self*(-1)

    def __iadd__(self,other):
        '''
        Overloaded self-addition(+=) operator.
        '''
        raise NotImplementedError("%s (+=) error: not implemented."%self.__class__.__name__)

    def __add__(self,other):
        '''
        Overloaded left addition(+) operator.
        '''
        raise NotImplementedError("%s (+) error: not implemented."%self.__class__.__name__)

    def __radd__(self,other):
        '''
        Overloaded right addition(+) operator.
        '''
        return self+other

    def __isub__(self,other):
        '''
        Overloaded self-subtraction(-=) operator.
        '''
        return self.__iadd__(-other)

    def __sub__(self,other):
        '''
        Overloaded subtraction(-) operator.
        '''
        return self+other*(-1.0)

    def __imul__(self,other):
        '''
        Overloaded self-multiplication(*=) operator.
        '''
        raise NotImplementedError("%s (*=) error: not implemented."%self.__class__.__name__)

    def __mul__(self,other):
        '''
        Overloaded left multiplication(*) operator.
        '''
        raise NotImplementedError("%s (*) error: not implemented."%self.__class__.__name__)

    def __rmul__(self,other):
        '''
        Overloaded right multiplication(*) operator.
        '''
        return self*other

    def __idiv__(self,other):
        '''
        Overloaded self-division(/=) operator.
        '''
        return self.__imul__(1.0/other)

    def __div__(self,other):
        '''
        Overloaded left division(/) operator.
        '''
        return self*(1.0/other)

    def __eq__(self,other):
        '''
        Overloaded equivalent(==) operator.
        '''
        raise NotImplementedError("%s (==) error: not implemented."%self.__class__.__name__)

    def __ne__(self,other):
        '''
        Overloaded not-equivalent(!=) operator.
        '''
        return not self==other
