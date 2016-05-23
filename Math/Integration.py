'''
Integration.
'''

__all__=['knots_and_weights','integration']

from numpy import *
from numpy.polynomial.legendre import leggauss

def knots_and_weights(a,b,deg,method='legendre'):
    '''
    This function returns the nodes and weights used for the Guass quadrature.
    Parameters:
        a,b: float
            The lower and upper limit of the sample interval.
        deg: integer
            The number of the sample points and weights.
        method: string,optional
            The type of the polynomials.
    Returns:
        knots: 1D ndarray
            The knots.
        weights: 1D ndarray
            The weights. 
    '''
    if method=='legendre':
        knots,weights=leggauss(deg)
        knots=(b-a)/2*knots+(a+b)/2
        weights=(b-a)/2*weights
    return knots,weights

def integration(func,a,b,args=(),deg=64,method='legendre'):
    '''
    This function calculates the integration of a given function at a given interval using the Guass quadrature.
    Parameters:
        func: function
            The function to be integrated.
        a,b: float
            The lower and upper limit of the integration.
        args: tuple
            The arguments of the function func.
        deg: integer, optional
            The number of sample points and weights used in the Guass quadrature.
        method: string, optional
            The type of the polynomials used in the Guass quadrature.
    Returns:
        result: same type with the returns of func
            The calculated integral.
    '''
    knots,weights=knots_and_weights(a,b,deg,method)
    result=0
    for knot,weight in zip(knots,weights):
        result+=func(knot,*args)*weight
    return result
