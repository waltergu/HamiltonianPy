'''
========
Calculus
========

Calculus related functions, including
    * functions: derivatives, bisect
'''

__all__=['derivatives','bisect']

import numpy as np
import scipy.interpolate as ip

def derivatives(xs,ys,ders=(1,)):
    '''
    Calculate the numerical derivatives of `ys` at points `xs` by use of the third-order spline interpolation.

    Parameters
    ----------
    xs : 1d ndarray
        The sample points of the argument.
    ys: 1d ndarray
        The corresponding sample points of the function.
    ders : tuple of integer
        The derivatives to calculate.

    Returns
    -------
    2d ndarray
        The derivatives at the sample points of the argument.
    '''
    assert len(xs)==len(ys)
    xs,ys=np.asarray(xs),np.asarray(ys)
    result=np.zeros((len(ders),len(xs)),dtype=ys.dtype)
    tck=ip.splrep(xs,ys,s=0)
    for i,der in enumerate(ders):
        result[i]=ip.splev(xs,tck,der=der)
    return result

def bisect(f,xs,args=()):
    '''
    Find the minimum interval that contains the root of a function using the bisection method.

    Parameters
    ----------
    f : callable
        The function whose root is to be searched.
    xs : 1d ndarray
        The discretized interval within which to search the root of the input function.
    args : tuple
        The extra parameters passed to the input function.

    Returns
    -------
    tuple
        The minimum interval.
    '''
    assert len(xs)>1
    xs,xdw,xup=sorted(xs),0,len(xs)-1
    fdw,fup=f(xs[xdw],*args),f(xs[xup],*args)
    if fdw==0:
        return xs[xdw],xs[xdw]
    elif fup==0:
        return xs[xup],xs[xup]
    else:
        assert fdw*fup<0
    while True:
        xnew=(xup+xdw)/2
        if xnew==xdw or xnew==xup: return xs[xdw],xs[xup]
        fnew=f(xs[xnew],*args)
        if fdw*fnew<0:
            fup=fnew
            xup=xnew
        elif fnew*fup<0:
            fdw=fnew
            xdw=xnew
        else:
            return xs[xnew],xs[xnew]
