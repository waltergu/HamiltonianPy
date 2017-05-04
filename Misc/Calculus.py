'''
========
Calculus
========

Calculus related functions, including
    * functions: derivatives
'''

__all__=['derivatives']

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
