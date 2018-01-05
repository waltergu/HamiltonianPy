'''
========
Calculus
========

Calculus related functions, including
    * functions: derivatives, bisect, fstable
'''

__all__=['derivatives','bisect','fstable']

import numpy as np
import numpy.linalg as nl
import scipy.interpolate as ip
import warnings

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

def fstable(fx,x0,args=(),step=10**-4,tol=10**-6,rate=0.9,maxiter=50):
    '''
    Find the stable point of a function.

    Parameters
    ----------
    fx : callable
        The function whose stable point is to be searched.
    x0 : 1d ndarray like
        The initial guess of the stable point.
    args : tuple, optional
        The extra parameters passed to `fx`.
    step : float, optional
        The initial step.
    tol : float, optional
        The tolerance of the solution.
    rate : float, optional
        The learning rate of the steps.
    maxiter : int, optional
        The maximum number of iterations.

    Returns
    -------
    xnew : 1d ndarray
        The solution of the stable point.
    f : float
        The function value at the stable point.
    err : float
        The err of the stable point.
    niter : int
        The number of iterations.
    '''
    def quadratic(fx,x0,args,steps):
        N=len(x0)
        xs=[x0]
        es=np.eye(N)
        for i in xrange(N):
            xs.append(x0+steps[i]*es[i])
            xs.append(x0-steps[i]*es[i])
            for j in xrange(i): xs.append(x0+steps[i]*es[i]+steps[j]*es[j])
        a=np.zeros(((N+1)*(N+2)/2,(N+1)*(N+2)/2))
        b=np.zeros((N+1)*(N+2)/2)
        for n,x in enumerate(xs):
            count=0
            for i,xi in enumerate(x):
                a[n,1+i]=xi
                a[n,N+1+count:N+1+count+i+1]=xi*x[:i+1]
                count+=i+1
            a[n,0]=1
            b[n]=fx(x,*args)
        coeff=nl.solve(a,b)
        fx0=coeff[0]
        fx1=coeff[1:N+1]
        fx2=np.zeros((N,N))
        count=0
        for i in xrange(N):
            for j in xrange(i+1):
                if j==i:
                    fx2[i,j]=coeff[N+1+count]
                else:
                    fx2[i,j]=coeff[N+1+count]/2
                    fx2[j,i]=fx2[i,j]
                count+=1
        return fx2,fx1,fx0,b[0]
    xold,err,niter,steps=np.asarray(x0),1.0,0,np.array([step]*len(x0))
    while err>tol and niter<maxiter:
        fx2,fx1,fx0,f=quadratic(fx,xold,args,steps)
        xnew=nl.solve(fx2,-fx1/2.0)
        diff=xnew-xold
        err,steps=nl.norm(diff),rate*diff
        xold=xnew
        niter+=1
    if err>tol: warnings.warn('fstable warning: not converged after %s iterations with current err being %s.'%(niter,err))
    return xnew,f,err,niter
