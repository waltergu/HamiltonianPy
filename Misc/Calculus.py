'''
========
Calculus
========

Calculus related functions, including
    * functions: derivatives, bisect, fstable, newton
'''

__all__=['derivatives','bisect','fstable','newton']

import numpy as np
import numpy.linalg as nl
import scipy.interpolate as ip
import scipy.optimize as op
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

def fstable(fun,x0,args=(),method='Newton',tol=10**-6,callback=None,options=None):
    '''
    Find the stable point of a function.

    Parameters
    ----------
    fun : callable
        The function whose stable point is to be searched.
    x0 : 1d ndarray
        The initial guess of the stable point.
    args : tuple, optional
        The extra parameters passed to `fun`.
    method : 'Newton','BFGS','CG', optional
        The method used to find the stable point.
    tol : float, optional
        The tolorence of the result.
    callback : callable, optional
        The callback function after each iteration.
    options : dict, optional
        The extra options of each specific method.

    Returns
    -------
    OptimizeResult
        The result.

    Notes
    -----
    Method 'Newton' searches for maxima, minima or saddle points while method 'BFGS' and 'CG' noly searches for minima.
    '''
    assert method in ('Newton','BFGS','CG')
    if method=='Newton':
        return newton(fun,x0,args,tol=tol,callback=callback,**(options or {}))
    else:
        return op.minimize(fun,x0,args,method,tol=tol,callback=callback,options=options)

def newton(fun,x0,args=(),tol=10**-6,callback=None,disp=False,eps=10**-4,rate=0.9,return_all=False,maxiter=50):
    '''
    Find the stable point of a function.

    Parameters
    ----------
    fun : callable
        The function whose stable point is to be searched.
    x0 : 1d ndarray like
        The initial guess of the stable point.
    args : tuple, optional
        The extra parameters passed to `fun`.
    tol : float, optional
        The tolerance of the solution.
    callback : callable, optional
        The callback function after each iteration.
    disp : logical, optional
        True for displaying the convergence messages.
    eps : float, optional
        The initial step size.
    rate : float, optional
        The learning rate of the steps.
    return_all : logical, optional
        True for returning all the convergence information.
    maxiter : int, optional
        The maximum number of iterations.

    Returns
    -------
    OptimizeResult
        The result.

    xnew : 1d ndarray
        The solution of the stable point.
    f : float
        The function value at the stable point.
    err : float
        The err of the stable point.
    niter : int
        The number of iterations.
    '''
    def quadratic(fun,x0,args,steps):
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
            b[n]=fun(x,*args)
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
        return fx2,fx1,fx0,b[0],len(xs)
    result=op.OptimizeResult()
    if return_all: result.allvecs=[np.asarray(x0)]
    xold,err,count,niter,steps=np.asarray(x0),10*np.abs(tol),0,0,np.array([eps]*len(x0))
    while err>tol and niter<maxiter:
        fx2,fx1,fx0,f,nfev=quadratic(fun,xold,args,steps)
        count+=nfev
        xnew=nl.solve(fx2,-fx1/2.0)
        diff=xnew-xold
        err,steps=nl.norm(diff),rate*diff
        xold=xnew
        niter+=1
        if return_all: result.allvecs.append(xold)
        if callback is not None: callback(xold)
    result.x,result.fun,result.nfev,result.nit=xold,f,count,niter
    if err>tol:
        message='newton warning: not converged after %s iterations with current err being %s.'%(niter,err)
        warnings.warn(message)
        result.success,result.status,result.message=False,1,message
    else:
        result.success,result.status,result.message=True,0,'Optimization terminated successfully.'
    if disp:
        print result.message
        print 'Current function value: %s'%result.fun
        print 'Iterations: %s'%result.nit
        print 'Function evaluations: %s'%result.nfev
    return result
