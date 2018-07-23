'''
========
Optimize
========

Optimization related functions, including
    * functions: bisect, derivatives, fpapprox, quadapprox, searchstable, newton, fstable
'''

__all__=['bisect','derivatives','fpapprox','quadapprox','linesearchstable','newton','fstable']

import numpy as np
import numpy.linalg as nl
import scipy.interpolate as ip
import scipy.optimize as op
import warnings

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

def derivatives(xs,ys,ders=(1,)):
    '''
    Calculate the numerical derivatives of `ys` at points `xs` by use of the third-order spline interpolation.

    Parameters
    ----------
    xs : 1d ndarray
        The sample points of the argument.
    ys: 1d ndarray
        The corresponding sample points of the function.
    ders : tuple of int
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

def fpapprox(fun,x0,args=(),eps=1.49e-06,fpmode=0):
    '''
    Finite-difference approximation of the gradient of a scalar function at a given point.

    Parameters
    ----------
    fun : callable
        The function whose gradient is to be approximated.
    x0 : 1d ndarray
        The given point.
    args : tuple, optional
        The extra arguments of the function ``fun``.
    eps : float, optional
        The step size.
    fpmode : 0 or 1, optional
        * 0 use ``(f(x+eps)-f(x))/eps`` to approximate fp;
        * 1 use ``(f(x+eps)-f(x-eps))/2/eps`` to approximate fp.

    Returns
    -------
    1d ndarray
        The approximated gradient.
    '''
    if fpmode==0:
        return op.approx_fprime(x0,fun,eps,*args)
    else:
        result,dx=np.zeros(len(x0)),np.eye(len(x0))
        for i in xrange(len(x0)):
            result[i]=(fun(x0+eps*dx[i],*args)-fun(x0-eps*dx[i],*args))/2/eps
        return result

def quadapprox(fun,x0,args=(),eps=1.49e-06):
    '''
    Quadratic approximation of a function at a given point.

    Parameters
    ----------
    fun : callable
        The function to be approximated.
    x0 : 1d ndarray
        The given point.
    args : tuple, optional
        The extra arguments of the function ``fun``.
    eps : float, optional
        The step size.

    Returns
    -------
    f0 : float
        The function value at the given point.
    fp1 : 1d ndarray
        The approximated first order derivatives at the given point.
    fp2 : 2d ndarray
        The approximated second order derivatives at the given point.
    '''
    N,xs,diffs=len(x0),[],[]
    es,eps=np.eye(N),[eps]*N if isinstance(eps,float) else eps
    for i in xrange(N):
        xs.append(x0+eps[i]*es[i])
        diffs.append(eps[i]*es[i])
        xs.append(x0-eps[i]*es[i])
        diffs.append(-eps[i]*es[i])
        for j in xrange(i):
            xs.append(x0+eps[i]*es[i]+eps[j]*es[j])
            diffs.append(eps[i]*es[i]+eps[j]*es[j])
    f0=fun(x0,*args)
    a,b=np.zeros((N*(N+3)/2,N*(N+3)/2)),np.zeros(N*(N+3)/2)
    for n,(x,diff) in enumerate(zip(xs,diffs)):
        count=0
        for i,di in enumerate(diff):
            a[n,i]=di
            a[n,N+count:N+count+i+1]=di*diff[:i+1]
            count+=i+1
        b[n]=fun(x,*args)
    coeff,count=nl.solve(a,b-f0),0
    fp1,fp2=coeff[:N],np.zeros((N,N))
    for i in xrange(N):
        for j in xrange(i+1):
            if j==i:
                fp2[i,j]=coeff[N+count]*2
            else:
                fp2[i,j]=coeff[N+count]
                fp2[j,i]=fp2[i,j]
            count+=1
    return f0,fp1,fp2

def linesearchstable(fun,x0,dx,fp1,fp2,args=(),eps=1.49e-06,fpmode=0,c1=1e-4):
    '''
    Use the Armijo condition to search the stationary point of a function along a direction.

    Parameters
    ----------
    fun : callable
        Objective function.
    x0 : 1d ndarray
        The starting point.
    dx : 1d ndarray
        The search direction.
    fp1 : 1d ndarray
        The first derivative of the objective function at the starting point.
    fp2 : 1d ndarray
        The second derivative of the objective function at the starting point.
    args : tuple, optional
        The extra arguments of the ojective function.
    eps : float, optioinal
        The step size to approximate the derivatives of the objective function.
    fpmode : 0 or 1, optional
        * 0 use ``(f(x+eps)-f(x))/eps`` to approximate fp;
        * 1 use ``(f(x+eps)-f(x-eps))/2/eps`` to approximate fp.
    c1 : float, optional
        Parameter for Armijo condition rule.

    Returns
    -------
    alpha : float
        The best alpha.
    fop : float
        The function value at the optimal point.
    fp1op : float
        The directional derivative of the object function at the optimal point.
    '''
    eps=eps/nl.norm(dx)
    record=[]
    fa=lambda alpha: fun(x0+alpha*dx,*args)
    fap=(lambda alpha:(fa(alpha+eps)-fa(alpha))/eps) if fpmode==0 else (lambda alpha:(fa(alpha+eps)-fa(alpha-eps))/2/eps)
    def phi(alpha):
        record.append(alpha)
        return fap(alpha)**2
    alpha=op.linesearch.scalar_search_armijo(phi,(fp1.dot(dx))**2,fp1.dot(fp2.dot(fp1)),c1=c1,amin=10*eps)[0]
    if alpha is None:
        warnings.warn('linesearchstable warning: not converged, last value used as the final alpha.')
        alpha=record[-1]
    fop,fp1op=fa(alpha),fap(alpha)
    return alpha,fop,fp1op

def newton(fun,x0,args=(),tol=10**-4,callback=None,disp=False,eps=1.49e-06,fpmode=0,hesmode='quadapprox',maxiter=50):
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
    fpmode : 0 or 1, optional
        * 0 use ``(f(x+eps)-f(x))/eps`` to approximate fp;
        * 1 use ``(f(x+eps)-f(x-eps))/2/eps`` to approximate fp.
    hesmode : 'quadapprox' or 'BFGS', optional
        * 'quadapprox' use ``quadapprox`` to approximate the Hessian matrix at each iteration;
        * 'BFGS' use ``quadapprox`` to approximate the Hessian matrix at the beginning and update it by the BFGS formula.
    maxiter : int, optional
        The maximum number of iterations.

    Returns
    -------
    OptimizeResult
        The result.
    '''
    record={}
    def fx(x):
        xt=tuple(x)
        if xt not in record: record[xt]=fun(x,*args)
        return record[xt]
    def fpquadapprox(f,x):
        f,fp1,fp2=quadapprox(f,x,eps=eps)
        return fp1,fp2
    def fpbfgs(f,x,dx,fp1,fp2):
        nfp=fpapprox(f,x,eps=eps,fpmode=fpmode)
        dfp=nfp-fp1
        fp1=nfp
        fp2+=np.einsum('i,j->ij',dfp,dfp)/dfp.dot(dx)-np.einsum('i,j->ij',fp2.dot(dx),fp2.T.dot(dx))/dx.dot(fp2.dot(dx))
        return fp1,fp2
    result=op.OptimizeResult()
    x=np.asarray(x0)
    fp1,fp2=fpquadapprox(fx,x)
    for niter in xrange(maxiter):
        diff=nl.solve(fp2,-fp1)
        print '\ndiff: %s\n'%diff
        err=nl.norm(diff)
        if err<=tol:
            x+=diff
            if callable(callback): callback(x)
            break
        alpha=linesearchstable(fx,x,diff,fp1,fp2,eps=eps,fpmode=fpmode)[0]
        print '\nalpha: %s\n'%alpha
        err*=np.abs(alpha)
        x+=alpha*diff
        if callable(callback): callback(x)
        if err<=tol: break
        fp1,fp2=fpquadapprox(fx,x) if hesmode=='quadapprox' else fpbfgs(fx,x,alpha*diff,fp1,fp2)
    result.x,result.fun,result.nfev,result.nit=x,fx(x),len(record),niter+1
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

def fstable(fun,x0,args=(),method='Newton',tol=10**-4,callback=None,options=None):
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
