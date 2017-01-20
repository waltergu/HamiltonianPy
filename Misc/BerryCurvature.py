'''
Berry curvature.
'''

__all__=['berry_curvature']

from numpy import *
from scipy.linalg import eigh

def berry_curvature(H,kx,ky,mu,d=10**-6):
    '''
    Calculate the Berry curvature of the occupied bands for a Hamiltonian with the given chemical potential using the Kubo formula.
    Parameters:
        H: function
            Input function which returns the Hamiltonian as a 2D array.
        kx,ky: float
            The two parameters which specify the 2D point at which the Berry curvature is to be calculated.
            They are also the input parameters to be conveyed to the function H.
        mu: float
            The chemical potential.
        d: float,optional
            The spacing to be used to calculate the derivates.
    Returns:
        result: float
            The calculated Berry curvature for function H at point kx,ky with chemical potential mu.
    '''
    result=0
    Vx=(H(kx+d,ky)-H(kx-d,ky))/(2*d)
    Vy=(H(kx,ky+d)-H(kx,ky-d))/(2*d)
    Es,Evs=eigh(H(kx,ky))
    for n in xrange(Es.shape[0]):
        for m in xrange(Es.shape[0]):
            if Es[n]<=mu and Es[m]>mu:
                result-=2*(vdot(dot(Vx,Evs[:,n]),Evs[:,m])*vdot(Evs[:,m],dot(Vy,Evs[:,n]))/(Es[n]-Es[m])**2).imag
    return result
