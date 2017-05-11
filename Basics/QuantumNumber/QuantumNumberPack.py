'''
----------------------------
Common classes and functions
----------------------------

Quantum number pack, including:
    * classes: SQN, PQN, SPQN, Z2QN
    * functions: SQNS, PQNS, SzPQNS, SPQNS, Z2QNS
'''

import numpy as np
import itertools as it
from QuantumNumber import *

__all__=['SQN','SQNS','PQN','PQNS','SPQN','SzPQNS','SPQNS','Z2QN','Z2QNS']

class SQN(QuantumNumber):
    '''
    The quantum number for a spin state with the z component Sz.

    Attributes
    ----------
    names : ('Sz',)
        The names of the quantum number.
    periods : (None,)
        The periods of the quantum number.
    '''
    names=('Sz',)
    periods=(None,)

def SQNS(S):
    '''
    The collection of quantum numbers for a single spin S.

    Parameters
    ----------
    S : integer / half integer
        The value of the spin.

    Returns
    -------
    QuantumNumbers
        The corresponding collection of quantum numbers.
    '''
    return QuantumNumbers('C',([SQN(sz) for sz in np.arange(-S,S+1)],range(int(2*S)+2)),protocal=QuantumNumbers.INDPTR)

class PQN(QuantumNumber):
    '''
    The quantum number for a single particle state with the particle number N.

    Attributes
    ----------
    names : ('N',)
        The names of the quantum number.
    periods : (None,)
        The periods of the quantum number.
    '''
    names=('N',)
    periods=(None,)

def PQNS(N):
    '''
    The collection of quantum numbers for a single site occupied with maximum N identical particles.

    Parameters
    ----------
    N: integer
        The maximum number of the particle number.

    Returns
    -------
    QuantumNumbers
        The corresponding collection of quantum numbers.
    '''
    return QuantumNumbers('C',([PQN(n) for n in xrange(N+1)],range(N+2)),protocal=QuantumNumbers.INDPTR)

class SPQN(QuantumNumber):
    '''
    The quantum number for a single particle state with particle number N and spin-z-component Sz.

    Attributes
    ----------
    names : ('N','Sz')
        The names of the quantum number.
    periods : (None,None)
        The periods of the quantum number.
    '''
    names=('N','Sz')
    periods=(None,None)

def SzPQNS(Sz):
    '''
    The collection of quantum numbers for a spin-z-component Sz single particle state occupied with zero or one particle.

    Parameters
    ----------
    Sz : integer / half integer
        The value of the particle's spin.

    Returns
    -------
    QuantumNumbers
        The corresponding collection of quantum numbers.
    '''
    return QuantumNumbers('C',([SPQN((0.0,0.0)),SPQN((1.0,Sz))],[1,1]))

def SPQNS(S):
    '''
    The collection of quantum numbers for a spin-S site occupied with any available number of particles.

    Parameters
    ----------
    S : integer / half integer
        The value of the particle's spin.

    Returns
    -------
    QuantumNumbers
        The corresponding collection of quantum numbers.
    '''
    qns,spins=[SPQN((0,0.0))],[SQN(sz) for sz in np.arange(-S,S+1)]
    for n in xrange(1,len(spins)+1):
        pn=PQN(n)
        for ss in it.combinations(spins,n):
            qns.append(SPQN.directsum(pn,sum(ss)))
    return QuantumNumbers('G',(qns,range(len(qns)+1)),protocal=QuantumNumbers.INDPTR).sort()

class Z2QN(QuantumNumber):
    '''
    The Z2-typed quantum number.

    Attributes
    ----------
    names : ('Z2',)
        The names of the quantum number.
    periods : (2,)
        The periods of the quantum number.
    '''
    names=('Z2',)
    periods=(2,)

def Z2QNS():
    '''
    The collection of Z2 quantum numbers for a single site.

    Returns
    -------
    QuantumNumbers
        As above.
    '''
    return QuantumNumbers('C',([Z2QN(0.0),Z2QN(1.0)],range(3)),protocal=QuantumNumbers.INDPTR)
