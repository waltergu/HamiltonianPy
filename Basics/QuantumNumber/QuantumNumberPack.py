'''
Quantum number pack, including:
1) classes: SQN, PQN, SPQN, Z2QN, Z2PQN
2) functions: SQNS, PQNS, SPQNS, Z2QNS, Z2PQNS
'''

import numpy as np
import itertools as it
from QuantumNumber import *

__all__=['SQN','SQNS','PQN','PQNS','SPQN','SPQNS','Z2QN','Z2QNS','Z2PQN','Z2PQNS']

class SQN(QuantumNumber):
    '''
    The quantum number for a spin state with the z component Sz.
    Attributes:
        names: ('Sz',)
            The names of the quantum number.
        periods: (None,)
            The periods of the quantum number.
    '''
    names=('Sz',)
    periods=(None,)

def SQNS(S):
    '''
    The collection of quantum numbers for a single spin S.
    Parameters:
        S: integer / half integer
            The value of the spin.
    Returns: QuantumNumbers
        The corresponding collection of quantum numbers.
    '''
    return QuantumNumbers('C',([SQN(sz) for sz in np.arange(-S,S+1)],range(int(2*S)+2)),protocal=QuantumNumbers.INDPTR)

class PQN(QuantumNumber):
    '''
    The quantum number for a spinless particle state with the particle number N.
    Attributes:
        names: ('N')
            The names of the quantum number.
        periods: (None)
            The periods of the quantum number.
    '''
    names=('N',)
    periods=(None,)

def PQNS(N):
    '''
    The collection of quantum numbers for a single site occupied with maximum N spinless particle.
    Parameters:
        N: integer
            The maximum number of the particle number.
    Returns: QuantumNumbers
        The corresponding collection of quantum numbers.
    '''
    return QuantumNumbers('C',([PQN(n) for n in xrange(N+1)],range(N+2)),protocal=QuantumNumbers.INDPTR)

class SPQN(QuantumNumber):
    '''
    The quantum number for a spinful particle state with particle number N and spin-z-component Sz.
    Attributes:
        names: ('N','Sz')
            The names of the quantum number.
        periods: (None,None)
            The periods of the quantum number.
    '''
    names=('N','Sz')
    periods=(None,None)

def SPQNS(S):
    '''
    The collection of quantum numbers for a single site occupied with maximum one spin-S particle.
    Parameters:
        S: integer / half integer
            The value of the particle's spin.
    Returns: QuantumNumbers
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
    Attributes:
        names: ('Z2',)
            The names of the quantum number.
        periods: (2,)
            The periods of the quantum number.
    '''
    names=('Z2',)
    periods=(2,)

def Z2QNS():
    '''
    The collection of Z2 quantum numbers.
    Returns: QuantumNumbers
        As above.
    '''
    return QuantumNumbers('C',([Z2QN(0.0),Z2QN(1.0)],range(3)),protocal=QuantumNumbers.INDPTR)

class Z2PQN(QuantumNumber):
    '''
    The quantum number for a spinful particle state with a Z2 spin parity.
    Attributes:
        names: ('N','Z2')
            The names of the quantum number.
        periods: (None,2)
            The periods of the quantum number.
    '''
    names=('N','Z2')
    periods=(None,2)

def Z2PQNS():
    '''
    The collection of quantum numbers for a single site occupied with a spin Z2 particle.
    Returns: QuantumNumbers
        As above.
    '''
    return QuantumNumbers('C',([Z2PQN((0.0,0.0)),Z2PQN((1.0,1.0)),Z2PQN((2.0,0.0))],[1,2,1]),protocal=QuantumNumbers.COUNTS)
