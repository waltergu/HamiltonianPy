'''
----------------------------
Common classes and functions
----------------------------

Quantum number pack, including:
    * classes: SQN, PQN, SPQN, Z2QN
    * functions: NewQuantumNumber, SQNS, PQNS, SzPQNS, SPQNS, Z2QNS, TQNS
'''

import numpy as np
import itertools as it
from QuantumNumber import *

__all__=['NewQuantumNumber','SQN','SQNS','PQN','PQNS','SPQN','SzPQNS','SPQNS','Z2QN','Z2QNS','TQNS']

template="""\
class {typename}(QuantumNumber):
    '''
    {doc}

    Attributes
    ----------
    names : {names}
        The names of the quantum number.
    periods : {periods}
        The periods of the quantum number.
    '''
    names={names}
    periods={periods}
"""

def NewQuantumNumber(typename,names,periods,doc=None):
    '''
     A factory method that defines new subclass of QuantumNumber.

    Parameters
    ----------
    typename : str
        The type name of the subclass.
    names : tuple of str
        The names of the quantum number of the subclass.
    periods : tuple of None/int
        The periods of the quantum number of the subclass.
    doc : str, optional
        The docstring of the subclass.

    Returns
    -------
    subclass of QuantumNumber
        The new subclass of QuantumNumber.
    '''
    assert len(names)==len(periods)
    for name,period in zip(names,periods):
        assert isinstance(name,str)
        assert (period is None) or (type(period) in (long,int) and period>0)
    namespace={'QuantumNumber':QuantumNumber}
    definition=template.format(typename=typename,names=names,periods=periods,doc=doc or typename)
    try:
        exec definition in namespace
    except SyntaxError as err:
        raise SyntaxError(err.message+':\n'+definition)
    return namespace[str(typename)]

class SQN(NewQuantumNumber('SQN',('Sz',),(None,),doc='The quantum number for a spin state with the z component Sz.')): pass
class PQN(NewQuantumNumber('PQN',('N',),(None,),doc='The quantum number for a single particle state with the particle number N.')): pass
class SPQN(NewQuantumNumber('SPQN',('N','Sz'),(None,None),doc='The quantum number for a single particle state with particle number N and spin-z-component Sz.')): pass
class Z2QN(NewQuantumNumber('Z2QN',('Z2',),(None,),doc='The Z2-typed quantum number.')): pass

def SQNS(S):
    '''
    The collection of quantum numbers for a single spin S.

    Parameters
    ----------
    S : int / half int
        The value of the spin.

    Returns
    -------
    QuantumNumbers
        The corresponding collection of quantum numbers.
    '''
    return QuantumNumbers('C',([SQN(sz) for sz in np.arange(-S,S+1)],range(int(2*S)+2)),protocol=QuantumNumbers.INDPTR)

def PQNS(N):
    '''
    The collection of quantum numbers for a single site occupied with maximum N identical particles.

    Parameters
    ----------
    N: int
        The maximum number of the particle number.

    Returns
    -------
    QuantumNumbers
        The corresponding collection of quantum numbers.
    '''
    return QuantumNumbers('C',([PQN(n) for n in xrange(N+1)],range(N+2)),protocol=QuantumNumbers.INDPTR)

def SzPQNS(Sz):
    '''
    The collection of quantum numbers for a spin-z-component Sz single particle state occupied with zero or one particle.

    Parameters
    ----------
    Sz : int / half int
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
    S : int / half int
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
    return QuantumNumbers('G',(qns,range(len(qns)+1)),protocol=QuantumNumbers.INDPTR).sorted()

def Z2QNS():
    '''
    The collection of Z2 quantum numbers for a single site.

    Returns
    -------
    QuantumNumbers
        As above.
    '''
    return QuantumNumbers('C',([Z2QN(0.0),Z2QN(1.0)],range(3)),protocol=QuantumNumbers.INDPTR)

def TQNS(n):
    '''
    Trivial quantum numbers.

    Parameters
    ----------
    n : int
        The dimension of the trivial quantum numbers.

    Returns
    -------
    QuantumNumbers
        As above.
    '''
    return QuantumNumbers.mono(PQN(0),n)
