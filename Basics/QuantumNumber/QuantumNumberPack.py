'''
Quantum number pack, including:
1) functions: SpinQN, SpinQNC, FermiQN, FermiQNC
'''

import numpy as np
from QuantumNumber import *
from itertools import combinations
from collections import Counter

__all__=['SpinQN','SpinQNC','FermiQN','FermiQNC']

def SpinQN(Sz):
    '''
    The quantum number for a state with spin Sz.
    Parameters:
        Sz: integer / half integer
            The value of the spin Sz.
    Returns: QuantumNumber
        The corresponding quantum number.
    '''
    return QuantumNumber([('Sz',Sz,'U1')])

def SpinQNC(S):
    '''
    The quantum number collection for spin S.
    Parameters:
        S: integer / half integer
            The value of the spin.
    Returns: QuantumNumberCollection
        The corresponding quantum number collection.
    '''
    result=[]
    for sz in reversed(np.linspace(-S,S,int(2*S+1),endpoint=True)):
        result.append((QuantumNumber([('Sz',sz,'U1')]),1))
    return QuantumNumberCollection(result)

def FermiQN(N,Sz):
    '''
    The quantum number for a state with fermion number N and spin Sz.
    Parameters:
        N: integer
            The value of the fermion number.
        Sz: integer / half integer
            The value of the spin Sz.
    Returns: QuantumNumber
        The corresponding quantum number.
    '''
    return QuantumNumber([('N',N,'U1'),('Sz',Sz,'U1')])

def FermiQNC(S):
    '''
    The quantum number collection for fermions with spin S.
    Parameters:
        S: integer / half integer
            The value of the fermion's spin.
    Returns: QuantumNumberCollection
        The corresponding quantum number collection.
    '''
    result=[]
    temp=SpinQNC(S).keys()
    for n in xrange(len(temp)+1):
        if n==0:
            result.append(QuantumNumber([('N',0,'U1'),('Sz',0,'U1')]))
        else:
            fn=QuantumNumber([('N',n,'U1')])
            for ss in combinations(temp,n):
                result.append(fn.direct_sum(sum(ss)))
    return QuantumNumberCollection(sorted(Counter(result).items()))
