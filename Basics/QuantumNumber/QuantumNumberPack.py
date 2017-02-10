'''
Quantum number pack, including:
1) classes: SpinQN, FermiQN
2) functions: SpinQNs, FermiQNs
'''

import numpy as np
from QuantumNumber import *
from itertools import combinations
from collections import Counter

__all__=['SpinQN','SpinQNs','FermiQN','FermiQNs']

class SpinQN(QuantumNumber):
    '''
    The quantum number for a state with spin Sz.
    Attributes:
        names: ('Sz',)
            The names of the quantum number.
        periods: (None,)
            The periods of the quantum number.
    '''
    names=('Sz',)
    periods=(None,)

def SpinQNs(S):
    '''
    The quantum number collection for spin S.
    Parameters:
        S: integer / half integer
            The value of the spin.
    Returns: QuantumNumbers
        The corresponding quantum number collection.
    '''
    result=[]
    for sz in reversed(np.linspace(-S,S,int(2*S+1),endpoint=True)):
        result.append((SpinQN(sz),1))
    return QuantumNumbers(result)

class FermiQN(QuantumNumber):
    '''
    The quantum number for a state with fermion number N and spin Sz.
    Attributes:
        names: ('N','Sz')
            The names of the quantum number.
        periods: (None,None)
            The periods of the quantum number.
    '''
    names=('N','Sz')
    periods=(None,None)

def FermiQNs(S):
    '''
    The quantum number collection for fermions with spin S.
    Parameters:
        S: integer / half integer
            The value of the fermion's spin.
    Returns: QuantumNumbers
        The corresponding quantum number collection.
    '''
    result=[]
    temp=SpinQNs(S).keys()
    for n in xrange(len(temp)+1):
        if n==0:
            result.append(FermiQN((0,0.0)))
        else:
            fn=QuantumNumber([('N',n,'U1')])
            for ss in combinations(temp,n):
                result.append(fn.direct_sum(sum(ss)))
    return QuantumNumbers(sorted(Counter(result).items()))
