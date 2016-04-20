'''
Fermionic Operator, including:
1) classes: OperatorF
2) functions: F_Linear, F_Quadratic, F_Hubbard
'''

__all__=['OperatorF','F_Linear','F_Quadratic','F_Hubbard']

from numpy import *
from DegreeOfFreedomPy import ANNIHILATION,CREATION
from ..OperatorPy import *
from copy import deepcopy

class OperatorF(Operator):
    '''
    This class gives a unified description of fermionic operators with different ranks.
    Attributes:
    mode: string
        The tag used to distinguish operators with different types or ranks.
    indices: list of Index
        The associated indices of the operator, whose length should be equal to the operator's rank;
    rcoords: list of 1D ndarray
        The associated real coordinates of the operator.
    icoords: list of 1D ndarray
        The associated lattice coordinates of the operator.
    seqs: tuple of integer
        The associated sequences of the operator, whose length should be equal to the operator's rank.
    Note:
    1) The lengths of rcoords and icoords are not forced to be equal to the operator's rank because:
        (1) some of its rank-1 terms may share the same rcoord or icoord, and 
        (2) the rcoords and icoords is the whole operator's property instead of each of its rank-1 component.
       However, for a set of operators with the same attribute mode, the lengths of their rcoords and icoords should be fixed and equal to each other respectively.
    2) Current supported modes include:
        (1) 'f_linear':
            rank==1 fermionic operators.
        (2) 'f_quadratic':
            rank==2 fermionic operators.
            For this mode, only one rcoord and one icoord are needed which are identical to the bond's rcoord and icoord where the quadratic operator is defined.
        (3) 'f_hubbard':
            rank==4 fermionic operators.
            For this mode, only one rcoord and icoord is needed because Hubbard operators are always on-site ones.
    '''
    
    def __init__(self,mode,value,indices,rcoords,icoords,seqs):
        self.mode=mode
        self.value=value
        self.indices=indices
        self.rcoords=[]
        for obj in rcoords:
            self.rcoords.append(array(obj))
        self.icoords=[]
        for obj in icoords:
            self.icoords.append(array(obj))
        self.seqs=tuple(seqs)
        self.set_id()

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        result=[]
        result.append('mode=%s'%self.mode)
        result.append('value=%s'%self.value)
        result.append('indices=%s'%self.indices)
        result.append('rcoords=%s'%self.rcoords)
        result.append('icoords=%s'%self.icoords)
        result.append('seqs='+str(self.seqs))
        return 'Operator('+', '.join(result)+')'

    def set_id(self):
        '''
        Set the unique id of this operator.
        '''
        result=[]
        result.append(self.mode)
        result.append(tuple(self.indices))
        result.append(str(self.rcoords))
        result.append(str(self.icoords))
        result.append(self.seqs)
        self.id=tuple(result)

    @property
    def dagger(self):
        '''
        The dagger, i.e. the Hermitian conjugate of an operator.
        '''
        indices=[]
        for obj in self.indices:
            indices.append(obj.replace(nambu=1-obj.nambu))
        indices.reverse()
        return OperatorF(mode=self.mode,value=conjugate(self.value),indices=indices,rcoords=list(reversed(self.rcoords)),icoords=list(reversed(self.icoords)),seqs=reversed(list(self.seqs)))

    @property
    def rank(self):
        '''
        Return the rank of the operator.
        '''
        return len(self.seqs)

    def is_normal_ordered(self):
        '''
        Judge whether an operator is normal ordered.
        '''
        buff=True
        for index in self.indices:
            if index.nambu==ANNIHILATION: buff=False
            if not buff and index.nambu==CREATION: return False
        return True

    def is_Hermitian(self):
        '''
        Judge whether an operator is Hermitian.
        '''
        return self==self.dagger

def F_Linear(value,indices,rcoords,icoords,seqs):
    '''
    A specialized constructor to create an Operator instance with mode='f_linear'.
    '''
    return Operator('f_linear',value,indices,rcoords,icoords,seqs)

def F_Quadratic(value,indices,rcoords,icoords,seqs):
    '''
    A specialized constructor to create an Operator instance with mode='f_quadratic'.
    '''
    return Operator('f_quadratic',value,indices,rcoords,icoords,seqs)

def F_Hubbard(value,indices,rcoords,icoords,seqs):
    '''
    A specialized constructor to create an Operator instance with mode='f_hubbard'.
    '''
    return Operator('f_hubbard',value,indices,rcoords,icoords,seqs)
