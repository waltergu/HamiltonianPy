'''
Spin Operator, including:
1) classes: OperatorS
'''

__all__=['OperatorS']

from ..Operator import *

class OperatorS(Operator):
    '''
    This class gives a unified description of spin operators.
    Attributes:
        indices: tuple of Index
            The associated indices of the operator, whose length should be equal to the operator's rank.
        spins: list of SpinMatrix
            The associated spin matrix of the operator, whose length should be equal to the operator's rank.
        rcoords: tuple of 1D ndarray
            The associated real coordinates of the operator.
        icoords: tuple of 1D ndarray
            The associated lattice coordinates of the operator.
        seqs: tuple of integer
            The associated sequences of the operator, whose length should be equal to the operator's rank.
    '''
    def __init__(self,value,indices,spins,rcoords,icoords,seqs):
        '''
        Constructor.
        '''
        super(OperatorS,self).__init__(value)
        self.indices=indices
        self.spins=spins
        self.rcoords=rcoords
        self.icoords=icoords
        self.seqs=seqs
        self.set_id()

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        return 'OperatorS(value=%s, indices=%s, spins=%s, rcoords=%s, icoords=%s, seqs=%s)'%(self.value,self.indices,self.spins,self.rcoords,self.icoords,self.seqs)

    def set_id(self):
        '''
        Set the unique id of this operator.
        '''
        self.id=tuple(list(self.indices)+[spin.id for spin in self.spins])

    @property
    def rank(self):
        '''
        Return the rank of the operator.
        '''
        return len(self.seqs)
