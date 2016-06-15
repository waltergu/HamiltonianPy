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
        rcoords: tuple of 1D ndarray, optional
            The associated real coordinates of the operator.
        icoords: tuple of 1D ndarray, optional
            The associated lattice coordinates of the operator.
        seqs: tuple of integer, optional
            The associated sequences of the operator, whose length should be equal to the operator's rank.
    '''
    def __init__(self,value,indices,spins,rcoords=None,icoords=None,seqs=None):
        '''
        Constructor.
        '''
        super(OperatorS,self).__init__(value)
        self.indices=tuple(indices)
        self.spins=tuple(spins)
        if rcoords is not None: self.rcoords=tuple(rcoords)
        if icoords is not None: self.icoords=tuple(icoords)
        if seqs is not None: self.seqs=tuple(seqs)
        self.set_id()

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        result=[]
        result.append('OperatorS(')
        result.append('value=%s'%self.value)
        result.append(', indices=%s'%(self.indices,))
        result.append(', spins=%s'%(self.spins,))
        if hasattr(self,'rcoords'): result.append(', rcoords=%s'%(self.rcoords,))
        if hasattr(self,'icoords'): result.append(', icoords=%s'%(self.icoords,))
        if hasattr(self,'seqs'): result.append(', seqs=%s'%(self.seqs,))
        result.append(')')
        return ''.join(result)

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
