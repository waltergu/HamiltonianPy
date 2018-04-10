'''
-------------
Spin operator
-------------

Spin operator, including:
    * classes: SOperator
'''

__all__=['SOperator']

from ..Operator import *
import itertools as it

class SOperator(Operator):
    '''
    This class gives a unified description of spin operators.

    Attributes
    ----------
    indices : tuple of Index
        The associated indices of the operator, whose length should be equal to the operator's rank.
    spins : list of SpinMatrix
        The associated spin matrix of the operator, whose length should be equal to the operator's rank.
    seqs : tuple of integer
        The associated sequences of the operator, whose length should be equal to the operator's rank.
    rcoord : 1d ndarray
        The associated real coordinates of the operator.
    icoord : 1d ndarray
        The associated lattice coordinates of the operator.
    '''

    def __init__(self,value,indices,spins,seqs=None,rcoord=None,icoord=None):
        '''
        Constructor.
        '''
        super(SOperator,self).__init__(value)
        self.indices=tuple(indices)
        self.spins=tuple(spins)
        self.seqs=None if seqs is None else tuple(seqs)
        self.rcoord=rcoord
        self.icoord=icoord

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        result=[]
        result.append('SOperator(')
        result.append('value=%s'%self.value)
        result.append(', indices=%s'%(self.indices,))
        result.append(', spins=%s'%(self.spins,))
        result.append(')')
        return ''.join(result)

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        result=[]
        result.append('SOperator(')
        result.append('value=%s'%self.value)
        result.append(', indices=%s'%(self.indices,))
        result.append(', spins=(%s)'%('\n'.join(str(spin) for spin in self.spins)))
        if self.seqs is not None: result.append(', seqs=%s'%(self.seqs,))
        if self.rcoord is not None: result.append(', rcoord=%s'%self.rcoord)
        if self.icoord is not None: result.append(', icoord=%s'%self.icoord)
        result.append(')')
        return ''.join(result)

    @property
    def id(self):
        '''
        The unique id of this operator.
        '''
        if self.rcoord is not None:
            return tuple(it.chain(self.indices,(spin.tag for spin in self.spins),(round(f,6) for f in self.rcoord)))
        else:
            return tuple(it.chain(self.indices,(spin.tag for spin in self.spins)))

    @property
    def rank(self):
        '''
        Return the rank of the operator.
        '''
        return len(self.indices)
