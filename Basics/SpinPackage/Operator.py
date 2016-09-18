'''
Spin Operator, including:
1) classes: OperatorS
'''

__all__=['OperatorS']

from numpy import *
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
        if hasattr(self,'rcoords'):
            self.id=tuple(list(self.indices)+[spin.id for spin in self.spins]+['%f'%i for i in concatenate(self.rcoords)])
        else:
            self.id=tuple(list(self.indices)+[spin.id for spin in self.spins])

    @property
    def rank(self):
        '''
        Return the rank of the operator.
        '''
        return len(self.indices)

    def decomposition(self,table1,table2):
        '''
        Decompose an operator into three parts, the coefficient, and two suboperators whose indices are in table1 and table2 respectively.
        Parameters:
            table1,table2: Table
                The index sequence table.
        Returns: number,OperatorS,OperatorS
            The coefficient and the two suboperators.
        '''
        indices1,indices2=[],[]
        spins1,spins2=[],[]
        rcoords1,rcoords2=([],[]) if hasattr(self,'rcoords') else (None,None)
        icoords1,icoords2=([],[]) if hasattr(self,'icoords') else (None,None)
        seqs1,seqs2=[],[]
        for i,(index,spin) in enumerate(zip(self.indices,self.spins)):
            if index in table1:
                indices1.append(index)
                spins1.append(spin)
                seqs1.append(table1[index])
                if hasattr(self,'rcoords'): rcoords1.append(self.rcoords[i])
                if hasattr(self,'icoords'): icoords1.append(self.rcoords[i])
            if index in table2:
                indices2.append(index)
                spins2.append(spin)
                seqs2.append(table2[index])
                if hasattr(self,'rcoords'): rcoords2.append(self.rcoords[i])
                if hasattr(self,'icoords'): icoords2.append(self.icoords[i])
        opt1=OperatorS(value=1.0,indices=indices1,spins=spins1,rcoords=rcoords1,icoords=icoords1,seqs=seqs1)
        opt2=OperatorS(value=1.0,indices=indices2,spins=spins2,rcoords=rcoords2,icoords=icoords2,seqs=seqs2)
        return operator.value,opt1,opt2
