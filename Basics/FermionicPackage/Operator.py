'''
------------------
Fermionic operator
------------------

Fermionic operator, including:
    * classes: FOperator
    * functions: FLinear, FQuadratic, FHubbard
'''

__all__=['FOperator','FLinear','FQuadratic','FHubbard']

from numpy import *
from DegreeOfFreedom import ANNIHILATION,CREATION
from ..Operator import *

class FOperator(Operator):
    '''
    This class gives a unified description of fermionic operators with different ranks.

    Attributes
    ----------
    mode : string
        The tag used to distinguish operators with different types or ranks.
    indices : tuple of Index
        The associated indices of the operator, whose length should be equal to the operator's rank.
    rcoords : tuple of 1D ndarray
        The associated real coordinates of the operator.
    icoords : tuple of 1D ndarray
        The associated lattice coordinates of the operator.
    seqs : tuple of integer, optional
         The associated sequences of the operator, whose length should be equal to the operator's rank.

    Notes
    -----
        * The lengths of `rcoords` and `icoords` are not forced to be equal to the operator's rank because:
            * Some of its rank-1 terms may share the same `rcoord` or `icoord`;
            * The `rcoords` and `icoords` is the whole operator's property instead of each of its rank-1 component.
            However, for operators with the same attribute `mode`, the lengths of their `rcoords` and `icoords` should be fixed and equal to each other.
        * Current supported modes include:
            * 'flinear': rank==1 fermionic operators
                The single particle operators.
            * 'fquadratic': rank==2 fermionic operators
                Only one rcoord and one icoord are needed which are identical to the bond's rcoord and icoord where the quadratic operator is defined.
            * 'fhubbard': rank==4 fermionic operators
                Only one rcoord and icoord is needed because Hubbard operators are always on-site ones.
    '''

    def __init__(self,mode,value,indices,rcoords,icoords,seqs=None):
        '''
        Constructor.

        Parameters
        ----------
        mode : string
            The tag used to distinguish operators with different types or ranks.
        indices : tuple of Index
            The associated indices of the operator, whose length should be equal to the operator's rank.
        rcoords : tuple of 1D ndarray
            The associated real coordinates of the operator.
        icoords : tuple of 1D ndarray
            The associated lattice coordinates of the operator.
        seqs : tuple of integer, optional
             The associated sequences of the operator, whose length should be equal to the operator's rank.
        '''
        super(FOperator,self).__init__(value)
        self.mode=mode
        self.indices=tuple(indices)
        self.rcoords=tuple([array(obj) for obj in rcoords])
        self.icoords=tuple([array(obj) for obj in icoords])
        if seqs is not None: self.seqs=tuple(seqs)

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        result=[]
        result.append('FOperator(')
        result.append('mode=%s'%self.mode)
        result.append(', value=%s'%self.value)
        result.append(', indices=%s'%(self.indices,))
        result.append(', rcoords=%s'%(self.rcoords,))
        result.append(', icoords=%s'%(self.icoords,))
        if hasattr(self,'seqs'): result.append(', seqs=%s'%(self.seqs,))
        result.append(')')
        return ''.join(result)

    @property
    def id(self):
        '''
        The unique id of this operator.
        '''
        return self.indices+tuple(['%f'%rcoord for rcoord in concatenate(self.rcoords)])

    @property
    def dagger(self):
        '''
        The dagger, i.e. the Hermitian conjugate of an operator.
        '''
        return FOperator(
                mode=           self.mode,
                value=          conjugate(self.value),
                indices=        reversed([obj.replace(nambu=1-obj.nambu) for obj in self.indices]),
                rcoords=        reversed(list(self.rcoords)),
                icoords=        reversed(list(self.icoords)),
                seqs=           reversed(list(self.seqs)) if hasattr(self,'seqs') else None
                )

    @property
    def rank(self):
        '''
        Return the rank of the operator.
        '''
        return len(self.indices)

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

def FLinear(value,indices,rcoords,icoords,seqs=None):
    '''
    A specialized constructor to create an Operator instance with mode='flinear'.
    '''
    return FOperator('flinear',value,indices,rcoords,icoords,seqs)

def FQuadratic(value,indices,rcoords,icoords,seqs=None):
    '''
    A specialized constructor to create an Operator instance with mode='fquadratic'.
    '''
    return FOperator('fquadratic',value,indices,rcoords,icoords,seqs)

def FHubbard(value,indices,rcoords,icoords,seqs=None):
    '''
    A specialized constructor to create an Operator instance with mode='fhubbard'.
    '''
    return FOperator('fhubbard',value,indices,rcoords,icoords,seqs)
