'''
--------------------------
Fermionic/bosonic operator
--------------------------

Fermionic/bosonic operator, including:
    * classes: FockOperator, FockLinear, FockQuadratic, FockHubbard, FockCoulomb,
               FOperator, FLinear, FQuadratic, FHubbard, FCoulomb, 
               BOperator, BLinear, BQuadratic, BHubbard, BCoulomb
'''

__all__=[   'FockOperator','FockLinear','FockQuadratic','FockHubbard','FockCoulomb',
            'FOperator','FLinear','FQuadratic','FHubbard','FCoulomb',
            'BOperator','BLinear','BQuadratic','BHubbard','BCoulomb'
            ]

from ..Operator import *
from ..Utilities import parity
from DegreeOfFreedom import ANNIHILATION,CREATION
import numpy as np

class FockOperator(Operator):
    '''
    This class gives a unified description of fermionic/bosonic operators with different ranks.

    Attributes
    ----------
    indices : tuple of Index
        The associated indices of the operator, whose length should be equal to the operator's rank.
    seqs : tuple of int
        The associated sequences of the operator, whose length should be equal to the operator's rank.
    rcoord : 1d ndarray
        The associated real coordinates of the operator.
    icoord : 1d ndarray
        The associated lattice coordinates of the operator.
    '''

    def __init__(self,value,indices,seqs=None,rcoord=None,icoord=None):
        '''
        Constructor.

        Parameters
        ----------
        value : number
            The coefficient of the operator.
        indices : tuple of Index
            The associated indices of the operator.
        seqs : tuple of int, optional
             The associated sequences of the operator.
        rcoord : 1d ndarray, optional
            The real coordinates of the operator.
        icoord : 1d ndarray, optional
            The lattice coordinates of the operator.
        '''
        super(FockOperator,self).__init__(value)
        self.indices=tuple(indices)
        self.seqs=None if seqs is None else tuple(seqs)
        if self.seqs is not None: assert len(self.seqs)==len(self.indices)
        self.rcoord=None if rcoord is None else np.array(rcoord)
        self.icoord=None if icoord is None else np.array(icoord)

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        result=[]
        result.append('%s('%self.__class__.__name__)
        result.append('%s'%self.value)
        result.append(', %s'%(self.indices,))
        if self.seqs is not None: result.append(', seqs=%s'%(self.seqs,))
        if self.rcoord is not None: result.append(', rcoord=%s'%self.rcoord)
        if self.icoord is not None: result.append(', icoord=%s'%self.icoord)
        result.append(')')
        return ''.join(result)

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        result=[]
        result.append('%s('%self.__class__.__name__)
        result.append('value=%s'%self.value)
        result.append(', indices=(%s)'%(','.join(str(index) for index in self.indices)))
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
        return self.indices if self.rcoord is None else self.indices+tuple([round(f,6) for f in self.rcoord])

    @property
    def rank(self):
        '''
        Return the rank of the operator.
        '''
        return len(self.indices)

    def isnormalordered(self):
        '''
        Judge whether an operator is normal ordered.
        '''
        flag=True
        for index in self.indices:
            if index.nambu==ANNIHILATION: flag=False
            if not flag and index.nambu==CREATION: return False
        return True

    def isHermitian(self):
        '''
        Judge whether an operator is Hermitian.
        '''
        return self==self.dagger

class FockLinear(FockOperator):
    '''
    Linear Fock operator.
    '''

    def __init__(self,value,index,seq=None,rcoord=None,icoord=None):
        '''
        Constructor.

        Parameters
        ----------
        value : number
            The coefficient of the linear operator.
        index : Index
            The index of the linear operator.
        seq : int, optional
            The associated sequence of the linear operator.
        rcoord : 1d ndarray, optional
            The real coordinates of the operator.
        icoord : 1d ndarray, optional
            The lattice coordinates of the operator.
        '''
        super(FockLinear,self).__init__(value,indices=(index,),seqs=None if seq is None else (seq,),rcoord=rcoord,icoord=icoord)

    @property
    def index(self):
        '''
        The index of the linear operator.
        '''
        return self.indices[0]

    @property
    def seq(self):
        '''
        The associated sequence of the linear operator.
        '''
        return None if self.seqs is None else self.seqs[0]

    @property
    def dagger(self):
        '''
        The Hermitian conjugate of the linear operator.
        '''
        return type(self)(
                value=      np.conjugate(self.value),
                index=      self.index.replace(nambu=1-self.index.nambu),
                seq=        self.seq,
                rcoord=     self.rcoord,
                icoord=     self.icoord
                )

class FockQuadratic(FockOperator):
    '''
    Quadratic Fock operator.
    '''

    def __init__(self,value,indices,seqs=None,rcoord=None,icoord=None):
        '''
        Constructor. See FockOperator.__init__ for details.
        '''
        assert len(indices)==2
        super(FockQuadratic,self).__init__(value,indices,seqs=seqs,rcoord=rcoord,icoord=icoord)

    @property
    def dagger(self):
        '''
        The Hermitian conjugate of the quadratic operator.
        '''
        return type(self)(
                value=      np.conjugate(self.value),
                indices=    [index.replace(nambu=1-index.nambu) for index in reversed(self.indices)],
                seqs=       None if self.seqs is None else reversed(self.seqs),
                rcoord=     None if self.rcoord is None else -self.rcoord,
                icoord=     None if self.icoord is None else -self.icoord
                )

class FockHubbard(FockOperator):
    '''
    Fock Hubbard operator.
    '''

    def __init__(self,value,indices,seqs=None,rcoord=None,icoord=None):
        '''
        Constructor. See FockOperator.__init__ for details.
        '''
        assert len(indices)==4
        super(FockHubbard,self).__init__(value,indices,seqs=seqs,rcoord=rcoord,icoord=icoord)

    @property
    def dagger(self):
        '''
        The Hermitian conjugate of the Hubbard operator.
        '''
        return type(self)(
                value=      np.conjugate(self.value),
                indices=    [index.replace(nambu=1-index.nambu) for index in reversed(self.indices)],
                seqs=       None if self.seqs is None else reversed(self.seqs),
                rcoord=     self.rcoord,
                icoord=     self.icoord
                )

class FockCoulomb(FockOperator):
    '''
    Fock density-density interaction operator.
    '''

    def __init__(self,value,indices,seqs=None,rcoord=None,icoord=None):
        '''
        Constructor. See FockOperator.__init__ for details.
        '''
        assert len(indices)==4
        super(FockCoulomb,self).__init__(value,indices,seqs=seqs,rcoord=rcoord,icoord=icoord)

    @property
    def dagger(self):
        '''
        The Hermitian conjugate of the density-density interaction operator.
        '''
        return type(self)(
                value=      np.conjugate(self.value),
                indices=    [index.replace(nambu=1-index.nambu) for index in reversed(self.indices)],
                seqs=       None if self.seqs is None else reversed(self.seqs),
                rcoord=     None if self.rcoord is None else -self.rcoord,
                icoord=     None if self.icoord is None else -self.icoord
        )

class FOperator(FockOperator):
    '''
    Fermionic operator.
    '''
    pass

class FLinear(FockLinear,FOperator):
    '''
    Fermionic linear operator.
    '''
    pass

class FQuadratic(FockQuadratic,FOperator):
    '''
    Fermionic quadratic operator.
    '''
    pass

class FHubbard(FockHubbard,FOperator):
    '''
    Fermionic Hubbard operator.
    '''

    def reorder(self,permutation):
        '''
        Return the reordered fermionic Hubbard operator according to the permutation information.

        Parameters
        ----------
        permutation : list of int
            The permutation of the fermionic operator.

        Returns
        -------
        FHubbard
            The reordered operator.
        '''
        assert len(permutation)==self.rank
        result=FockOperator.__new__(self.__class__)
        super(FockOperator,result).__init__(self.value*parity(permutation))
        result.indices=tuple(self.indices[i] for i in permutation)
        result.seqs=None if self.seqs is None else tuple(self.seqs[i] for i in permutation)
        result.rcoord=self.rcoord
        result.icoord=self.icoord
        return result

class FCoulomb(FockCoulomb,FOperator):
    '''
    Fermionic density-density interaction operator.
    '''
    pass

class BOperator(FockOperator):
    '''
    Bosonic operator.
    '''
    pass

class BLinear(FockLinear,BOperator):
    '''
    Bosonic linear operator.
    '''
    pass

class BQuadratic(FockQuadratic,BOperator):
    '''
    Bosonic quadratic operator.
    '''
    pass

class BHubbard(FockHubbard,BOperator):
    '''
    Bosonic Hubbard operator.
    '''

    def reorder(self,permutation):
        '''
        Return the reordered bosonic Hubbard operator according to the permutation information.

        Parameters
        ----------
        permutation : list of int
            The permutation of the bosonic operator.

        Returns
        -------
        BHubbard
            The reordered operator.
        '''
        assert len(permutation)==self.rank
        result=FockOperator.__new__(self.__class__)
        super(FockOperator,result).__init__(self.value)
        result.indices=tuple(self.indices[i] for i in permutation)
        result.seqs=None if self.seqs is None else tuple(self.seqs[i] for i in permutation)
        result.rcoord=self.rcoord
        result.icoord=self.icoord
        return result

class BCoulomb(FockCoulomb,BOperator):
    '''
    Bosonic density-density interaction operator.
    '''
    pass
