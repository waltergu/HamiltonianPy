'''
Spin Operator, including:
1) classes: OperatorS
'''

__all__=['OperatorS']

from ..Operator import *

class OperatorS(Operator):
    '''
    This class gives a unified description of spin operators.
    '''
    def __init__(self,value,indices,spins,rcoords,icoords,seqs):
        self.value=value
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
