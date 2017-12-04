'''
-----------------------
Occupation number basis
-----------------------

The basis of fermionic systems in the occupation number representation, including:
    * classes: FBasis
    * function: sequence, FBases
'''

__all__=['FBasis','sequence','FBases']

import numpy as np
from math import factorial
from itertools import combinations
from numba import jit

class FBasis(object):
    '''
    Basis of fermionic systems in the occupation number representation.

    Attributes
    ----------
    mode : 'FS','FP','FG'
        * 'FS': particle-conserved and spin-conserved basis
        * 'FP': particle-conserved and spin-non-conserved basis
        * 'FG': particle-non-conserved and spin-non-conserved basis
    table : 1d ndarray of int
        The table of the binary representations of the basis.
    nbasis : int
        The dimension of the basis.
    nstate : int
        The number of total single-particle states of the basis.
    nparticle : int
        The number of total particles of the basis.
    spinz : half integer
        The z component of the total spin of the basis.
    '''

    def __init__(self,nstate,nparticle=None,spinz=None):
        '''
        Constructor.

        Parameters
        ----------
        nstate : int
            The number of total single-particle states of the basis.
        nparticle : int, optional
            The number of total particles of the basis.
        spinz : half integer, optional
            The z component of the total spin of the basis.
        '''
        assert nstate%2==0
        if nparticle is None:
            assert spinz is None
            self.mode='FG'
            self.table=np.array([])
            self.nbasis=2**nstate
        elif spinz is None:
            self.mode='FP'
            self.table=table_ep(nstate,nparticle,dtype=np.int64)
            self.nbasis=len(self.table)
        else:
            self.mode='FS'
            self.table=table_es(nstate,nparticle,spinz,dtype=np.int64)
            self.nbasis=len(self.table)
        self.nstate=nstate
        self.nparticle=nparticle
        self.spinz=spinz

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return '\n'.join('{:}: {:b}'.format(i,v) for i,v in enumerate(xrange(self.nbasis) if self.mode=='FG' else self.table))

    @property
    def rep(self):
        '''
        The string representation of the basis.
        '''
        if 'FS' in self.mode:
            return '%s(%s,%s,%s)'%(self.mode,self.nstate,self.nparticle,self.spinz)
        elif 'FP' in self.mode:
            return '%s(%s,%s)'%(self.mode,self.nstate,self.nparticle)
        else:
            return '%s(%s)'%(self.mode,self.nstate)

    def replace(self,**karg):
        '''
        Replace `nstate`,`nparticle` or `spinz` of a basis and construct a new one.
        '''
        keys={'nstate','nparticle','spinz'}
        assert set(karg.iterkeys())<=keys
        return FBasis(**{key:karg.get(key,getattr(self,key)) for key in keys})

def table_ep(nstate,nparticle,dtype=np.int64):
    '''
    This function generates the table of binary representations of a particle-conserved and spin-non-conserved basis.
    '''
    result=np.zeros(factorial(nstate)/factorial(nparticle)/factorial(nstate-nparticle),dtype=dtype)
    buff=combinations(xrange(nstate),nparticle)
    for i,v in enumerate(buff):
        basis=0
        for num in v:
            basis+=(1<<num)
        result[i]=basis
    result.sort()
    return result

def table_es(nstate,nparticle,spinz,dtype=np.int64):
    '''
    This function generates the table of binary representations of a particle-conserved and spin-conserved basis.
    '''
    n,nup,ndw=nstate/2,(nparticle+int(2*spinz))/2,(nparticle-int(2*spinz))/2
    result=np.zeros(factorial(n)/factorial(nup)/factorial(n-nup)*factorial(n)/factorial(ndw)/factorial(n-ndw),dtype=dtype)
    buff_up=list(combinations(xrange(1,2*n,2),nup))
    buff_dw=list(combinations(xrange(0,2*n,2),ndw))
    count=0
    for vup in buff_up:
        buff=0
        for num in vup:
            buff+=(1<<num)
        for vdw in buff_dw:
            basis=buff
            for num in vdw:
                basis+=(1<<num)
            result[count]=basis
            count+=1
    result.sort()
    return result

@jit
def sequence(rep,table):
    '''
    This function returns the sequence of a basis in the basis table.

    Parameters
    ----------
    rep : integer
        The binary representation of a basis.
    table : 1d ndarray of integers
        The basis table.

    Returns
    -------
    integer
        The corresponding sequence of the basis.
    '''
    if len(table)==0:
        return rep
    else:
        lb=0;ub=len(table);count=0
        result=(lb+ub)/2
        while table[result]!=rep:
            count+=1
            if table[result]>rep:
                ub=result
            else:
                lb=result
            if 2**(count-2)>len(table): break
            result=(lb+ub)/2
        else:
            return result
        raise ValueError('sequence error: the input rep is not in the table.')

def FBases(mode,nstate,select=None):
    '''
    This function returns a list of FBasis with the input mode and nstate.

    Parameters
    ----------
    mode : 'FS','FP','FG'
        The mode of the basis.
    select : callable, optional
        The select function of the basis.

    Returns
    -------
    list of FBasis
        The wanted list of FBasis.
    '''
    assert mode in ('FS','FP','FG') and nstate%2==0
    if mode=='FG':
        return [FBasis(nstate)]
    elif mode=='FP':
        return [FBasis(nstate,n) for n in xrange(nstate+1) if select is None or select(n)]
    else:
        return [FBasis(nstate,n,sz) for n in xrange(nstate+1) for sz in (n/2.0-np.array(xrange(max(n-nstate/2,0),min(n,nstate/2)+1))) if select is None or select(n,sz)]
