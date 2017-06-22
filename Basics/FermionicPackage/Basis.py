'''
-----------------------
Occupation number basis
-----------------------

The basis of fermionic systems in the occupation number representation, including:
    * classes: FBasis
    * function: sequence
'''

__all__=['FBasis','sequence']

from numpy import *
from math import factorial
from itertools import combinations
from numba import jit

class FBasis(object):
    '''
    Basis of fermionic systems in the occupation number representation. It provides a unified description of the three often-encountered cases:
        * particle-non-conserved systems
        * particle-conserved systems
        * spin-conserved systems

    Attributes
    ----------
    mode : string
        A flag to tag the type of the three kinds of fore-mentioned systems:
            * 'FG': particle-non-conserved systems
            * 'FP': particle-conserved systems
            * 'FS': spin-conserved systems
    nstate : 1d ndarray of integers
        An array containing the numbers of states.
    nparticle : 1d ndarray of integers
        An array containing the numbers of particles.
    table : 1d ndarray of integers
        The table of allowed binary basis of the Hilbert space.
    nbasis : integer 
        The dimension of the Hilbert space.
    '''
    
    def __init__(self,tuple=(),up=(),down=(),nstate=0,dtype=int64):
        '''
        Constructor. It can be used in three different ways:
            * ``FBasis(nstate=...,dtype=...)``, which generates a a particle-non-conserved basis.
            * ``FBasis((...,...),dtype=...)``, which generates a a particle-non-conserved basis
            * ``FBasis(up=(...,...),down=(...,...),dtype=...)``, which generates a a particle-non-conserved basis

        Parameters
        ----------
        tuple : 2-tuple, optional
            This tuple contains the information to generate a particle-conserved basis:
                * tuple[0]: integer
                    The number of generalized orbitals.
                * tuple[1]: integer
                    The number of total electrons.
        up,down : 2-tuple, optional
            These two tuples contain the information to generate a spin-conserved basis:
                * up[0]/down[0]: integer
                    The number of spin-up/spin-down orbitals.
                * up[1]/down[1]: intger
                    The number of spin-up/spin-down electrons.
        nstate : integer,optional
            The number of states which is used to generate a particle-non-conserved basis.
        dtype : dtype
            The data type of the basis table.

        Notes
        -----
        If more parameters than needed to generate a certain kind a basis are assigned, this method obeys the following priority to create the instance: "FP" > "FS" > "FG".
        '''
        if len(tuple)==2:
            self.mode="FP"
            self.nstate=array(tuple[0])
            self.nparticle=array(tuple[1])
            self.table=table_ep(tuple[0],tuple[1],dtype=dtype)
            self.nbasis=len(self.table)
        elif len(up)==2 and len(down)==2:
            self.mode="FS"
            self.nstate=array([up[0],down[0]])
            self.nparticle=array([up[1],down[1]])
            self.table=table_es(up,down,dtype=dtype)
            self.nbasis=len(self.table)
        else:
            self.mode="FG"
            self.nstate=array(nstate)
            self.nparticle=array([])
            self.table=array([])
            self.nbasis=2**nstate

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        result=''
        if self.mode=='FG':
            for i in xrange(self.nbasis):
                result+=str(i)+': '+'{0:b}'.format(i)+'\n'
        else:
            for i,v in enumerate(self.table):
                result+=str(i)+': '+'{0:b}'.format(v)+'\n'
        return result

    @property
    def rep(self):
        '''
        The string representation of the basis.
        '''
        if self.mode=='FP':
            return 'FP(%s-%s)'%(self.nstate,self.nparticle)
        elif self.mode=='FS':
            return 'FS(%s-%s,%s-%s)'%(self.nstate[0],self.nparticle[0],self.nstate[1],self.nparticle[1])
        else:
            return 'FG(%s)'%(self.nstate)

def table_ep(nstate,nparticle,dtype=int64):
    '''
    This function generates the binary basis table with nstate orbitals occupied by nparticle electrons.
    '''
    result=zeros(factorial(nstate)/factorial(nparticle)/factorial(nstate-nparticle),dtype=dtype)
    buff=combinations(xrange(nstate),nparticle)
    for i,v in enumerate(buff):
        basis=0
        for num in v:
            basis+=(1<<num)
        result[i]=basis
    result.sort()
    return result

def table_es(up,down,dtype=int64):
    '''
    This function generates the binary basis table according to the up and down tuples.
    '''
    assert up[0]==down[0]
    result=zeros(factorial(up[0])/factorial(up[1])/factorial(up[0]-up[1])*factorial(down[0])/factorial(down[1])/factorial(down[0]-down[1]),dtype=dtype)
    buff_up=list(combinations(xrange(1,2*up[0],2),up[1]))
    buff_dn=list(combinations(xrange(0,2*up[0],2),down[1]))
    count=0
    for vup in buff_up:
        buff=0
        for num in vup:
            buff+=(1<<num)
        for vdn in buff_dn:
            basis=buff
            for num in vdn:
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
            if 2**(count-2)>len(table):
                error=True
                break
            result=(lb+ub)/2
        else:
            return result
        raise ValueError('sequence error: the input rep is not in the table.')
