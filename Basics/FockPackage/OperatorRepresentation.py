'''
---------------------------------------------------
Fermionic/Hard-core-bosonic operator representation
---------------------------------------------------

Fermionic/Hard-core-bosonic operator representation, including:
    * functions: foptrep
'''

__all__=['foptrep']

import numpy as np
from .Basis import *
from .Operator import FOperator,BOperator
from scipy.sparse import *
from numba import jit

def foptrep(operator,basis,transpose=False,dtype=np.complex128):
    '''
    This function returns the csr_formed or csc_formed sparse matrix representation of an fermionic/hard-core-bosonic operator on the occupation number basis.

    Parameters
    ----------
    operator : FOperator/BOperator
        Eight kinds of operators are supported, FLinear, FQuadratic, FHubbard, FCoulomb, BLinear, BQuadratic, BHubbard and BCoulomb.
    basis : FBasis or 2-list of FBasis
        * When operator.rank is odd it should be a 2-list of FBasis. Otherwise it is an instance of FBasis.
        * When the input operator represents a pairing term, basis.mode must be "FG" because of the non-conservation of particle numbers.
    transpose : logical, optional
        A flag to tag which form of sparse matrix the result is used. True for csr-formed and False for csc-formed.
    dtype : dtype, optional
        The data type of the non-zero values of the returned sparse matrix.

    Returns
    -------
    csr_matrix
        The sparse matrix representation of the operator.

    Notes
    -----
        * All of those operators' representations are generated in the real space. 
        * The returned sparse matrix is always constructed by ``csr_matrix(...)`` since a csc-matrix is just a transpose of a csr-formed matrix.
    '''
    assert isinstance(operator,FOperator) or isinstance(operator,BOperator)
    value,nambus,seqs=operator.value,(np.array([index.nambu for index in operator.indices])>0)[::-1],np.array(operator.seqs)[::-1]
    if operator.rank%2==0:
        OPTREP=fermioptrep_even if isinstance(operator,FOperator) else boseoptrep_even
        content=OPTREP(value,nambus,seqs,basis.table,basis.nbasis,dtype)
        result=csr_matrix(content,shape=(basis.nbasis,basis.nbasis))
    else:
        assert len(basis)==2
        OPTREP=fermioptrep_odd if isinstance(operator,FOperator) else boseoptrep_odd
        content=OPTREP(value,nambus,seqs,basis[0].table,basis[1].table,basis[0].nbasis,dtype)
        result=csr_matrix(content,shape=(basis[0].nbasis,basis[1].nbasis))
    return result.T if transpose else result

@jit([  "Tuple((float32[:],int64[:],int64[:]))(pyobject,boolean[:],int64[:],int64[:],int64,pyobject)",
        "Tuple((float64[:],int64[:],int64[:]))(pyobject,boolean[:],int64[:],int64[:],int64,pyobject)",
        "Tuple((complex64[:],int64[:],int64[:]))(pyobject,boolean[:],int64[:],int64[:],int64,pyobject)",
        "Tuple((complex128[:],int64[:],int64[:]))(pyobject,boolean[:],int64[:],int64[:],int64,pyobject)"
])
def fermioptrep_even(value,nambus,seqs,table,nbasis,dtype):
    ndata,data,indices,indptr=0,np.zeros(nbasis,dtype=dtype),np.zeros(nbasis,dtype=np.int64),np.zeros(nbasis+1,dtype=np.int64)
    eye,temp=int(1),np.zeros(len(seqs)+1,dtype=np.int64)
    for i in range(nbasis):
        indptr[i]=ndata
        temp[0]=i if len(table)==0 else table[i]
        for j in range(len(seqs)):
            if bool(temp[j]&eye<<seqs[j])==nambus[j]: break
            temp[j+1]=temp[j]|eye<<seqs[j] if nambus[j] else temp[j]&~(eye<<seqs[j])
        else:
            nsign=0
            for j in range(len(seqs)):
                for k in range(seqs[j]):
                    if temp[j]&eye<<k: nsign+=1
            indices[ndata]=sequence(temp[-1],table)
            data[ndata]=(-1)**nsign*value
            ndata+=1
    indptr[-1]=ndata
    return data,indices,indptr

@jit([  "Tuple((float32[:],int64[:],int64[:]))(pyobject,boolean[:],int64[:],int64[:],int64[:],int64,pyobject)",
        "Tuple((float64[:],int64[:],int64[:]))(pyobject,boolean[:],int64[:],int64[:],int64[:],int64,pyobject)",
        "Tuple((complex64[:],int64[:],int64[:]))(pyobject,boolean[:],int64[:],int64[:],int64[:],int64,pyobject)",
        "Tuple((complex128[:],int64[:],int64[:]))(pyobject,boolean[:],int64[:],int64[:],int64[:],int64,pyobject)"
])
def fermioptrep_odd(value,nambus,seqs,table1,table2,nbasis,dtype):
    ndata,data,indices,indptr=0,np.zeros(nbasis,dtype=dtype),np.zeros(nbasis,dtype=np.int64),np.zeros(nbasis+1,dtype=np.int64)
    eye,temp=int(1),np.zeros(len(seqs)+1,dtype=np.int64)
    for i in range(nbasis):
        indptr[i]=ndata
        temp[0]=i if len(table1)==0 else table1[i]
        for j in range(len(seqs)):
            if bool(temp[j]&eye<<seqs[j])==nambus[j]: break
            temp[j+1]=temp[j]|eye<<seqs[j] if nambus[j] else temp[j]&~(eye<<seqs[j])
        else:
            nsign=0
            for j in range(len(seqs)):
                for k in range(seqs[j]):
                    if temp[j]&eye<<k: nsign+=1
            indices[ndata]=sequence(temp[-1],table2)
            data[ndata]=(-1)**nsign*value
            ndata+=1
    indptr[-1]=ndata
    return data,indices,indptr

@jit([  "Tuple((float32[:],int64[:],int64[:]))(pyobject,boolean[:],int64[:],int64[:],int64,pyobject)",
        "Tuple((float64[:],int64[:],int64[:]))(pyobject,boolean[:],int64[:],int64[:],int64,pyobject)",
        "Tuple((complex64[:],int64[:],int64[:]))(pyobject,boolean[:],int64[:],int64[:],int64,pyobject)",
        "Tuple((complex128[:],int64[:],int64[:]))(pyobject,boolean[:],int64[:],int64[:],int64,pyobject)"
])
def boseoptrep_even(value,nambus,seqs,table,nbasis,dtype):
    ndata,data,indices,indptr=0,np.zeros(nbasis,dtype=dtype),np.zeros(nbasis,dtype=np.int64),np.zeros(nbasis+1,dtype=np.int64)
    eye,temp=int(1),np.zeros(len(seqs)+1,dtype=np.int64)
    for i in range(nbasis):
        indptr[i]=ndata
        temp[0]=i if len(table)==0 else table[i]
        for j in range(len(seqs)):
            if bool(temp[j]&eye<<seqs[j])==nambus[j]: break
            temp[j+1]=temp[j]|eye<<seqs[j] if nambus[j] else temp[j]&~(eye<<seqs[j])
        else:
            indices[ndata]=sequence(temp[-1],table)
            data[ndata]=value
            ndata+=1
    indptr[-1]=ndata
    return data,indices,indptr

@jit([  "Tuple((float32[:],int64[:],int64[:]))(pyobject,boolean[:],int64[:],int64[:],int64[:],int64,pyobject)",
        "Tuple((float64[:],int64[:],int64[:]))(pyobject,boolean[:],int64[:],int64[:],int64[:],int64,pyobject)",
        "Tuple((complex64[:],int64[:],int64[:]))(pyobject,boolean[:],int64[:],int64[:],int64[:],int64,pyobject)",
        "Tuple((complex128[:],int64[:],int64[:]))(pyobject,boolean[:],int64[:],int64[:],int64[:],int64,pyobject)"
])
def boseoptrep_odd(value,nambus,seqs,table1,table2,nbasis,dtype):
    ndata,data,indices,indptr=0,np.zeros(nbasis,dtype=dtype),np.zeros(nbasis,dtype=np.int64),np.zeros(nbasis+1,dtype=np.int64)
    eye,temp=int(1),np.zeros(len(seqs)+1,dtype=np.int64)
    for i in range(nbasis):
        indptr[i]=ndata
        temp[0]=i if len(table1)==0 else table1[i]
        for j in range(len(seqs)):
            if bool(temp[j]&eye<<seqs[j])==nambus[j]: break
            temp[j+1]=temp[j]|eye<<seqs[j] if nambus[j] else temp[j]&~(eye<<seqs[j])
        else:
            indices[ndata]=sequence(temp[-1],table2)
            data[ndata]=value
            ndata+=1
    indptr[-1]=ndata
    return data,indices,indptr
