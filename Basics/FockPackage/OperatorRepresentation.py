'''
---------------------------------------------------
Fermionic/Hard-core-bosonic operator representation
---------------------------------------------------

Fermionic/Hard-core-bosonic operator representation, including:
    * functions: foptrep
'''

__all__=['foptrep']

import numpy as np
from .Basis import sequence
from .Operator import FOperator,BOperator
from scipy.sparse import csr_matrix
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
    value,nambus,seqs=dtype(operator.value),(np.array([index.nambu for index in operator.indices])>0)[::-1],np.array(operator.seqs)[::-1]
    if operator.rank%2==0:
        content=foptrepeven(value,nambus,seqs,basis.table,basis.nbasis,0 if isinstance(operator,FOperator) else 1)
        result=csr_matrix(content,shape=(basis.nbasis,basis.nbasis))
    else:
        assert len(basis)==2
        content=foptrepodd(value,nambus,seqs,basis[0].table,basis[1].table,basis[0].nbasis,0 if isinstance(operator,FOperator) else 1)
        result=csr_matrix(content,shape=(basis[0].nbasis,basis[1].nbasis))
    return result.T if transpose else result

@jit("Tuple((pyobject,int64[:],int64[:]))(pyobject,boolean[:],int64[:],int64[:],int64,int64)")
def foptrepeven(value,nambus,seqs,table,nbasis,mode):
    ndata,data,indices,indptr=0,np.zeros(nbasis,dtype=value.dtype),np.zeros(nbasis,dtype=np.int64),np.zeros(nbasis+1,dtype=np.int64)
    eye,temp=int(1),np.zeros(len(seqs)+1,dtype=np.int64)
    for i in range(nbasis):
        indptr[i]=ndata
        temp[0]=i if len(table)==0 else table[i]
        for j in range(len(seqs)):
            if bool(temp[j]&eye<<seqs[j])==nambus[j]: break
            temp[j+1]=temp[j]|eye<<seqs[j] if nambus[j] else temp[j]&~(eye<<seqs[j])
        else:
            nsign=0
            if mode==0:
                for j in range(len(seqs)):
                    for k in range(seqs[j]):
                        if temp[j]&eye<<k: nsign+=1
            indices[ndata]=sequence(temp[-1],table)
            data[ndata]=(-1)**nsign*value
            ndata+=1
    indptr[-1]=ndata
    return data,indices,indptr

@jit("Tuple((pyobject,int64[:],int64[:]))(pyobject,boolean[:],int64[:],int64[:],int64[:],int64,int64)")
def foptrepodd(value,nambus,seqs,table1,table2,nbasis,mode):
    ndata,data,indices,indptr=0,np.zeros(nbasis,dtype=value.dtype),np.zeros(nbasis,dtype=np.int64),np.zeros(nbasis+1,dtype=np.int64)
    eye,temp=int(1),np.zeros(len(seqs)+1,dtype=np.int64)
    for i in range(nbasis):
        indptr[i]=ndata
        temp[0]=i if len(table1)==0 else table1[i]
        for j in range(len(seqs)):
            if bool(temp[j]&eye<<seqs[j])==nambus[j]: break
            temp[j+1]=temp[j]|eye<<seqs[j] if nambus[j] else temp[j]&~(eye<<seqs[j])
        else:
            nsign=0
            if mode==0:
                for j in range(len(seqs)):
                    for k in range(seqs[j]):
                        if temp[j]&eye<<k: nsign+=1
            indices[ndata]=sequence(temp[-1],table2)
            data[ndata]=(-1)**nsign*value
            ndata+=1
    indptr[-1]=ndata
    return data,indices,indptr
