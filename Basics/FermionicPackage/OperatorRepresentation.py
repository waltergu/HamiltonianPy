'''
---------------------------------
Fermionic operator representation
---------------------------------

Fermionic operator representation, including:
    * functions: foptrep
'''

__all__=['foptrep']

from numpy import *
from DegreeOfFreedom import *
from Operator import *
from Basis import *
from scipy.sparse import *
from numba import jit

def foptrep(operator,basis,transpose=False,dtype=complex128):
    '''
    This function returns the csr_formed or csc_formed sparse matrix representation of an operator on the occupation number basis.

    Parameters
    ----------
    operator : FOperator
        Three kinds of operators are supported, flinear, fquadratic and fhubbard.
    basis : FBasis or list of FBasis
        * When operator.rank==1 it is a list of FBasis with len==2. Otherwise it is an instance of FBasis.
        * When the input operator represents a pairing term, basis.mode must be "FG" because of the non-conservation of particle numbers.
    transpose : logical, optional
        A flag to tag which form of sparse matrix the result is used. True for csr-formed and False for csc-formed.
    dtype : dtype, optional
        The data type of the non-zero values of the returned sparse matrix.

    Returns
    -------
    csr_matrix.
        The sparse matrix representation of the operator.

    Notes
    -----
        * All of those operators' representations are generated in the real space. 
        * The returned sparse matrix is always constructed by ``csr_matrix(...)`` since a csc-matrix is just a transpose of a csr-formed matrix.
    '''
    assert operator.rank in (1,2,4)
    if operator.rank==1:
        assert len(basis)==2
        return foptrep_1(operator,basis[0],basis[1],transpose,dtype=dtype)
    elif operator.rank==2:
        return foptrep_2(operator,basis,transpose,dtype=dtype)
    else:
        assert operator.is_normal_ordered()
        return foptrep_4(operator,basis,transpose,dtype=dtype)

def foptrep_1(operator,basis1,basis2,transpose,dtype=complex128):
    seq=operator.seqs[0]
    table1,table2=basis1.table,basis2.table
    nbasis1,nbasis2=basis1.nbasis,basis2.nbasis
    data,indices,indptr=zeros(nbasis1,dtype=dtype),zeros(nbasis1,dtype=int32),zeros(nbasis1+1,dtype=int32)
    if operator.indices[0].nambu==CREATION:
        foptrep_1_1(data,indices,indptr,nbasis1,nbasis2,table1,table2,seq)
    else:
        foptrep_1_0(data,indices,indptr,nbasis1,nbasis2,table1,table2,seq)
    if transpose==False:
        return csr_matrix((data*operator.value,indices,indptr),shape=(nbasis1,nbasis2))
    else:
        return csr_matrix((data*operator.value,indices,indptr),shape=(nbasis1,nbasis2)).T

@jit
def foptrep_1_1(data,indices,indptr,nbasis1,nbasis2,table1,table2,seq):
    ndata=0
    for i in xrange(nbasis1):
        indptr[i]=ndata
        rep=i if len(table1)==0 else table1[i]
        if not rep&1<<seq:
            nsign=0
            for j in xrange(seq):
                if rep&1<<j: nsign+=1
            rep=rep|1<<seq
            indices[ndata]=sequence(rep,table2)
            data[ndata]=(-1)**nsign
            ndata+=1
    indptr[nbasis1]=ndata

@jit
def foptrep_1_0(data,indices,indptr,nbasis1,nbasis2,table1,table2,seq):
    ndata=0
    for i in xrange(nbasis1):
        indptr[i]=ndata
        rep=i if len(table1)==0 else table1[i]
        if rep&1<<seq:
            nsign=0
            for j in xrange(seq):
                if rep&1<<j: nsign+=1
            rep=rep&~(1<<seq)
            indices[ndata]=sequence(rep,table2)
            data[ndata]=(-1)**nsign
            ndata+=1
    indptr[nbasis1]=ndata

def foptrep_2(operator,basis,transpose,dtype=complex128):
    seq1,seq2=operator.seqs[0],operator.seqs[1]
    table,nbasis=basis.table,basis.nbasis
    data,indices,indptr=zeros(nbasis,dtype=dtype),zeros(nbasis,dtype=int32),zeros(nbasis+1,dtype=int32)
    if operator.indices[0].nambu==CREATION and operator.indices[1].nambu==ANNIHILATION:
        foptrep_2_10(data,indices,indptr,nbasis,table,seq1,seq2)
    elif operator.indices[0].nambu==ANNIHILATION and operator.indices[1].nambu==CREATION:
        foptrep_2_01(data,indices,indptr,nbasis,table,seq1,seq2)
    elif operator.indices[0].nambu==ANNIHILATION and operator.indices[1].nambu==ANNIHILATION:
        foptrep_2_00(data,indices,indptr,nbasis,table,seq1,seq2)
    else:
        foptrep_2_11(data,indices,indptr,nbasis,table,seq1,seq2)
    if transpose==False:
        return csr_matrix((data*operator.value,indices,indptr),shape=(nbasis,nbasis))
    else:
        return csr_matrix((data*operator.value,indices,indptr),shape=(nbasis,nbasis)).T

@jit
def foptrep_2_10(data,indices,indptr,nbasis,table,seq1,seq2):
    ndata=0
    for i in xrange(nbasis):
        indptr[i]=ndata
        rep=i if len(table)==0 else table[i]
        if rep&1<<seq2:
            rep1=rep&~(1<<seq2)
            if not rep1&(1<<seq1):
                nsign=0
                for j in xrange(seq2):
                    if rep&1<<j: nsign+=1
                for j in xrange(seq1):
                    if rep1&1<<j: nsign+=1
                rep2=rep1|1<<seq1
                indices[ndata]=sequence(rep2,table)
                data[ndata]=(-1)**nsign
                ndata+=1
    indptr[nbasis]=ndata

@jit
def foptrep_2_01(data,indices,indptr,nbasis,table,seq1,seq2):
    ndata=0
    for i in xrange(nbasis):
        indptr[i]=ndata
        rep=i if len(table)==0 else table[i]
        if not rep&1<<seq2:
            rep1=rep|1<<seq2
            if rep1&1<<seq1:
                nsign=0
                for j in xrange(seq2):
                    if rep&1<<j: nsign+=1
                for j in xrange(seq1):
                    if rep1&1<<j: nsign+=1
                rep2=rep1&~(1<<seq1)
                indices[ndata]=sequence(rep2,table)
                data[ndata]=(-1)**nsign
                ndata+=1
    indptr[nbasis]=ndata

@jit
def foptrep_2_00(data,indices,indptr,nbasis,table,seq1,seq2):
    ndata=0
    for i in xrange(nbasis):
        indptr[i]=ndata
        rep=i if len(table)==0 else table[i]
        if rep&1<<seq2:
            rep1=rep&~(1<<seq2)
            if rep1&1<<seq1:
                nsign=0
                for j in xrange(seq2):
                    if rep&1<<j: nsign+=1
                for j in xrange(seq1):
                    if rep1&1<<j: nsign+=1
                rep2=rep1&~(1<<seq1)
                indices[ndata]=sequence(rep2,table)
                data[ndata]=(-1)**nsign
                ndata+=1
    indptr[nbasis]=ndata

@jit
def foptrep_2_11(data,indices,indptr,nbasis,table,seq1,seq2):
    ndata=0
    for i in xrange(nbasis):
        indptr[i]=ndata
        rep=i if len(table)==0 else table[i]
        if not rep&1<<seq2:
            rep1=rep|1<<seq2
            if not rep1&1<<seq1:
                nsign=0
                for j in xrange(seq2):
                    if rep&1<<j: nsign+=1
                for j in xrange(seq1):
                    if rep1&1<<j: nsign+=1
                rep2=rep1|1<<seq1
                indices[ndata]=sequence(rep2,table)
                data[ndata]=(-1)**nsign
                ndata+=1
    indptr[nbasis]=ndata

def foptrep_4(operator,basis,transpose,dtype=complex128):
    table,nbasis=basis.table,basis.nbasis
    data,indices,indptr=zeros(nbasis,dtype=dtype),zeros(nbasis,dtype=int32),zeros(nbasis+1,dtype=int32)
    seq1,seq2,seq3,seq4=operator.seqs[0],operator.seqs[1],operator.seqs[2],operator.seqs[3]
    foptrep_4_1100(data,indices,indptr,nbasis,table,seq1,seq2,seq3,seq4)
    if transpose==False:
        return csr_matrix((data*operator.value,indices,indptr),shape=(nbasis,nbasis))
    else:
        return csr_matrix((data*operator.value,indices,indptr),shape=(nbasis,nbasis)).T

@jit
def foptrep_4_1100(data,indices,indptr,nbasis,table,seq1,seq2,seq3,seq4):
    ndata=0
    for i in xrange(nbasis):
        indptr[i]=ndata
        rep=i if len(table)==0 else table[i]
        if rep&1<<seq4:
            rep1=rep&~(1<<seq4)
            if rep1&1<<seq3:
                rep2=rep1&~(1<<seq3)
                if not rep2&1<<seq2:
                    rep3=rep2|1<<seq2
                    if not rep3&1<<seq1:
                        nsign=0
                        for j in xrange(seq4):
                            if rep&1<<j: nsign+=1
                        for j in xrange(seq3):
                            if rep1&1<<j: nsign+=1
                        for j in xrange(seq2):
                            if rep2&1<<j: nsign+=1
                        for j in xrange(seq1):
                            if rep3&1<<j: nsign+=1
                        rep4=rep3|1<<seq1
                        indices[ndata]=sequence(rep4,table)
                        data[ndata]=(-1)**nsign
                        ndata+=1
    indptr[nbasis]=ndata
