'''
Block for DMRG algorithm, including:
1) functions: kron
2) classes: Block
'''

__all__=['kron','Block']

from ..Basics import *
from copy import copy,deepcopy
from numpy import array,ones
import scipy.sparse as sp

def kron(m1,m2,qns1=None,qns2=None,qns=None,target=None,format='csr'):
    '''
    Kronecker product of two matrices.
    Parameters:
        m1,m2: 2d ndarray-like
            The matrices.
        qns1,qns2: QuantumNumberCollection, optional
            The corresponding quantum number collections of the two matrices.
        qns: QuantumNumberCollection, optional
            The corresponding quantum number collection of the product.
        target: QuantumNumber/list of QuantumNumber
            The target subspace of the product.
        format: string, optional
            The format of the product.
    Returns: sparse matrix whose format is specified by the parameter format
        The product.
    '''
    if qns1 is None and qns2 is None:
        result=sp.kron(m1,m2,format=format)
        if format=='csr':result.eliminate_zeros()
        return result
    elif qns1 is not None and qns2 is not None and qns is not None:
        if m1.shape!=(qns1.n,qns1.n) or m2.shape!=(qns2.n,qns2.n) or qns.n!=qns1.n*qns2.n:
            raise ValueError("kron error: the matrices and the quantum number collections don't match.")
        P=sp.coo_matrix((ones(qns.n),(range(qns.n),qns.permutation)),shape=(qns.n,qns.n))
        result=P.dot(sp.kron(m1,m2,format='csr').dot(P.T))
        if target is not None:
            if isinstance(target,QuantumNumber):
                result=result[qns[target],qns[target]]
            else:
                buff=[]
                for qn in target:
                    buff.append(result[qns[qn],qns[qn]])
                result=sp.block_diag(buff,format=format)
        if format=='csr':result.eliminate_zeros()
        return result
    else:
        raise ValueError('kron error: all of or none of qns1, qns2 and qns should be None.')

def decomposition(operator,table1,table2):
    indices1,indices=[],[]
    for index in operator.indices:
        if index in table1: indices1.append(index)
        if index in table2: indices2.append(index)
    opt1=OperatorS(
            value=      1.0,
            indices=    indices1,
            spins=      array(operator.spins)[[operator.indices.index(ind) for ind in indices1]]
            )
    opt2=OperatorS(
            value=      1.0,
            indices=    indices2,
            spins=      array(operator.spins)[[operator.indices.index(ind) for ind in indices2]]
            )
    return operator.value,opt1,opt2

def s_opt_rep_mps(operator,mps,qns=None):
    pass

class Block(object):
    '''
    '''
    def __init__(self,length,lattice,config,terms,qns,mps,H):
        '''
        Constructor.
        '''
        self.length=length
        self.lattice=lattice
        self.config=config
        self.table=config.table()
        self.terms=terms
        self.qns=qns
        self.mps=mps
        self.H=H

    def combination(self,other,target=None):
        '''
        The combination of two block.
        '''
        length=self.length+other.length
        lattice=Lattice(
                name=       self.name,
                points=     self.points.values()+other.points.values(),
                nneighbour= self.nneighbour
                )
        config=copy(self.config)
        config.update(other.config)
        terms=self.terms
        qns=self.qns+other.qns
        mps=self.mps.combination(other.mps)
        connections=Generator(
                bonds=      [bond for bond in lattice.bonds if (bond.spoint.pid in self.lattice.points and bond.epoint.pid in other.lattice.points) or (bond.spoint.pid in other.lattice.points and bond.epoint.pid in self.lattice.bonds)],
                config=     config,
                terms=      terms
                )
        H=None
        for opt in connections.operator.values():
            value,opt1,opt2=decomposition(opt,self.table,other.table)
            m1,m2=s_opt_rep_mps(opt1*value,self.mps,self.qns),s_opt_rep_mps(opt2,self.mps,self.qns)
            if H is None:
                H=kron(m1,m2,self.qns,other.qns,qns,target)
            else:
                H+=kron(m1,m2,self.qns,other.qns,qns,target)
        H+=kron(self.H,other.H,self.qns,other.qns,qns,target)
        return Block(length=length,lattice=lattice,config=config,terms=terms,qns=qns,mps=mps,H=H)

    def truncate(self,U,indices):
        pass
