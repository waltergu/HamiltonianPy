'''
DMRG.
'''

__all__=['kron','Block','IDMRG']

from ..Math.Tensor import *
from ..Math.MPS import *
from ..Basics import *
from copy import copy,deepcopy
import numpy as np
import scipy.sparse as ssp

def kron(m1,m2,qns1=None,qns2=None,qns=None,target=None,format='csr'):
    if qns1 is None and qns2 is None:
        result=ssp.kron(m1,m2,format=format)
        if format=='csr':result.eliminate_zeros()
        return result
    elif qns1 is not None and qns2 is not None and qns is not None:
        if m1.shape!=(qns1.n,qns1.n) or m2.shape!=(qns2.n,qns2.n) or qns.n!=qns1.n*qns2.n:
            raise ValueError("kron error: the matrices and the quantum number collections don't match.")
        P=ssp.coo_matrix((np.ones(qns.n),(np.array(xrange(0,qns.n)),qns.permutation)),shape=(qns.n,qns.n))
        result=P.dot(ssp.kron(m1,m2,format='csr').dot(P.T))
        if target is not None:
            if isinstance(target,QuantumNumber):
                result=result[qns[target],qns[target]]
            else:
                buff=[]
                for qn in target:
                    buff.append(result[qns[qn],qns[qn]])
                result=ssp.block_diag(buff,format=format)
        if format=='csr':result.eliminate_zeros()
        return result
    else:
        raise ValueError('kron error: all of or none of qns1, qns2 and qns should be None.')

class Block(object):
    '''
    '''
    def __init__(self,length,lattice,config,qns,mps,H):
        self.length=length
        self.lattice=lattice
        self.config=config
        self.qns=qns
        self.mps=mps
        self.H=H

    def combination(self,other,target=None):
        pass

class IDMRG(Engine):
    '''
    '''
    def __init__(self,name,lattice,config,term,**karg):
        self.name=name
        self.lattice=lattice
        self.config=config
        self.table=config.table
        self.term=term
        self.sys=Block()
        self.env=Block()

    def proceed(self):
        self.enlarge_lattice()
        self.find_operators()
        self.set_matrix()
        self.find_gs()
        self.truncate()

    def enlarge_lattice(self):
        sp,ep=self.lattice.point.values[0],self.lattice.point.values[0]
