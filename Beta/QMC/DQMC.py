'''
'''

__all__=['B','Bs']


import numpy as np
from numpy.linalg import svd
from copy import copy

class B(object):
    '''
    '''
    
    def __init__(self,U,S,V):
        self.U=U
        self.S=S
        self.V=V

    def __str__(self):
        return 'U,S,V:\n%s\n%s\n%s'%(self.U,self.S,self.V)

    @property
    def M(self):
        return np.einsum('ij,j,jk->ik',self.U,self.S,self.V)

    def ldot(self,M):
        U,S,V=svd(np.einsum('ij,jk,k->ik',M,self.U,self.S))
        V=V.dot(self.V)
        return B(U,S,V)

    def rdot(self,M):
        U,S,V=svd(np.einsum('i,ij,jk->ik',self.S,self.V,M))
        U=self.U.dot(U)
        return B(U,S,V)

class Bs(list):
    '''
    '''

    def __init__(self,T,Vs,pos=0):
        if pos>len(Vs) or pos<0:
            raise ValueError('Bs __init__ error: pos(%s) should be in range(0,%s).'%(pos,len(Vs)))
        self.pos=pos
        self.extend([None]*len(Vs))
        Vs=np.array(Vs)
        for i,V in enumerate(Vs[0:pos]):
            if i==0:
                self[i]=B(*svd(V.dot(T)))
            else:
                self[i]=self[i-1].ldot(V.dot(T))
        for i,V in enumerate(Vs[pos:][::-1]):
            if i==0:
                self[len(Vs)-1-i]=B(*svd(V.dot(T)))
            else:
                self[i]=self[len(Vs)-1-i]=self[len(Vs)-i].rdot(V.dot(T))

    def __rshift__(self,M):
        return copy(self).__irshift__(M)

    def __irshift__(self,M):
        if self.pos>0:
            if self.pos==len(self):
                self[self.pos-1]=B(*svd(M))
            else:
                self[self.pos-1]=self[self.pos].rdot(M)
            self.pos=self.pos-1
            return self
        else:
            raise ValueError('Bs >> error: pos is 0 and cannot be right shifted any more.')

    def __lshift__(self,M):
        return copy(self).__ilshift__(M)

    def __ilshift__(self,M):
        if self.pos<len(self):
            if self.pos==0:
                self[self.pos]=B(*svd(M))
            else:
                self[self.pos]=self[self.pos-1].ldot(M)
            self.pos=self.pos+1
            return self
        else:
            raise ValueError("Bs '<<' error: pos is %s and cannot be left shifted any more."%(len(self)))

    @property
    def G(self):
        if self.pos in (0,len(self)):
            B=self[self.pos] if self.pos==0 else self[self.pos-1]
            U,S,V=svd(B.V.dot(B.U).transpose().conjugate()+np.diag(B.S))
            U,V=B.U.dot(U).transpose().conjugate(),V.dot(B.V).transpose().conjugate()
            return np.einsum('ij,j,jk->ik',V,1.0/S,U)
        else:
            BR,BL=self[self.pos-1],self[self.pos]
            U,S,V=svd(BL.V.dot(BR.U).transpose().conjugate()+np.einsum('i,ij,jk,k->ik',BR.S,BR.V,BL.U,BL.S))
            U,V=BR.U.dot(U).transpose().conjugate(),V.dot(BL.V).transpose().conjugate()
            return np.einsum('ij,j,jk->ik',V,1.0/S,U)


