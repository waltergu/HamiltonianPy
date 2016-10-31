'''
Matrix product state, including:
1) classes: MPS, Vidal
'''

__all__=['MPS','Vidal']

import numpy as np
from HamiltonianPy.Math.Tensor import *
from HamiltonianPy.Math.linalg import truncated_svd
from copy import copy,deepcopy

class MPS(list):
    '''
    The general matrix product state.
        For each of its elements: Tensor
            The matrices of the mps.
    Attributes:
        Lambda: Tensor
            The Lambda matrix (singular values) on the connecting link.
        cut: integer
            The index of the connecting link.
        table: dict
            For each of its (key,value) pair,
                key: any hashable object
                    The site label of each matrix in the mps.
                value: integer
                    The index of the corresponding matrix in the mps.
    Note the left-canonical MPS, right-canonical MPS and mixed-canonical MPS are considered as special cases of this form.
    '''
    L,S,R=0,1,2

    def __init__(self,ms,labels=None,Lambda=None,cut=None):
        '''
        Constructor.
        Parameters:
            ms: list of 3d Tensor / 3d ndarray
                The matrices.
            labels: list of 3 tuples
                The labels of the axis of the matrices, thus its length should be equal to that of ms.
                For each label in labels, 
                    label[0],label[1],label[2]: any hashable object
                        The left link / site / right link label of the matrix.
            Lambda: 1d ndarray / 1d Tensor, optional
                The Lambda matrix (singular values) on the connecting link.
            cut: integer, optional
                The index of the connecting link.
        '''
        if (Lambda is None)!=(cut is None):
            raise ValueError('MPS construction error: cut and Lambda should be both or neither be None.')
        elif Lambda is None and cut is None:
            self.Lambda=None
            self.cut=None
        elif cut<0 or cut>len(ms):
            raise ValueError('MPS construction error: the cut(%s) is out of range [0,%s].'%(cut,len(ms)))
        self.table={}
        if labels is None:
            for i,m in enumerate(ms):
                self.append(m)
            if Lambda is not None and cut is not None:
                assert isinstance(Lambda,Tensor)
                self.Lambda=Lambda
                self.cut=cut
        else:
            assert len(ms)==len(labels)
            for m,label in zip(ms,labels):
                self.append(Tensor(m,labels=list(label)))
            if Lambda is not None and cut is not None:
                if cut==0:
                    self.Lambda=Tensor(Lambda,labels=[deepcopy(labels[cut][0])])
                else:
                    self.Lambda=Tensor(Lambda,labels=[deepcopy(labels[cut-1][2])])
                self.cut=cut

    @staticmethod
    def from_state(state,shapes,labels,cut=0,nmax=None,tol=None,print_truncation_err=False):
        '''
        Convert the normal representation of a state to the matrix product representation.
        Parameters:
            state: 1d ndarray
                The normal representation of a state.
            shapes: list of integers
                The physical dimension of every site.
            labels: list of 3-tuple
                Please see MPS.__init__ for details.
            cut: integer, optional
                The index of the connecting link.
            namx,tol,print_truncation_err: optional
                For details, please refer to HamiltonianPy.Math.linalg.truncated_svd.
        Returns: MPS
            The corresponding mixed-canonical mps.
        '''
        if len(state.shape)!=1:
            raise ValueError("MPS.from_state error: the original state must be a pure state.")
        ms,nd=[None]*len(shapes),1
        for i in xrange(cut):
            u,s,v=truncated_svd(state.reshape((nd*shapes[i],-1)),full_matrices=False,nmax=nmax,tol=tol,print_truncation_err=print_truncation_err)
            ms[i]=u.reshape((nd,shapes[i],-1))
            if i==cut-1:
                if cut==len(shapes):
                    Lambda=v.transpose().dot(s)
                else:
                    Lambda,state=s,v
            else:
                state=np.einsum('i,ij->ij',s,v)
            nd=len(s)
        nd=1
        for i in xrange(len(shapes)-1,cut-1,-1):
            if i==cut:
                if cut==0:
                    u,s,v=truncated_svd(state.reshape((-1,shapes[i]*nd)),full_matrices=False,nmax=nmax,tol=tol,print_truncation_err=print_truncation_err)
                    ms[i]=v.reshape((-1,shapes[i],nd))
                    Lambda=u.dot(s)
                else:
                    ms[i]=state.reshape((-1,shapes[i],nd))
            else:
                u,s,v=truncated_svd(state.reshape((-1,shapes[i]*nd)),full_matrices=False,nmax=nmax,tol=tol,print_truncation_err=print_truncation_err)
                ms[i]=v.reshape((-1,shapes[i],nd))
                state=np.einsum('ij,j->ij',u,s)
            nd=len(s)
        return MPS(ms,labels,Lambda=Lambda,cut=cut)

    def append(self,m):
        '''
        Overloaded append.
        '''
        assert isinstance(m,Tensor) and m.ndim==3
        list.append(self,m)
        self.table[m.labels[self.S]]=self.nsite-1

    def insert(self,index,m):
        '''
        Overloaded insert.
        '''
        assert isinstance(m,Tensor) and m.ndim==3
        list.insert(self,index,m)
        self.table={m.labels[self.S]:i for i,m in enumerate(self)}

    def extend(self,ms):
        '''
        Overloaded extend.
        '''
        for m in ms:
            self.append(m)

    @property
    def As(self):
        '''
        The A matrices.
        '''
        return self[0:self.cut]

    @property
    def Bs(self):
        '''
        The B matrices.
        '''
        return self[self.cut:self.nsite]

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        result=[str(m) for m in self]
        if self.cut is not None:
            result.insert(self.cut,str(self.Lambda))
        return '\n'.join(result)

    @property
    def nsite(self):
        '''
        The number of total sites.
        '''
        return len(self)

    @property
    def state(self):
        '''
        Convert to the normal representation.
        Returns: two cases,
            1) 1d ndarray
                The MPS is a pure state.
                Its norm is omitted.
            2) 2d ndarray 
                The MPS is a mixed state with the columns being the contained pure states.
                The singular value for each pure state is omitted.
        '''
        if self.cut in (0,self.nsite,None):
            result=contract(*self,sequence='sequential')
        else:
            A,B=contract(*self.As,sequence='sequential'),contract(*self.Bs,sequence='sequential')
            result=contract(A,self.Lambda,B)
        legs=set(result.labels)-set(self.table)
        if len(legs)==0:
            return np.asarray(result).ravel()
        elif len(legs)==2:
            if self[0].shape[MPS.L]>1:
                flag=True
            if self[-1].shape[MPS.R]>1:
                flag=False
            buff,temp=1,1
            for label,n in zip(result.labels,result.shape):
                if label not in self.table and n>1:
                    temp=n
                else:
                    buff*=n
            if flag:
                return np.asarray(result).reshape((temp,buff)).T
            else:
                return np.asarray(result).reshape((buff,temp))
        else:
            raise ValueError('MPS state error: %s link labels%s are left.'%(len(legs),tuple(legs)))

    @property
    def norm(self):
        '''
        The norm of the matrix product state.
        '''
        temp=copy(self)
        temp._reset_(reset=0)
        temp>>=temp.nsite
        return np.asarray(temp.Lambda)

    def _reset_(self,merge='L',reset=None):
        '''
        Reset the mps.
        This function does two things,
        1) merge the Lamdbda matrix on the link to its left neighbouring matrix or right neighbouring matrix acoording to the parameter merge,
        2) reset Lambda and cut acoording to the parameter reset.
        Parameters:
            merge: 'L' or 'R', optional
                When 'L', self.Lambda will be merged into its left neighbouring matrix;
                When 'R', self.Lambda will be merged into its right neighbouring matrix.
            reset: None or an integer, optional
                When None, self.cut and self.Lambda will be reset to None;
                When an integer, self.cut will be reset to this value and self.Lambda will be reset to a scalar Tensor with the data equal to 1.0.
        NOTE: When self.cut==0, the Lambda matrix will be merged with self[0] no matter what merge is, and
              When self.cut==self.nsite, the Lambda matrix will be merged with self[-1] no matter what merge is.
        '''
        if self.cut is not None:
            if merge=='L':
                if self.cut==0:
                    self[0]=contract(self[0],self.Lambda,select=self.Lambda.labels)
                else:
                    self[self.cut-1]=contract(self[self.cut-1],self.Lambda,select=self.Lambda.labels)
            elif merge=='R':
                if self.cut==self.nsite:
                    self[-1]=contract(self[-1],self.Lambda,select=self.Lambda.labels)
                else:
                    self[self.cut]=contract(self.Lambda,self[self.cut%self.nsite],select=self.Lambda.labels)
            else:
                raise ValueError("MPS _reset_ error: merge must be 'L' or 'R' but now it is %s."%(merge))
        if reset is None:
            self.cut=None
            self.Lambda=None
        elif reset>=0 and reset<=self.nsite:
            self.cut=reset
            self.Lambda=Tensor(1.0,labels=[])
        else:
            raise ValueError("MPS _reset_ error: reset(%s) should be None or in the range [%s,%s]"%(reset,0,self.nsite))

    def _set_ABL_(self,m,Lambda):
        '''
        Set the matrix at a certain position and the Lambda of an mps.
        Parameters:
            m: Tensor
                The matrix at a certain position of the mps.
            Lambda: Tensor
                The singular values at the connecting link of the mps.
        '''
        L,S,R=m.labels[MPS.L],m.labels[MPS.S],m.labels[MPS.R]
        pos=self.table[S]
        self[pos]=m
        self.Lambda=Lambda
        if Lambda.labels[0]==L:
            self.cut=pos
        elif Lambda.labels[0]==R:
            self.cut=pos+1
        else:
            raise ValueError("MPS _set_ABL_ error: the labels of m(%s) and Lambda(%s) do not match."%(m.labels,Lambda.labels))

    def _set_B_and_lmove_(self,M,nmax=None,tol=None,print_truncation_err=True):
        '''
        Set the B matrix at self.cut and move leftward.
        Parameters:
            M: Tensor
                The tensor used to set the B matrix.
            nmax: integer, optional
                The maximum number of singular values to be kept. 
            tol: float64, optional
                The truncation tolerance.
            print_truncation_err: logical, optional
                If it is True, the truncation err will be printed.
        '''
        if self.cut==0:
            raise ValueError('MPS _set_B_and_lmove_ error: the cut is already zero.')
        L,S,R=M.labels[self.L],M.labels[self.S],M.labels[self.R]
        u,s,v=M.svd([L],(L,),[S,R],nmax=nmax,tol=tol,print_truncation_err=print_truncation_err)
        v.relabel(news=[L],olds=[(L,)])
        self[self.cut-1]=v
        if self.cut==1:
            if len(s)>1:
                raise ValueError('MPS _set_B_and_lmove_ error(not supported operation): the MPS is a mixed state and is to move to the end.')
            self.Lambda=contract(u,s)
        else:
            s.relabel(news=[L],olds=[(L,)])
            self.Lambda=s
            L=self[self.cut-2].labels[MPS.R]
            self[self.cut-2]=contract(self[self.cut-2],u)
            self[self.cut-2].relabel(news=[L],olds=[(L,)])
        self.cut=self.cut-1

    def _set_A_and_rmove_(self,M,nmax=None,tol=None,print_truncation_err=True):
        '''
        Set the A matrix at self.cut and move rightward.
        Parameters:
            M: Tensor
                The tensor used to set the A matrix.
            nmax: integer, optional
                The maximum number of singular values to be kept. 
            tol: float64, optional
                The truncation tolerance.
            print_truncation_err: logical, optional
                If it is True, the truncation err will be printed.
        '''
        if self.cut==self.nsite:
            raise ValueError('MPS _set_A_and_rmove_ error: the cut is already maximum.')
        L,S,R=M.labels[self.L],M.labels[self.S],M.labels[self.R]
        u,s,v=M.svd([L,S],(R,),[R],nmax=nmax,tol=tol,print_truncation_err=print_truncation_err)
        u.relabel(news=[R],olds=[(R,)])
        self[self.cut]=u
        if self.cut==self.nsite-1:
            if len(s)>1:
                raise ValueError('MPS _set_A_and_rmove_ error(not supported operation): the MPS is a mixed state and is to move to the end.')
            self.Lambda=contract(s,v)
        else:
            s.relabel(news=[R],olds=[(R,)])
            self.Lambda=s
            R=self[self.cut+1].labels[MPS.L]
            self[self.cut+1]=contract(v,self[self.cut+1])
            self[self.cut+1].relabel(news=[R],olds=[(R,)])
        self.cut=self.cut+1

    def __ilshift__(self,other):
        '''
        Operator "<<=", which shift the connecting link leftward by a non-negative integer.
        Parameters:
            other: two cases,
                1) integer
                    The number of times that self.cut will move leftward.
                2) 3-tuple
                    tuple[0]: integer
                        The number of times that self.cut will move leftward.
                    tuple[1]: integer
                        The maximum number of singular values to be kept.
                    tuple[2]: float64
                        The truncation tolerance.
        '''
        nmax,tol=None,None
        if isinstance(other,tuple):
            k,nmax,tol=other
        else:
            k=other
        for i in xrange(k):
            M=Tensor(np.asarray(contract(self[self.cut-1],self.Lambda,select=self.Lambda.labels)),labels=self[self.cut-1].labels)
            self._set_B_and_lmove_(M,nmax,tol)
        return self

    def __lshift__(self,other):
        '''
        Operator "<<".
        Parameters:
            other: integer or 3-tuple.
                Please see MPS.__ilshift__ for details.
        '''
        return copy(self).__ilshift__(other)

    def __irshift__(self,other):
        '''
        Operator ">>=", which shift the connecting link rightward by a non-negative integer.
        Parameters:
            other: two cases,
                1) integer
                    The number of times that self.cut will move rightward.
                2) 3-tuple
                    tuple[0]: integer
                        The number of times that self.cut will move rightward.
                    tuple[1]: integer
                        The maximum number of singular values to be kept.
                    tuple[2]: float64
                        The truncation tolerance.
        '''
        nmax,tol=None,None
        if isinstance(other,tuple):
            k,nmax,tol=other
        else:
            k=other
        for i in xrange(k):
            M=Tensor(np.asarray(contract(self.Lambda,self[self.cut],select=self.Lambda.labels)),labels=self[self.cut].labels)
            self._set_A_and_rmove_(M,nmax,tol)
        return self

    def __rshift__(self,other):
        '''
        Operator ">>".
        Parameters:
            other: integer or 3-tuple.
                Please see MPS.__irshift__ for details.
        '''
        return copy(self).__irshift__(other)

    def canonicalization(self,cut):
        '''
        Make the MPS in the mixed canonical form.
        Parameters:
            link: integer
                The cut of the A,B part.
        Returns: MPS
            The mixed canonical MPS.
        '''
        if self.cut<=self.nsite/2:
            self._reset_(reset=self.nsite)
            self<<=self.nsite
            self>>=cut
        else:
            self._reset_(reset=0)
            self>>=self.nsite
            self<<=(self.nsite-self.cut)

    def is_canonical(self):
        '''
        Judge whether each site of the MPS is in the canonical form.
        '''
        result=[]
        for i,M in enumerate(self):
            temp=[np.asarray(M.take(indices=j,axis=self.S)) for j in xrange(M.shape[self.S])]
            buff=None
            for matrix in temp:
                if buff is None:
                    buff=matrix.T.conjugate().dot(matrix) if i<self.cut else matrix.dot(matrix.T.conjugate())
                else:
                    buff+=matrix.T.conjugate().dot(matrix) if i<self.cut else matrix.dot(matrix.T.conjugate())
            result.append((abs(buff-np.identity(M.shape[self.R if i<self.cut else self.L]))<5*10**-14).all())
        return result

    def copy(self,copy_data=False):
        '''
        Make a copy of the mps.
        Parameters:
            copy_data: logical, optional
                When True, both the labels and data of each tensor in this mps will be copied;
                When False, only the labels of each tensor in this mps will be copied.
        Returns: MPS
            The copy of self.
        '''
        ms=[m.copy(copy_data=copy_data) for m in self]
        Lambda=None if self.Lambda is None else self.Lambda.copy(copy_data=copy_data)
        return MPS(ms,Lambda=Lambda,cut=self.cut)

class Vidal(object):
    '''
    The Vidal canonical matrix product state.
    Attributes:
        Gammas: list of Tensor
            The Gamma matrices on the site.
        Lambdas: list of Tensor
            The Lambda matrices (singular values) on the link.
    '''
    L,S,R=0,1,2

    def __init__(self,Gammas,Lambdas,labels=None):
        '''
        Constructor.
        Parameters:
            Gammas: list of 3d ndarray/Tensor
                The Gamma matrices on the site.
            Lamdas: list of 1d ndarray/Tensor
                The Lambda matrices (singular values) on the link.
            labels: list of 3 tuples, optional
                The labels of the axis of the Gamma matrices.
                Its length should be equal to that of Gammas.
                For each label in labels, 
                    label[0],label[1],label[2]: any hashable object
                        The left link / site / right link label of the matrix.
        '''
        assert len(Gammas)==len(Lambdas)+1
        self.Gammas=[]
        self.Lambdas=[]
        if labels is None:
            assert len(Gammas)==len(labels)
            buff=[]
            for i,(Gamma,label) in enumerate(zip(Gammas,labels)):
                assert Gamma.ndim==3
                if i<len(Gammas)-1:
                    buff.append(R)
                self.Gammas.append(Tensor(Gamma,labels=list(label)))
            for Lambda,label in zip(Lambdas,buff):
                assert Lambda.ndim==1
                self.Lambdas.append(Tensor(Lambda,labels=[label]))
        else:
            for Gamma in Gammas:
                assert isinstance(Gamma,Tensor)
                assert Gamma.ndim==3
                self.Gammas.append(Gamma)
            for Lambda in Lambdas:
                assert isinstance(Lambda,Tensor)
                assert Lambda.ndim==1
                self.Lambdas.append(Lambda)

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        result=[]
        for i,Gamma in enumerate(self.Gammas):
            result.append(str(Gamma))
            if i<len(self.Gammas)-1:
                result.append(str(self.Lambdas[i]))
        return '\n'.join(result)

    @property
    def nsite(self):
        '''
        The number of total sites.
        '''
        return len(self.Gammas)

    @property
    def state(self):
        '''
        Convert to the normal representation.
        Returns: 1d ndarray
            The corresponding normal representation of the state.
        '''
        result=None
        for i,Gamma in enumerate(self.Gammas):
            if result is None:
                result=Gamma
            else:
                result=contract(result,self.Lambdas[i-1],Gamma)
        return np.asarray(result).ravel()

    def to_mixed(self,cut):
        '''
        Convert to the mixed MPS representation.
        Parameters:
            cut: integer
                The index of the connecting link.
        Retruns: MPS
            The corresponding mixed MPS.
        '''
        ms,labels,Lambda=[],[],None
        shape=[1]*3
        shape[self.S]=-1
        for i,Gamma in enumerate(self.Gammas):
            L,S,R=Gamma.labels[self.L],Gamma.labels[self.S],Gamma.labels[self.R]
            labels.append((L,S,R))
            if i<cut:
                if i==0:
                    ms.append(np.asarray(Gamma))
                else:
                    ms.append(np.asarray(Gamma)*np.asarray(self.Lambdas[i-1]).reshape(shape))
            else:
                if i>0 and i==cut:
                    Lambda=np.asarray(self.Lambdas[i-1])
                if i<len(self.Lambdas):
                    ms.append(np.asarray(Gamma)*np.asarray(self.Lambdas[i]).reshape(shape))
                else:
                    ms.append(np.asarray(Gamma))
        return MPS(ms,labels,Lambda,cut)
