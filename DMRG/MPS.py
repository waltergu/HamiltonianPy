'''
Matrix product state, including:
1) constants: LLINK, SITE, RLINK
2) classes: MPSBase, MPS, Vidal
'''

__all__=['LLINK','SITE','RLINK','MPSBase','Vidal','MPS']

from numpy import *
from HamiltonianPy.Math.Tensor import *
from copy import deepcopy

LLINK,SITE,RLINK=0,1,2

class MPSBase(object):
    '''
    The base class for matrix product states.
    Attributes:
        order: list of int
            The order of the three axis of each matrix.
        tol: float
            The error tolerance.
    '''
    order=[LLINK,SITE,RLINK]
    nmax=200
    tol=10**-14

    @property
    def nsite(self):
        '''
        The number of total sites.
        '''
        raise NotImplementedError()

    @property
    def L(self):
        '''
        The axes of LLINK.
        '''
        return self.order.index(LLINK)

    @property
    def S(self):
        '''
        The axes of SITE.
        '''
        return self.order.index(SITE)

    @property
    def R(self):
        '''
        The axes of RLINK.
        '''
        return self.order.index(RLINK)

    @property
    def state(self):
        '''
        Convert to the normal representation.
        '''
        raise NotImplementedError()

    @classmethod
    def from_state(cls,state,*arg,**karg):
        '''
        Convert the normal representation of a state to the matrix product representation.
        Parameters:
            cls: subclass of MPSBase
                The subclass of MPSBase.
            state: 1d ndarray
                The normal representation of a state.
        Returns: MPSBase
            The corresponding matrix product state.
        '''
        raise NotImplementedError()

class MPS(MPSBase):
    '''
    The general matrix product state.
    Attributes:
        ms: list of Tensor
            The matrices.
        Lambda: Tensor
            The Lambda matrix (singular values) on the connecting link.
        cut: integer
            The index of the connecting link.
    Note the left-canonical MPS, right-canonical MPS and mixed-canonical MPS are considered as special cases of this form.
    '''

    def __init__(self,ms,labels,Lambda=None,cut=0):
        '''
        Constructor.
        Parameters:
            ms: list of 3d ndarray
                The matrices.
            labels: list of 3 tuples
                The labels of the axis of the matrices, thus its length should be equal to that of ms.
                For each label in labels, 
                    label[0],label[1],label[2]: any hashable object
                        The left link / site / right link label of the matrix.
            Lambda: 1d ndarray, optional
                The Lambda matrix (singular values) on the connecting link.
            cut: integer, optional
                The index of the connecting link.
        '''
        if len(ms)!=len(labels):
            raise ValueError('MPS construction error: the number of matrices(%s) is not equal to that of the labels(%s).'%(len(ms),len(labels)))
        self.ms=[]
        temp=[None]*3
        for i,(m,label) in enumerate(zip(ms,labels)):
            if m.ndim!=3:
                raise ValueError('MPS construction error: all input matrices should be 3 dimensional.')
            L,S,R=label
            temp[self.L]=L
            temp[self.S]=S
            temp[self.R]=R
            self.ms.append(Tensor(m,labels=deepcopy(temp)))
        if cut>0 and cut<len(ms):
            if Lambda is None:
                raise ValueError("MPS construction error: cut is %s and Lambda is not assigned."%(cut))
            else:
                self.Lambda=Tensor(Lambda,labels=[deepcopy(labels[cut-1][2])])
            self.cut=cut
        elif cut==0 or cut==len(ms):
            labels=[] if Lambda is None else [deepcopy(labels[cut][0] if cut==0 else labels[cut-1][2])]
            self.Lambda=Tensor(1 if Lambda is None else Lambda,labels=labels)
            self.cut=cut
        else:
            raise ValueError('MPS construction error: the cut(%s) is out of range [0,%s].'%(cut,len(ms)))

    @classmethod
    def compose(cls,As,Lambda,Bs,labels=None):
        '''
        Construct an MPS from As, Lambda and Bs.
        Parameters:
            As,Bs: list of 3d Tensor or list of 3d ndarray
                The A/B matrices of the MPS.
            Lambda: 1d Tensor or 1d ndarray
                The Lambda matrix (singular values) on the connecting link.
            labels: list of 3-tuple, optional.
                It is used only when As, Lambda and Bs are ndarrays.
                For details, see MPS.__init__.
        Returns: MPS
            The constructed MPS.
        '''
        if all([isinstance(A,Tensor) for A in As]) and all([isinstance(B,Tensor) for B in Bs]) and isinstance(Lambda,Tensor):
            result=cls.__new__(cls)
            result.ms=As+Bs
            result.Lambda=Lambda
            result.cut=len(As)
            return result
        elif labels is not None:
            return cls(ms=As+Bs,labels=labels,Lambda=Lambda,cut=len(As))

    @property
    def As(self):
        '''
        The A matrices.
        '''
        return self.ms[0:self.cut]

    @property
    def Bs(self):
        '''
        The B matrices.
        '''
        return self.ms[self.cut:self.nsite]

    @property
    def decompose(self):
        '''
        Decompose the MPS into A matrices, Lambda matrix and B matrices.
        '''
        return self.As,self.Lambda,self.Bs

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        result=[]
        for i,M in enumerate(self.ms):
            if i==self.cut:
                result.append(str(self.Lambda))
            result.append(str(M))
        return '\n'.join(result)

    @property
    def nsite(self):
        '''
        The number of total sites.
        '''
        return len(self.ms)

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
        if self.cut in (0,self.nsite):
            result=contract(*self.ms,sequence='sequential')
        else:
            A,B=contract(*self.As,sequence='sequential'),contract(*self.Bs,sequence='sequential')
            result=contract(A,self.Lambda,B)
        legs=set(result.labels)-set(m.labels[self.S] for m in self.ms)
        if len(legs)==0:
            return asarray(result).ravel()
        elif len(legs)==1:
            buff=1
            for label,n in zip(result.labels,result.shape):
                if label in legs:
                    temp=ndim
                else:
                    buff*=ndim
            return asarray(result).reshape((buff,temp))
        else:
            raise ValueError('MPS state error: %s link labels%s are left.'%(len(legs),tuple(legs)))

    @property
    def norm(self):
        '''
        The norm of the matrix product state.
        '''
        temp=deepcopy(self)
        temp._reset_(reset=0)
        temp>>=temp.nsite
        return asarray(temp.Lambda)

    def _reset_(self,merge='A',reset=0):
        '''
        Merge the Lamdbda matrix on the link to its neighbouring A matrix or B matrix and reset the cut to 0 or to self.nsite.
        Parameters:
            merge: 'A' or 'B', optional
                When 'A', self.Lambda will be merged into its neighbouring A matrix;
                When 'B', self.Lambda will be merged into its neighbouring B matrix.
            reset: 0 or self.nsite, optional
                Reset self.cut to this integer.
        '''
        if self.cut==self.nsite or (merge=='A' and self.cut!=0):
            self.ms[self.cut-1]=contract(self.ms[self.cut-1],self.Lambda,mask=self.Lambda.labels)
        else:
            self.ms[self.cut]=contract(self.Lambda,self.ms[self.cut],mask=self.Lambda.labels)
        self.cut=0 if reset==0 else self.nsite
        self.Lambda=Tensor(1.0,labels=[])

    def _set_B_and_lmove_(self,M,nmax=MPSBase.nmax,tol=MPSBase.tol,print_truncation_err=True):
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
        u,s,v=M.svd([L],'_'+str(L),[S,R],nmax=nmax,tol=tol,print_truncation_err=print_truncation_err)
        v.relabel(news=[L],olds=['_'+str(L)])
        self.ms[self.cut-1]=v
        if self.cut==1:
            if len(s)>1:
                raise ValueError('MPS _set_B_and_lmove_ error(not supported operation): the MPS is a mixed state and is to move to the end.')
            self.Lambda=contract(u,s)
        else:
            s.relabel(news=[L],olds=['_'+str(L)])
            self.Lambda=s
            self.ms[self.cut-2]=contract(self.ms[self.cut-2],u)
            self.ms[self.cut-2].relabel(news=[L],olds=['_'+str(L)])
        self.cut=self.cut-1

    def _set_A_and_rmove_(self,M,nmax=MPSBase.nmax,tol=MPSBase.tol,print_truncation_err=True):
        '''
        Set the A matrix at self.cut and move rightward.
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
        if self.cut==self.nsite:
            raise ValueError('MPS _set_A_and_rmove_ error: the cut is already maximum.')
        L,S,R=M.labels[self.L],M.labels[self.S],M.labels[self.R]
        u,s,v=M.svd([L,S],'_'+str(R),[R],nmax=nmax,tol=tol,print_truncation_err=print_truncation_err)
        u.relabel(news=[R],olds=['_'+str(R)])
        self.ms[self.cut]=u
        if self.cut==self.nsite-1:
            if len(s)>1:
                raise ValueError('MPS _set_A_and_rmove_ error(not supported operation): the MPS is a mixed state and is to move to the end.')
            self.Lambda=contract(s,v)
        else:
            s.relabel(news=[R],olds=['_'+str(R)])
            self.Lambda=s
            self.ms[self.cut+1]=contract(v,self.ms[self.cut+1])
            self.ms[self.cut+1].relabel(news=[R],olds=['_'+str(R)])
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
        nmax,tol=self.nmax,self.tol
        if isinstance(other,tuple):
            k,nmax,tol=other
        else:
            k=other
        for i in xrange(k):
            #M=self.ms[self.cut-1]*(asarray(self.Lambda)[newaxis,:] if self.Lambda.ndim>0 else asarray(self.Lambda))
            M=contract(self.ms[self.cut-1],self.Lambda,mask=self.Lambda.labels)
            self._set_B_and_lmove_(M,nmax,tol)
        return self

    def __lshift__(self,other):
        '''
        Operator "<<".
        Parameters:
            other: integer or 3-tuple.
                Please see MPS.__ilshift__ for details.
        '''
        return deepcopy(self).__ilshift__(other)

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
        nmax,tol=self.nmax,self.tol
        if isinstance(other,tuple):
            k,nmax,tol=other
        else:
            k=other
        for i in xrange(k):
            #M=self.ms[self.cut]*(asarray(self.Lambda)[:,newaxis] if self.Lambda.ndim>0 else asarray(self.Lambda))
            M=contract(self.Lambda,self.ms[self.cut],mask=self.Lambda.labels)
            self._set_A_and_rmove_(M,nmax,tol)
        return self

    def __rshift__(self,other):
        '''
        Operator ">>".
        Parameters:
            other: integer or 3-tuple.
                Please see MPS.__irshift__ for details.
        '''
        return deepcopy(self).__irshift__(other)

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
        Judge whether the MPS is in the canonical form.
        '''
        result=[]
        for i,M in enumerate(self.ms):
            temp=[asarray(M.take(indices=j,axis=self.S)) for j in xrange(M.shape[self.S])]
            buff=None
            for matrix in temp:
                if buff is None:
                    buff=matrix.T.conjugate().dot(matrix) if i<self.cut else matrix.dot(matrix.T.conjugate())
                else:
                    buff+=matrix.T.conjugate().dot(matrix) if i<self.cut else matrix.dot(matrix.T.conjugate())
            result.append(all(abs(buff-identity(M.shape[self.R if i<self.cut else self.L]))<self.tol))
        return result

    def prime(self,labels=None,copy_data=False):
        '''
        '''
        pass

    def to_vidal(self):
        '''
        Convert to the Vidal MPS representation.
        '''
        Gammas,Lambdas,labels=[],[],[]
        for i,M in enumerate(self.ms):
            L,S,R=M.labels[self.L],M.labels[self.S],M.labels[self.R]
            if i==0:
                temp=M
            else:
                if i==self.cut:
                    temp=contract(v*asarray(s)[:,newaxis],self.Lambda,M)
                else:
                    temp=contract(v*asarray(old)[:,newaxis],M)
                temp.relabel(news=[L],olds=['_'+L])
            u,new,v=temp.svd([L,S],'_'+str(R),[R])
            labels.append((L,S,R))
            if i==0:
                Gammas.append(asarray(u))
            else:
                Gammas.append(asarray(u)/asarray(old)[:,newaxis])
            old=new
            if i<len(self.ms)-1:
                Lambdas.append(asarray(new))
            else:
                norm=abs((asarray(v)*asarray(new)[:,newaxis])[0,0])
                if abs(norm-1.0)>self.err:
                    raise ValueError('MPS to_vidal error: the norm(%s) of original MPS does not equal to 1.'%norm)
        return Vidal(Gammas,Lambdas,labels)

class Vidal(MPSBase):
    '''
    The Vidal canonical matrix product state.
    Attributes:
        Gammas: list of Tensor
            The Gamma matrices on the site.
        Lambdas: list of Tensor
            The Lambda matrices (singular values) on the link.
    '''

    def __init__(self,Gammas,Lambdas,labels):
        '''
        Constructor.
        Parameters:
            Gammas: list of 3d ndarray
                The Gamma matrices on the site.
            Lamdas: list of 1d ndarray
                The Lambda matrices (singular values) on the link.
            labels: list of 3 tuples
                The labels of the axis of the Gamma matrices.
                Its length should be equal to that of Gammas.
                For each label in labels, 
                    label[0],label[1],label[2]: any hashable object
                        The left link / site / right link label of the matrix.
        '''
        if len(Gammas)!=len(Lambdas)+1:
            raise ValueError('Vidal construction error: there should be one more Gamma matrices(%s) than the Lambda matrices(%s).'%(len(Gammas),len(Lambdas)))
        if len(Gammas)!=len(labels):
            raise ValueError('Vidal construction error: the number of Gamma matrices(%s) is not equal to that of the labels(%s).'%(len(Gammas),len(labels)))
        self.Gammas=[]
        self.Lambdas=[]
        temp,buff=[None]*3,[]
        for i,(Gamma,label) in enumerate(zip(Gammas,labels)):
            if Gamma.ndim!=3:
                raise ValueError('Vidal construction error: all Gamma matrices should be 3 dimensional.')
            L,S,R=label
            if i<len(Gammas)-1:
                buff.append(R)
            temp[self.L]=L
            temp[self.S]=S
            temp[self.R]=R
            self.Gammas.append(Tensor(Gamma,labels=deepcopy(temp)))
        for Lambda,label in zip(Lambdas,buff):
            if Lambda.ndim!=1:
                raise ValueError("Vidal construction error: all Lambda matrices should be 1 dimensional.")
            self.Lambdas.append(Tensor(Lambda,labels=[label]))

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

    def state(self):
        '''
        Convert to the normal representation.
        '''
        result=None
        for i,Gamma in enumerate(self.Gammas):
            if result is None:
                result=Gamma
            else:
                result=contract(result,self.Lambdas[i-1],Gamma)
        return asarray(result).ravel()

    def to_mixed(self,cut):
        '''
        Convert to the mixed MPS representation.
        '''
        ms,labels,Lambda=[],[],None
        shape=[1]*3
        shape[self.S]=-1
        for i,Gamma in enumerate(self.Gammas):
            L,S,R=Gamma.labels[self.L],Gamma.labels[self.S],Gamma.labels[self.R]
            labels.append((L,S,R))
            if i<cut:
                if i==0:
                    ms.append(asarray(Gamma))
                else:
                    ms.append(asarray(Gamma)*asarray(self.Lambdas[i-1]).reshape(shape))
            else:
                if i>0 and i==cut:
                    Lambda=asarray(self.Lambdas[i-1])
                if i<len(self.Lambdas):
                    ms.append(asarray(Gamma)*asarray(self.Lambdas[i]).reshape(shape))
                else:
                    ms.append(asarray(Gamma))
        return MPS(ms,labels,Lambda,cut)
