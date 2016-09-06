'''
Matrix product state, including:
1) constants: LLINK, SITE, RLINK
2) classes: MPSBase, Vidal, MPS
3) functions: from_state
'''

__all__=['LLINK','SITE','RLINK','MPSBase','Vidal','MPS','from_state']

from numpy import *
from HamiltonianPy.Math.Tensor import *
from copy import deepcopy

LLINK,SITE,RLINK=0,1,2

def from_state(state,form='L'):
    '''
    Convert a normal state representation to the Vidal/Mixed canonical MPS representation.
    '''
    pass

class MPSBase(object):
    '''
    The base class for matrix product states.
    Attributes:
        order: list of int
            The order of the three axis of each matrix.
        err: float
            The error tolerance.
    '''
    order=[LLINK,SITE,RLINK]
    err=10**-10

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

    def state(self,form='flat'):
        '''
        Convert to the normal representation.
        Parameters:
            form: 'flat','tensor','component'
                The form of the final state.
            returns: it depends on form.
                1) 'flat': 1d ndarray
                    The flattened 1D array representation of the state.
                2) 'tensor': Tensor
                    The tensor representation of the state.
                3) 'component': list of 2-tuple
                        tuple[0]: tuple
                            The index of a non-zero value of the tensor representation.
                        tuple[1]: number
                            The corresponding non-zero value.
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
                The labels of the axis of the matrices.
                Its length should be equal to that of ms.
                For each label in labels, 
                    label[0]: any hashable object
                        The left link label of the matrix.
                    label[1]: any hashable object
                        The site label of the matrix.
                    label[2]: any hashable object
                        The right link label of the matrix.
            Lambda: 1d ndarray, optional
                The Lambda matrix on the connecting link.
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
        if cut<0 or cut>len(ms):
            raise ValueError('MPS construction error: the cut(%s) is out of range [0,%s].'%(cut,len(ms)))
        self.Lambda=None if Lambda is None else Tensor(Lambda,labels=[deepcopy(labels[cut][2])])
        self.cut=cut

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        result=[]
        for i,M in enumerate(self.ms):
            if self.Lambda is not None and i==self.cut:
                result.append(str(self.Lambda))
            result.append(str(M))
        return '\n'.join(result)

    @property
    def nsite(self):
        '''
        The number of total sites.
        '''
        return len(self.ms)

    def state(self,form='flat'):
        '''
        Convert to the normal representation.
        Parameters:
            form: 'flat','tensor','component'
                The form of the final state.
            returns: it depends on form.
                1) 'flat': 1d ndarray
                    The flattened 1D array representation of the state.
                2) 'tensor': Tensor
                    The tensor representation of the state.
                3) 'component': list of 2-tuple
                        tuple[0]: tuple
                            The index of a non-zero value of the tensor representation.
                        tuple[1]: number
                            The corresponding non-zero value.
        '''
        for i,m in enumerate(self.ms):
            if i==0:
                result=m
            else:
                if i==self.cut:
                    result=contract(result,self.Lambda,m)
                else:
                    result=contract(result,m)
        if form in ('flat'):
            return asarray(result).ravel()
        elif form in ('tensor'):
            return result
        else:
            return result.components(zero=self.err)

    @property
    def norm(self):
        '''
        The norm of the matrix product state.
        '''
        for i,M in enumerate(self.ms):
            L,S,R=M.labels[self.L],M.labels[self.S],M.labels[self.R]
            if i==0:
                temp=M
            else:
                if i==self.cut:
                    temp=contract(v*asarray(s)[:,newaxis],self.Lambda,M)
                else:
                    temp=contract(v*asarray(s)[:,newaxis],M)
                temp.relabel(news=[L],olds=['_'+L])
            u,s,v=temp.svd([L,S],'_'+str(R),[R])
        return (asarray(v)*asarray(s)[:,newaxis])[0,0]

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

    def is_canonical(self):
        '''
        Judge whether the MPS is in the cannonical form.
        '''
        result=[]
        for i,M in enumerate(self.ms):
            temp=[asarray(M.take(indices=i,axis=M.labels[self.S])) for i in xrange(M.shape[self.S])]
            buff=None
            for matrix in temp:
                if buff is None:
                    buff=matrix.T.conjugate().dot(matrix) if i<self.cut else matrix.dot(matrix.T.conjugate())
                else:
                    buff+=matrix.T.conjugate().dot(matrix) if i<self.cut else matrix.dot(matrix.T.conjugate())
            result.append(all(abs(buff-identity(M.shape[self.R if i<self.cut else self.L]))<self.err))
        return result

    def to_left(self,normalization=False):
        '''
        Convert to the left canonical MPS representation.
        '''
        ms,labels,Lambda,cut=[],[],None,self.nsite
        for i,M in enumerate(self.ms):
            L,S,R=M.labels[self.L],M.labels[self.S],M.labels[self.R]
            if i==0:
                temp=M
            else:
                if i==self.cut:
                    temp=contract(v*asarray(s)[:,newaxis],self.Lambda,M)
                else:
                    temp=contract(v*asarray(s)[:,newaxis],M)
                temp.relabel(news=[L],olds=['_'+L])
            labels.append((L,S,R))
            u,s,v=temp.svd([L,S],'_'+str(R),[R])
            if normalization:
                ms.append(asarray(u))
            else:
                if i<self.nsite-1:
                    ms.append(asarray(u))
                else:
                    ms.append(asarray(temp))
        return MPS(ms,labels,Lambda,cut)

    def to_right(self,normalization=False):
        '''
        Convert to the right canonical MPS representation.
        '''
        ms,labels,Lambda,cut=[],[],None,0
        for i,M in enumerate(reversed(self.ms)):
            L,S,R=M.labels[self.L],M.labels[self.S],M.labels[self.R]
            if i==0:
                temp=M
            else:
                if self.cut is not None and i==self.nsite-self.cut:
                    temp=contract(M,self.Lambda,u*asarray(s)[newaxis,:])
                else:
                    temp=contract(M,u*asarray(s)[newaxis,:])
                temp.relabel(news=[R],olds=['_'+str(R)])
            labels.append((L,S,R))
            u,s,v=temp.svd([L],'_'+str(L),[S,R])
            if normalization:
                ms.append(asarray(v))
            else:
                if i<self.nsite-1:
                    ms.append(asarray(v))
                else:
                    ms.append(asarray(temp))
        return MPS(list(reversed(ms)),list(reversed(labels)),Lambda,cut)

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
                    label[0]: any hashable object
                        The left link label of the matrix.
                    label[1]: any hashable object
                        The site label of the matrix.
                    label[2]: any hashable object
                        The right link label of the matrix.
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
