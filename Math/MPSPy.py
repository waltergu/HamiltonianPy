'''
Matrix product state, including:
1) constants: LLINK, SITE, RLINK
2) classes: GMPS, VMPS, MMPS
'''

__all__=['LLINK','SITE','RLINK','GMPS','VMPS','MMPS']

from numpy import *
from TensorPy import *
from copy import deepcopy

LLINK,SITE,RLINK=0,1,2

class GMPS(object):
    '''
    The general matrix product state.
    Attributes:
        ms: list of Tensor
            The matrices.
        order: list of int
            The order of the three axis of each matrix.
    '''

    def __init__(self,ms,labels,order=[LLINK,SITE,RLINK]):
        '''
        Constructor.
        Parameters:
            ms: list of 3d ndarray
                The matrices.
            labels: list of 2 tuples
                The labels of the axis of the matrices.
                Its length should be equal to that of ms.
                For each label in labels, 
                    label[0]: any hashable object
                        The point label of the matrix
                    label[1]: any hashable object
                        The bond label of the matrix
            order: list of int, optional
                The order of the three axis of each matrix.
        '''
        if len(ms)!=len(labels):
            raise ValueError('GMPS construction error: the length of matrices(%s) is not equal to that of the labels(%s).'%(len(ms),len(labels)))
        self.order=order
        self.ms=[]
        temp=[None]*3
        for i,(m,label) in enumerate(zip(ms,labels)):
            if m.ndim!=3:
                raise ValueError('GMPS construction error: all input matrices should be 3 dimensional.')
            S,R=label
            if i==0:
                L=labels[len(labels)-1][1]
            else:
                L=temp[self.order.index(RLINK)]
            temp[self.order.index(LLINK)]=L
            temp[self.order.index(SITE)]=S
            temp[self.order.index(RLINK)]=R
            self.ms.append(Tensor(m,labels=deepcopy(temp)))

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return '\n'.join(['%s'%m for m in self.ms])

    @property
    def state(self):
        '''
        Convert to the normal state representation.
        '''
        result=None
        for m in self.ms:
            if result is None:
                result=m
            else:
                result=contract(result,m)
        return asarray(result).ravel()

    def to_vmps(self):
        '''
        Convert to VMPS.
        '''
        pass

    def to_mmps(self):
        '''
        Convert to MMPS.
        '''
        pass

class VMPS(object):
    '''
    The Vidal canonical matrix product state.
    Attributes:
        Gammas: list of Tensor
            The Gamma matrices on the site.
        Lambdas: list of Tensor
            The Lambda matrices on the bond.
        order: list of int
            The order of the three axis of each Gamma matrix.
    '''

    def __init__(self,Gammas,Lambdas,labels,order=[LLINK,SITE,RLINK]):
        '''
        Constructor.
        Parameters:
            Gammas: list of 3d ndarray
                The Gamma matrices on the site.
            Lamdas: list of 1d ndarray
                The Lambda matrices on the bond.
            labels: list of 2 tuples
                The labels of the axis of the Gamma matrices.
                Its length should be equal to that of Gammas.
                For each label in labels, 
                    label[0]: any hashable object
                        The point label of the Gamma matrix
                    label[1]: any hashable object
                        The bond label of the Gamma matrix
            order: list of int, optional
                The order of the three axis of each matrix.
        '''
        if len(gs)!=len(ls)+1:
            raise ValueError('VMPS construction error: there should be one more Gamma matrices(%s) than the Lambda matrices(%s).'%(len(gs),len(ls)))
        if len(gs)!=len(labels):
            raise ValueError('VMPS construction error: the length of Gamma matrices(%s) is not equal to that of the labels(%s).'%(len(ms),len(labels)))
        self.order=order
        self.Gammas=[]
        self.Lambdas=[]
        temp,buff=[None]*3,[]
        for i,(Gamma,label) in enumerate(Gammas,labels):
            if Gamma.ndim!=3:
                raise ValueError('VMPS construction error: all Gamma matrices should be 3 dimensional.')
            S,R=label
            if i<len(Gammas)-1:
                buff.append(R)
            if i==0:
                L=labels[len(labels)-1][1]
            else:
                L=temp[self.order.index(RLINK)]
            temp[self.order.index(LLINK)]=L
            temp[self.order.index(SITE)]=S
            temp[self.order.index(RLINK)]=R
            self.Gammas.append(Tensor(Gamma,labels=deepcopy(temp)))
        for Lambda,label in zip(Lambdas,buff):
            if Lambda.ndim!=1:
                raise ValueError("VMPS construction error: all Lambda matrices should be 1 dimensional.")
            self.Lambdas.append(Tensor(Lambda,labels=[label]))

    @property
    def state(self):
        '''
        Convert to the normal state representation.
        '''
        pass

    def to_mmps(self):
        '''
        Convert to MMPS.
        '''
        pass

class MMPS(object):
    '''
    The mixed canonical matrix product state.
    Note the left-canonical MPS and right-canonical MPS are considered as special cases of this form.
    '''
    def __init__(self):
        pass
