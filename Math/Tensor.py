'''
Tensor and tensor operations, including:
1) classes: Prime, Tensor
2) functions: prime, contract
'''

from numpy import *
from opt_einsum import contract as einsum
from collections import namedtuple,Counter
from copy import deepcopy
from HamiltonianPy.Math.linalg import truncated_svd

__all__=['PRIME','prime','Tensor','contract']

class PRIME(tuple):
    '''
    The prime of an arbitary object.
    '''

    def __init__(self,para):
        '''
        Constructor.
        Parameters:
            para: any arbitary object.
                The object that wanted primed.
        '''
        tuple.__init__(self,para)

def prime(obj):
    '''
    The prime of an object.
    '''
    if isinstance(obj,PRIME):
        return obj[0]
    else:
        return PRIME((obj,))

class Tensor(ndarray):
    '''
    Tensor class with labeled axes. 
    Attributes:
        labels: list of hashable objects, e.g. string, tuple, etc.
            The labels of the axes.
    '''

    def __new__(cls,array,labels,*args,**kargs):
        '''
        Initialize an instance through the explicit construction, i.e. constructor.
        Parameters:
            array: ndarray like
                The data of the Tensor.
            labels: list of hashable objects
                The labels of the Tensor.
        '''
        temp=asarray(array,*args,**kargs)
        if len(labels)!=temp.ndim:
            raise ValueError("Tensor construction error: the number of labels(%s) and the dimension(%s) of tensors are not equal."%(len(labels),temp.ndim))
        result=temp.view(cls)
        result.labels=labels
        return result

    def __array_finalize__(self,obj):
        '''
        Initialize an instance through both explicit and implicit constructions, i.e. construtor, view and slice.
        '''
        if obj is None:
            return
        else:
            self.labels=getattr(obj,'labels',None)

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        if self.ndim>1:
            return "Tensor(labels=%s,\ndata=%s)"%(self.labels,super(Tensor,self).__str__())
        else:
            return "Tensor(labels=%s, data=%s)"%(self.labels,super(Tensor,self).__str__())

    def copy(self,copy_data=False):
        '''
        Make a copy of a tensor.
        Parameters:
            copy_data: logical, optional
                When True, both the data and labels of the tensor will be copied;
                When False, only the labels of the tensor will be copyied.
        Returns: Tensor
            The copy of the tensor.
        '''
        if copy_data:
            return deepcopy(self)
        else:
            return Tensor(asarray(self),labels=deepcopy(self.labels))

    def label(self,axis):
        '''
        Return the corresponding label of an axis.
        Parameters:
            axis: integer
                The axis whose corresponding label is inquired.
            Returns: any hashable object
                The corresponding label.
        '''
        return self.labels[axis]

    def axis(self,label):
        '''
        Return the corresponding axis of a label.
        Parameters:
            label: any hashable object
                The label whose corresponding axis is inquired.
            Returns: integer
                The corresponding axis.
        '''
        return self.labels.index(label)

    def relabel(self,news,olds=None):
        '''
        Change the labels of the tensor.
        Parameters:
            news: list of hashable objects
                The new labels of the tensor's axes.
            olds: list of hashable objects, optional
                The old labels of the tensor's axes.
        '''
        if olds is None:
            if self.ndim!=len(news):
                raise ValueError("Tensor relabel error: the number of lables and the dimension of tensors should be equal.")
            self.labels=news
        else:
            if len(news)!=len(olds):
                raise ValueError("Tensor relabel error: the number of new labels(%s) and old labels(%s) are not equal."%(len(news),len(olds)))
            for old,new in zip(olds,news):
                self.labels[self.axis(old)]=new

    def transpose(self,labels=None,axes=None):
        '''
        Change the order of the tensor's axes and return the new tensor.
        Parameters:
            labels: list of hashable objects, optional
                The permutation of the original labels.
            axes: list of integers, optional
                The permutation of the original axes.
            NOTE: labels and axes should not be assigned at the same time. 
                  But if this does happen, axes will be omitted.
        Returns: Tensor
            The new tensor with the reordered axes.
        '''
        if labels is not None:
            if len(labels)!=len(self.labels):
                raise ValueError("Tensor transpose error: the number of labels doesn't match the tensor.")
            return Tensor(self.view(ndarray).transpose([self.axis(label) for label in labels]),labels=labels)
        elif axes is not None:
            if len(axes)!=len(self.labels):
                raise ValueError("Tensor transpose error: the number of axes doesn't match the tensor.")
            return Tensor(self.view(ndarray).transpose(axes),labels=[self.label(axis) for axis in axes])
        else:
            raise ValueError("Tensor transpose error: labels and axes cannot be None simultaneously.")

    def take(self,indices,label=None,axis=None):
        '''
        Take elements from a tensor along an axis.
        Parameters:
            indices: integer / list of integers
                The indices of the values to extract.
            label: any hashable object, optional
                The label of the axis along which to select values.
            axis: integer, optional
                The axis along which to select values.
            NOTE: label and axis should not be assigned at the same time. 
                  But if this does happen, axis will be omitted.
        Returns: Tensor
        '''
        if label is not None:
            if isinstance(indices,int) or isinstance(indices,long) or len(indices)==1:
                labels=[i for i in self.labels if i!=label]
            else:
                labels=deepcopy(self.labels)
            return Tensor(self.view(ndarray).take(indices,axis=self.axis(label)),labels=labels)
        elif axis is not None:
            if isinstance(indices,int) or len(indices)==1:
                labels=[label for i,label in enumerate(self.labels) if i!=axis]
            else:
                labels=deepcopy(self.labels)
            return Tensor(self.view(ndarray).take(indices,axis=axis),labels=labels)
        else:
            raise ValueError("Tensor take error: label and axis cannot be None simultaneously.")

    def reshape(self,shape,labels):
        '''
        Reshape and relabel of the Tensor.
        Parameters:
            shape: tuple
                The new shape of the tensor.
            labels: list of hashable objects
                The new labels of the tensor.
        Returns: Tensor
            The new tensor.
            The data of the new tensor is just a view of the original one.
        '''
        return Tensor(asarray(self).reshape(shape),labels=labels)

    def svd(self,labels1,new,labels2,nmax=None,tol=None,print_truncation_err=False):
        '''
        Perform the svd.
        Parameters:
            labels1,labels2: list of any hashable object
                The axis labels of the two groups.
            new: any hashable object
                The new axis label after the svd.
            nmax: integer, optional
                The maximum number of singular values to be kept. 
                If it is None, it takes no effect.
            tol: float64, optional
                The truncation tolerance.
                If it is None, it taks no effect.
            print_truncation_err: logical, optional
                If it is True, the truncation err will be printed.
        Returns:
            U,S,V: Tensor
        '''
        axes1,axes2,shape1,shape2=[],[],(),()
        for label in labels1:
            axis=self.axis(label)
            axes1.append(axis)
            shape1+=(self.shape[axis],)
        for label in labels2:
            axis=self.axis(label)
            axes2.append(axis)
            shape2+=(self.shape[axis],)
        if [i for i in xrange(len(self.labels)) if (i not in axes1) and (i not in axes2)]:
            raise ValueError('Tensor svd error: all axis should be divided into two group to perform the svd.')
        m=asarray(self).transpose(axes1+axes2).reshape((product(shape1),)+(product(shape2),))
        u,s,v=truncated_svd(m,nmax=nmax,tol=tol,print_truncation_err=print_truncation_err,full_matrices=False)
        U=Tensor(u.reshape(shape1+(-1,)),labels=labels1+[new])
        S=Tensor(s,labels=[new])
        V=Tensor(v.reshape((-1,)+shape2),labels=[new]+labels2)
        return U,S,V

def contract(*tensors,**karg):
    '''
    Contract a collection of tensors.
    Parametes:
        tensors: list of Tensor
            The tensors to be contracted.
        karg['sequence']: list of tuple, 'sequential','reversed'
            The contraction sequence of the tensors.
        karg['mask']: list of hashable objects
            The labels that are repeated but not summed over.
    Returns: Tensor
        The contracted tensor.
    '''
    if len(tensors)==0:
        raise ValueError("Tensor contract error: there are no tensors to contract.")
    sequence=karg.pop('sequence',[])
    if len(sequence)==0:
        return _contract_(*tensors,**karg)
    else:
        if sequence=='sequential':
            sequence=[(i,) for i in xrange(len(tensors))]
        elif sequence=='reversed':
            sequence=[(i,) for i in xrange(len(tensors)-1,-1,-1)]
        for i,seq in enumerate(sequence):
            temp=[tensors[i] for i in seq]
            if i==0:
                result=_contract_(*temp,**karg)
            else:
                result=_contract_(result,*temp,**karg)
        return result

def _contract_(*tensors,**karg):
    '''
    Contract a small collection of tensors.
    '''
    mask={key:True for key in karg.get('mask',[])}
    lists=[tensor.labels for tensor in tensors]
    alls=[label for labels in lists for label in labels]
    counts=Counter(alls)
    table={key:i for i,key in enumerate(counts)}
    subscripts=[''.join(chr(table[label]+97) for label in labels) for labels in lists]
    contracted_labels=[label for label in alls if (counts[label]==1 or mask.pop(label,False))]
    contracted_subscript=''.join(chr(table[label]+97) for label in contracted_labels)
    return Tensor(einsum('%s->%s'%(','.join(subscripts),contracted_subscript),*tensors),labels=contracted_labels)
