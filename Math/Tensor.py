'''
Tensor and tensor operations, including:
1) classes: Tensor
2) functions: concode, contract
'''

from numpy import *
from numpy.linalg import svd
from opt_einsum import contract as einsum
from copy import deepcopy,copy

__all__=['concode','contract','Tensor']

def concode(*labels_list):
    '''
    Compute the subscripts for label contraction and the contracted labels.
    Parameters:
        labels_list: list of labels
    Returns: tuple
        The subscripts for label contraction and the contracted labels.
    '''
    labels_all=concatenate(labels_list)
    labels_unique,counts=unique(labels_all,return_counts=True)
    table={key:value for value,key in enumerate(labels_unique)}
    subscripts=[]
    for labels in labels_list:
        subscripts.append(''.join([chr(table[label]+97) for label in labels]))
    contracted_labels=[label for label in labels_all if counts[table[label]]==1]
    contracted_subscript=''.join(chr(table[label]+97) for label in contracted_labels)
    return '%s->%s'%(','.join(subscripts),contracted_subscript),contracted_labels

def contract(*tensors):
    '''
    Contract a collection of tensors.
    Parametes:
        tensors: list of Tensor.
    Returns: Tensor
        The contracted tensor.
    '''
    if len(tensors)==0:
        raise ValueError("Tensor contract error: there are no tensors to contract.")
    tag,labels=concode(*[tensor.labels for tensor in tensors])
    return Tensor(einsum(tag,*tensors),labels=labels)

class Tensor(ndarray):
    '''
    Tensor class with labeled axes. 
    Attributes:
        labels: list of hashable objects, e.g. string, tuple, etc.
            The labels of the axes.
    Usage:
        Tensor(shape,labels,**kargs):
            Create a Tensor with random data with specified shape.
        Tensor(array,labels,**kargs):
            Create a Tensor converted from array.
    '''

    def __new__(cls,para,labels,*args,**kargs):
        '''
        Initialize an instance through the explicit construction, i.e. constructor.
        '''
        if isinstance(para,tuple):
            if len(labels)!=len(para):
                raise ValueError("Tensor construction error: the number of labels(%s) and the dimension(%s) of tensors are not equal."%(len(labels),para.ndim))
            result=ndarray.__new__(cls,shape=para,*args,**kargs)
        elif isinstance(para,cls):
            if len(labels)!=para.ndim:
                raise ValueError("Tensor construction error: the number of labels(%s) and the dimension(%s) of tensors are not equal."%(len(labels),para.ndim))
            result=para
        else:
            if len(labels)!=asarray(para).ndim:
                raise ValueError("Tensor construction error: the number of labels(%s) and the dimension(%s) of tensors are not equal."%(len(labels),para.ndim))
            result=asarray(para,*args,**kargs).view(cls)
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

    def label(self,axis):
        '''
        Return the corresponding label of an axis.
        Parameters:
            axis: int or any hashable object
                The axis whose corresponding label is inquired.
            Returns: any hashable object
                The corresponding label.
        '''
        if axis.__class__.__name__==self.labels[0].__class__.__name__:
            if axis in self.labels:
                return axis
            else:
                raise ValueError("Tensor label error: the axis(%s) doesn't correspond to any exsiting label."%axis)
        elif isinstance(int):
            return self.labels[axis]
        else:
            raise ValueError("Tensor label error: the input axis(%s) is not in fact a legal axis."%axis)

    def axis(self,label):
        '''
        Return the corresponding axis of a label.
        Parameters:
            label: any hashable object
                The label whose corresponding axis is inquired.
            Returns: int
                The corresponding axis.
        '''
        if label.__class__.__name__==self.label[0].__class__.__name__:
            return self.label.index(axis)
        elif isinstance(int):
            if label<len(self.labels):
                return label
            else:
                raise ValueError("Tensor axis error: the label(%s) doesn't correspond to any existing axis."%label)
        else:
            raise ValueError("Tensor axis error: the input label(%s) is not in fact a legal label."%label)

    def relabel(self,news,olds=None):
        '''
        Change the labels of the tensor and return the new one.
        Parameters:
            news: list of hashable objects
                The new labels of the tensor's axes.
            olds: list of hashable objects
                The old labels of the tensor's axes.
        Returns: Tensor
            The new tensor with the new labels.
        '''
        if olds is None:
            if self.ndim!=len(news):
                raise ValueError("Tensor relabel error: the number of lables and the dimension of tensors should be equal.")
            self.labels=news
        else:
            if len(news)!=len(olds):
                raise ValueError("Tensor relabel error: the number of new labels(%s) and old labels(%s) are not equal."%(len(news),len(olds)))
            for old,new in zip(olds,news):
                index=self.labels.index(old)
                self.labels[index]=new
        return self

    def reorder(self,paras):
        '''
        Change the order of the tensor's axes and return the new one.
        Parameters:
            paras: list of integers/hashable objects
                The permutation of the original axes.
        Returns: Tensor
            The new tensor with the reordered axes.
        '''
        if len(paras)!=len(self.labels):
            raise ValueError("Tensor reorder error: the number of axes doesn't match the tensor.")
        axes,labels=[],[]
        label_class=self.labels[0].__class__.__name__
        for axis in paras:
            if axis.__class__.__name__==label_class:
                axes.append(self.labels.index(axis))
                labels.append(axis)
            elif isinstance(axis,int):
                axes.append(axis)
                labels.append(self.labels[axis])
            else:
                raise ValueError("Tensor reorder error: the new order cannot be recognized.")
        return Tensor(super(Tensor,self).transpose(axes),labels=labels)

    def take(self,indices,axis):
        '''
        Take elements from a tensor along an axis.
        Parameters:
            indices: integer / list of integers
                The indices of the values to extract.
            axis: int or hashable object
                The axis along which to select values.
        Returns: Tensor
        '''
        if axis.__class__.__name__==self.labels[0].__class__.__name__:
            if isinstance(indices,int) or len(indices)==1:
                labels=[label for label in self.labels if label!=axis]
            else:
                labels=deepcopy(self.labels)
            return Tensor(super(Tensor,self).take(indices,axis=self.labels.index(axis)),labels=labels)
        elif isinstance(axis,int):
            if isinstance(indices,int) or len(indices)==1:
                labels=[label for i,label in enumerate(self.labels) if i!=axis]
            else:
                labels=deepcopy(self.labels)
            return Tensor(super(Tensor,self).take(indices,axis=axis),labels=labels)
        else:
            raise ValueError("Tensor take error: the only parameter should be a string or a integer.")

    def svd(self,labels1,new,labels2):
        '''
        Perform the svd.
        Parameters:
            labels1,labels2: list of any hashable object
                The axis labels of the two groups.
            new: any hashable object
                The new axis label after the svd.
        Returns:
            U,S,V: Tensor
        '''
        axes1,axes2,shape1,shape2=[],[],(),()
        for label in labels1:
            index=self.labels.index(label)
            axes1.append(index)
            shape1+=(self.shape[index],)
        for label in labels2:
            index=self.labels.index(label)
            axes2.append(index)
            shape2+=(self.shape[index],)
        if [i for i in xrange(len(self.labels)) if (i not in axes1) and (i not in axes2)]:
            raise ValueError('Tensor svd error: all axis should be divided into two group to perform the svd.')
        m=asarray(self).transpose(axes1+axes2).reshape((product(shape1),)+(product(shape2),))
        u,s,v=svd(m,full_matrices=False)
        U=Tensor(u.reshape(shape1+(-1,)),labels=labels1+[new])
        S=Tensor(s,labels=[new])
        V=Tensor(v.reshape((-1,)+shape2),labels=[new]+labels2)
        return U,S,V
