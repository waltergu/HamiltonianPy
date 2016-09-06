'''
Tensor and tensor operations, including:
1) classes: Tensor
2) functions: contract
'''

from numpy import *
from numpy.linalg import svd
from opt_einsum import contract as einsum
from copy import deepcopy

__all__=['contract','Tensor']

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
    labels_list=[tensor.labels for tensor in tensors]
    labels_all=concatenate(labels_list)
    labels_unique,counts=unique(labels_all,return_counts=True)
    table={key:value for value,key in enumerate(labels_unique)}
    subscripts=[''.join(chr(table[label]+97) for label in labels) for labels in labels_list]
    contracted_labels=[label for label in labels_all if counts[table[label]]==1]
    contracted_subscript=''.join(chr(table[label]+97) for label in contracted_labels)
    return Tensor(einsum('%s->%s'%(','.join(subscripts),contracted_subscript),*tensors),labels=contracted_labels)

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
        u,s,v=svd(m,full_matrices=False)
        U=Tensor(u.reshape(shape1+(-1,)),labels=labels1+[new])
        S=Tensor(s,labels=[new])
        V=Tensor(v.reshape((-1,)+shape2),labels=[new]+labels2)
        return U,S,V

    def components(self,zero=0.0):
        '''
        Returns a list of 2-tuple which contains all the indices and values for the non-zero components of the tensor.
        Parameters:
            zero: float64, optional
                The user defined zero.
        '''
        return [(tuple(index),self[tuple(index)]) for index in argwhere(abs(self)>zero)]
