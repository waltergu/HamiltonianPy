'''
==========
TensorBase
==========

The base class for dense and sparse labeled multi-dimensional tensors, including
    * classes: Label, TensorBase
'''

import numpy as np
from copy import copy
from abc import ABCMeta,abstractproperty,abstractmethod
from HamiltonianPy import QuantumNumbers

__all__=['Label','TensorBase']

class Label(tuple):
    '''
    The label for a dimension of a tensor.

    Attributes
    ----------
    names : ('identifier','prime')
        The names of the immutable part of the label.
    qns : int or QuantumNumbers
        * When int, it is the dimension of the label;
        * When QuantumNumbers, it is the quantum numbers (i.e. the block structure) of the label.
    flow : -1/0/+1/None
        The flow of the quantum numbers, `-1` means flow out, `+1` means flow in, `0` means no flow and `None` means undefined.
    '''
    names=('identifier','prime')

    def __new__(cls,identifier,qns,flow=0,prime=False):
        '''
        Constructor.

        Parameters
        ----------
        identifier : any hashale object
            The identifier of the label.
        qns : int or QuantumNumbers
            * When int, it is the dimension of the label;
            * When QuantumNumbers, it is the quantum numbers (i.e. the block structure) of the label.
        flow : -1/0/+1/None, optional
            The flow of the quantum numbers.
        prime : logical, optional
            When True, the label is in the prime form; otherwise not.
        '''
        self=tuple.__new__(cls,(identifier,prime))
        self.qns=qns
        self.flow=flow
        return self

    @property
    def P(self):
        '''
        The prime of the label.
        '''
        return self.replace(prime=not self.prime,flow=None if self.flow is None else -self.flow)

    @property
    def inverse(self):
        '''
        Return a new label with the flow of the quantum numbers inversed.
        '''
        return self.replace(flow=-self.flow)

    @property
    def dim(self):
        '''
        The length of the dimension this label labels.
        '''
        return len(self.qns) if isinstance(self.qns,QuantumNumbers) else self.qns

    @property
    def qnon(self):
        '''
        True when good quantum numbers are used. Otherwise False.
        '''
        return isinstance(self.qns,QuantumNumbers)

    @staticmethod
    def union(labels,identifier,flow=0,prime=False,mode=0):
        '''
        The union of a set of labels as a new label.

        Parameters
        ----------
        labels : list of Label
            The set of labels.
        identifier : any hashale object
            The identifier of the union.
        flow : -1/0/+1/None
            The flow of the union.
        prime : logical, optional
            When True, the union is in the prime form; otherwise not.
        mode : -2/-1/0/+1/+2, optional
            * 0: When qnon, use QuantumNumbers.kron to get the qns of the union, no sorting and no returning the permutation array
            * -1: When qnon, use QuantumNumbers.kron to get the qns of the union with sorting but without returning the permutation array
            * +1: When qnon, use QuantumNumbers.kron to get the qns of the union with sorting and with returning the permutation array
            * -2: When qnon, use QuantumNumbers.ukron to get the qns of the union without returning the record dict
            * +2: When qnon, use QuantumNumbers.ukron to get the qns of the union with returning the record dict

        Returns
        -------
        result : Label
            The union of the input set of labels.
        permutation/record : 1d-ndarray-of-int/dict, optional
            The permutation/record of the quantum numbers of the union.
        '''
        qnon=labels[0].qnon
        assert all(label.qnon==qnon for label in labels[1:]) and mode in {-2,-1,0,1,2}
        if qnon:
            assert flow in {-1,1}
            qnses,signs=[label.qns for label in labels],[1 if label.flow==flow else -1 for label in labels]
            if mode in {-1,0,+1}:
                qns=QuantumNumbers.kron(qnses,signs=signs)
                if mode==-1: qns=qns.sorted(history=False)
                if mode==+1: qns,permutation=qns.sorted(history=True)
                result=Label(identifier,qns,flow=flow,prime=prime)
                return (result,permutation) if mode==1 else result
            else:
                if mode==-2: qns=QuantumNumbers.ukron(qnses,signs=signs,history=False)
                if mode==+2: qns,record=QuantumNumbers.ukron(qnses,signs=signs,history=True)
                result=Label(identifier,qns,flow=flow,prime=prime)
                return (result,record) if mode==2 else result
        else:
            result=Label(identifier,np.product([label.qns for label in labels]),flow=None if flow is None else 0,prime=prime)
            return (result,None) if mode in (1,2) else result

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return "Label(%s)%s<%s>%s"%(self.identifier,"'" if self.prime else "",self.qns,"+" if self.flow==1 else "-" if self.flow==-1 else "" if self.flow==0 else "*")

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        return "Label(%s)%s<%s>%s"%(repr(self.identifier),"'" if self.prime else "",repr(self.qns),"+" if self.flow==1 else "-" if self.flow==-1 else "" if self.flow==0 else "*")

    def __getnewargs__(self):
        '''
        Return the arguments for Label.__new__, required by copy and pickle.
        '''
        return self.identifier,self.qns,self.flow,self.prime

    def __getstate__(self):
        '''
        Since Label.__new__ constructs everything, self.__dict__ can be omitted for copy and pickle.
        '''
        pass

    def __getattr__(self,key):
        '''
        Overloaded operator(.).
        '''
        try:
            return self[type(self).names.index(key)]
        except ValueError:
            raise AttributeError("'Label' object has no attribute %s."%key)

    def replace(self,**karg):
        '''
        Return a new label with some of its attributes replaced.

        Parameters
        ----------
        karg : dict in the form (key,value), with
            * key: str
                The attributes of the label
            * value: any object
                The corresponding value.

        Returns
        -------
        Label
            The new label.
        '''
        result=tuple.__new__(self.__class__,map(karg.pop,type(self).names,self))
        for key,value in self.__dict__.items():
            setattr(result,key,karg.pop(key,value))
        if karg:
            raise ValueError("Label replace error: %s are not the attributes of the label."%list(karg.keys()))
        return result

    def equivalent(self,other):
        '''
        Judge whether two labels are equivalent to each other.

        Parameters
        ----------
        other : Label
            The other label.

        Returns
        -------
        logical
            True when the two labels are equivalent to each other, otherwise False.
        '''
        return self==other and self.qns==other.qns and self.flow==other.flow

class TensorBase(object,metaclass=ABCMeta):
    '''
    The base class for dense and sparse labeled multi-dimensional tensors.

    Attributes
    ----------
    data : object
        The data of the tensor.
    labels : list of Label
        The labels of the axes of the tensor.
    '''
    DIMCHECK=False

    @property
    def shape(self):
        '''
        The shape of the tensor.
        '''
        return tuple(label.dim for label in self.labels)

    @property
    def ndim(self):
        '''
        The dimension of the tensor.
        '''
        return len(self.labels)

    @property
    def qnon(self):
        '''
        True for the labels using good quantum numbers otherwise False.
        '''
        return next(iter(self.labels)).qnon if len(self.labels)>0 else False

    def __copy__(self):
        '''
        The shallow copy of a tensor.
        '''
        return type(self)(self.data,copy(self.labels))

    def label(self,axis):
        '''
        Return the corresponding label of an axis.

        Parameters
        ----------
        axis : int
            The axis whose corresponding label is inquired.

        Returns
        -------
        Label
            The corresponding label.
        '''
        return self.labels[axis]

    def axis(self,label):
        '''
        Return the corresponding axis of a label.

        Parameters
        ----------
        label : Label
            The label whose corresponding axis is inquired.

        Returns
        -------
        int
            The corresponding axis.
        '''
        return self.labels.index(label)

    def reflow(self,axes=None):
        '''
        Reverse the flows of some axes of the tensor.

        Parameters
        ----------
        axes : list of Label/int, optional
            The labels/axes whose flows to be reversed.
        '''
        axes=[self.axis(axis) if isinstance(axis,Label) else axis for axis in (range(self.ndim) if axes is None else axes)]
        self.relabel(olds=axes,news=[self.labels[axis].inverse for axis in axes])

    @abstractproperty
    def dtype(self):
        '''
        The data type of the tensor.
        '''
        raise NotImplementedError('%s dtype error: not implemented.'%self.__class__.__name__)

    @abstractproperty
    def ttype(self):
        '''
        Tensor type.
        '''
        raise NotImplementedError('%s ttype error: not implemented.'%self.__class__.__name__)

    @abstractproperty
    def norm(self):
        '''
        The norm of the tensor.
        '''
        raise NotImplementedError('%s norm error: not implemented.'%self.__class__.__name__)

    @abstractproperty
    def dagger(self):
        '''
        The dagger of the tensor.
        '''
        raise NotImplementedError('%s dagger error: not implemented.'%self.__class__.__name__)

    @abstractmethod
    def dimcheck(self):
        '''
        Check whether or not the dimensions of the labels and the data match each other.

        Returns
        --------
        logical
            True for match and False for not.
        '''
        raise NotImplementedError('%s dimcheck error: not implemented.'%self.__class__.__name__)

    @abstractmethod
    def toarray(self):
        '''
        Convert to ndarray.

        Returns
        -------
        ndarray
            The ndarray representation of the tensor.
        '''
        raise NotImplementedError('%s toarray error: not implemented.'%self.__class__.__name__)

    @abstractmethod
    def dotarray(self,axis,array):
        '''
        Multiply a certain axis of a tensor with an array.

        Parameters
        ----------
        axis : Label/int
            The Label/axis of the tensor to be multiplied.
        array : 1d ndarray
            The multiplication array.

        Returns
        -------
        subclass of TensorBase
            The new tensor.
        '''
        raise NotImplementedError('%s dotarray error: not implemented.'%self.__class__.__name__)

    @abstractmethod
    def relabel(self,news,olds=None):
        '''
        Change the labels of the tensor.

        Parameters
        ----------
        news : list of Label
            The new labels of the tensor.
        olds : list of Label/int, optional
            The old labels/axes of the tensor.
        '''
        raise NotImplementedError('%s relabel error: not implemented.'%self.__class__.__name__)

    @abstractmethod
    def transpose(self,axes=None):
        '''
        Change the order of the tensor's axes and return the new tensor.

        Parameters
        ----------
        axes : list of Label/int, optional
            The permutation of the original labels/axes.

        Returns
        -------
        subclass of TensorBase
            The new tensor with the reordered axes.
        '''
        raise NotImplementedError('%s transpose error: not implemented.'%self.__class__.__name__)

    @abstractmethod
    def take(self,index,axis):
        '''
        Take the index-th elements along an axis of a tensor.

        Parameters
        ----------
        index : int
            The index of the elements to be extracted.
        axis : int/Label
            The axis along which to extract elements.

        Returns
        -------
        subclass of TensorBase
            The extracted tensor.
        '''
        raise NotImplementedError('%s take error: not implemented.'%self.__class__.__name__)

    @abstractmethod
    def reorder(self,*args):
        '''
        Reorder a dimension of a tensor with a permutation and optionally set a new qns for this dimension.

        Usage: ``tensor.reorder((axis,permutation,<qns>),(axis,permutation,<qns>),...)``
            * axis: int/Label
                The axis of the dimension to be reordered.
            * permutation: 1d ndarray of int
                The permutation array.
            * qns: QuantumNumbers, optional
                The new quantum number collection of the dimension if good quantum numbers are used.

        Returns
        -------
        subclass of TensorBase
            The reordered tensor.

        Notes
        -----
        If `qns` is not passed, the new qns will be automatically set according to the permutation array.
        '''
        raise NotImplementedError('%s reorder error: not implemented.'%self.__class__.__name__)

    @abstractmethod
    def merge(self,*args):
        '''
        Merge some continuous and ascending labels of a tensor into a new one with the help of optional extra info.

        Usage: ``tensor.merge((olds,new,<info>),(olds,new,<info>),...)``
            * olds: list of Label/int
                The old labels/axes to be merged.
            * new: Label
                The new label.
            * info: object, optional
                Extra info.

        Returns
        -------
        subclass of TensorBase
            The new tensor.
        '''
        raise NotImplementedError('%s merge error: not implemented.'%self.__class__.__name__)

    @abstractmethod
    def split(self,*args):
        '''
        Split a label into small ones with the help of optional extra info.

        Usage: ``tensor.split((old,news,<info>),(old,news,<info>),...)``
            * old: Label/int
                The label/axis to be split.
            * news: list of Label
                The new labels.
            * info: object, optional
                Extra info.

        Returns
        -------
        subclass of TensorBase
            The new tensor.
        '''
        raise NotImplementedError('%s split error: not implemented.'%self.__class__.__name__)
