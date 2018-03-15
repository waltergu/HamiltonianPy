'''
======
Tensor
======

Dense and sparse labled multi-dimensional tensors, including:
    * classes: DTensor, STensor
    * functions: Tensor, contract
'''

import numpy as np
import itertools as it
import HamiltonianPy.Misc as hm
from warnings import warn
from copy import copy
from collections import OrderedDict
from HamiltonianPy import QuantumNumbers,Arithmetic
from TensorBase import Label,TensorBase

__all__=['DTensor','STensor','Tensor','contract']

class DTensor(TensorBase,Arithmetic):
    '''
    Dense tensor class with labeled axes.

    Attributes
    ----------
    data : ndarray
        The data of the tensor.
    labels : list of Label
        The labels of the axes of the tensor.
    '''

    def __init__(self,data,labels):
        '''
        Constructor.

        Parameters
        ----------
        data : ndarray
            The data of the tensor.
        labels : list of Label
            The labels of the axes of the tensor.
        '''
        self.data=np.asarray(data)
        self.labels=labels
        assert self.data.ndim==len(self.labels) and self.dimcheck()

    @property
    def dtype(self):
        '''
        The data type of the tensor.
        '''
        return self.data.dtype

    @property
    def norm(self):
        '''
        The norm of the tensor.
        '''
        return np.linalg.norm(self.data)

    @property
    def dagger(self):
        '''
        The dagger of the tensor.
        '''
        return DTensor(self.data.conjugate(),labels=[label.P for label in self.labels])

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return "DTensor(\nlabels=%s,\ndata=%s%s\n)"%(self.labels,'\n' if self.ndim>1 else '',self.data)

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        return "DTensor(\nlabels=%s,\ndata=%s%s\n)"%(repr(self.labels),'\n' if self.ndim>1 else '',repr(self.data))

    def __iadd__(self,other):
        '''
        Overloaded self-addition(+=) operator.
        '''
        if isinstance(other,DTensor):
            assert self.ndim==other.ndim and all(l1.equivalent(l2) for l1,l2 in zip(self.labels,other.labels))
            self.data+=other.data
        else:
            self.data+=other
        return self

    def __add__(self,other):
        '''
        Overloaded left addition(+) operator.
        '''
        if isinstance(other,DTensor):
            assert self.ndim==other.ndim and all(l1.equivalent(l2) for l1,l2 in zip(self.labels,other.labels))
            data=self.data+other.data
        else:
            data=self.data+other
        return DTensor(data,self.labels)

    def __imul__(self,other):
        '''
        Overloaded self-multiplication(*=) operator.
        '''
        if isinstance(other,DTensor) or isinstance(other,STensor):
            return contract(self,other)
        elif isinstance(other,tuple):
            assert len(other)==2
            return contract(self,other[0],other[1])
        else:
            self.data*=other
            return self

    def __mul__(self,other):
        '''
        Overloaded left multiplication(*) operator.
        '''
        if isinstance(other,DTensor) or isinstance(other,STensor):
            return contract(self,other)
        elif isinstance(other,tuple):
            assert len(other)==2
            return contract(self,other[0],other[1])
        else:
            return DTensor(self.data*other,self.labels)

    def __eq__(self,other):
        '''
        Overloaded equivalent(==) operator.
        '''
        if isinstance(other,DTensor):
            assert self.ndim==other.ndim and all(l1.equivalent(l2) for l1,l2 in zip(self.labels,other.labels))
            return self.data==other.data
        else:
            return self.data==other

    def toarray(self):
        '''
        Convert to ndarray.

        Returns
        -------
        ndarray
            The ndarray representation of the tensor.
        '''
        return self.data

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
        DTensor
            The new tensor.
        '''
        return self*array[[slice(None) if i==(self.axis(axis) if isinstance(axis,Label) else axis) else np.newaxis for i in xrange(self.ndim)]]

    def reflow(self,axes=None):
        '''
        Reverse the flows of some axes of the tensor.

        Parameters
        ----------
        axes : list of Label/int, optional
            The labels/axes whose flows to be reversed.
        '''
        for axis in xrange(self.ndim) if axes is None else axes:
            axis=self.axis(axis) if isinstance(axis,Label) else axis
            self.labels[axis]=self.labels[axis].inverse

    def transpose(self,axes=None):
        '''
        Change the order of the tensor's axes and return the new tensor.

        Parameters
        ----------
        axes : list of Label/int, optional
            The permutation of the original labels/axes.

        Returns
        -------
        DTensor
            The new tensor with the reordered axes.
        '''
        axes=range(self.ndim-1,-1,-1) if axes is None else [self.axis(axis) if isinstance(axis,Label) else axis for axis in axes]
        return DTensor(self.data.transpose(axes),labels=[self.labels[axis] for axis in axes])

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
        DTensor
            The extracted tensor.
        '''
        if isinstance(axis,Label): axis=self.axis(axis)
        return DTensor(self.data.take(index,axis),[label for i,label in enumerate(self.labels) if i!=axis])

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
        DTensor
            The reordered tensor.

        Notes
        -----
        If `qns` is not passed, the new qns will be automatically set according to the permutation array.
        '''
        result,qnon=copy(self),self.qnon
        for arg in args:
            assert len(arg) in (2,3)
            axis,permutation=arg[0],arg[1]
            if permutation is not None:
                axis=self.axis(axis) if isinstance(axis,Label) else axis
                label=result.labels[axis]
                result.labels[axis]=label.replace(qns=arg[2] if len(arg)==3 else label.qns.reorder(permutation) if qnon else len(permutation))
                result.data=hm.reorder(result.data,axes=[axis],permutation=permutation)
        return result

    def merge(self,*args):
        '''
        Merge some continuous and ascending labels of a tensor into a new one with an optional permutation.

        Usage: ``tensor.merge((olds,new,<permutation>),(olds,new,<permutation>),...)``
            * olds: list of Label/int
                The old labels/axes to be merged.
            * new: Label
                The new label.
            * permutation: 1d ndarray of int, optional
                The permutation of the quantum number collection of the new label.

        Returns
        -------
        DTensor
            The new tensor.
        '''
        permutations={}
        keep=OrderedDict([(i,i) for i in xrange(self.ndim)])
        labels=OrderedDict([(i,label) for i,label in enumerate(self.labels)])
        for arg in args:
            assert len(arg) in (2,3)
            olds,new,permutation=(arg[0],arg[1],None) if len(arg)==2 else arg
            axes=np.array([self.axis(old) if isinstance(old,Label) else old for old in olds])
            if len(axes)!=max(axes)-min(axes)+1 or not all(axes[1:]>axes[:-1]):
                raise ValueError('DTensor merge error: the axes to be merged should be continuous and ascending, please call transpose first.')
            permutations[new]=permutation
            keep[axes[0]]=slice(axes[0],axes[-1]+1)
            labels[axes[0]]=new
            for axis in axes[1:]:
                keep.pop(axis)
                labels.pop(axis)
        data=self.data.reshape(tuple(np.product(self.data.shape[ax]) if isinstance(ax,slice) else self.data.shape[ax] for ax in keep.itervalues()))
        labels=labels.values()
        for label,permutation in permutations.iteritems():
            data=hm.reorder(data,axes=[labels.index(label)],permutation=permutation)
        return DTensor(data,labels=labels)

    def split(self,*args):
        '''
        Split a label into small ones with an optional permutation.

        Usage: ``tensor.split((old,news,<permutation>),(old,news,<permutation>),...)``
            * old: Label/int
                The label/axis to be split.
            * news: list of Label
                The new labels.
            * permutation: 1d ndarray of int, optional
                The permutation of the quantum number collection of the old label.

        Returns
        -------
        DTensor
            The new tensor.
        '''
        table={(self.axis(arg[0]) if isinstance(arg[0],Label) else arg[0]):i for i,arg in enumerate(args)}
        shape,labels,axes,permutations=(),[],[],[]
        for axis,dim,label in zip(xrange(self.ndim),self.shape,self.labels):
            if axis in table:
                arg=args[table[axis]]
                assert len(arg) in (2,3)
                shape+=tuple(new.dim for new in arg[1])
                labels.extend(arg[1])
                axes.append(axis)
                permutations.append(arg[2] if len(arg)==3 else None)
            else:
                shape+=(dim,)
                labels.append(label)
        data=self.data
        for axis,permutation in zip(axes,permutations):
            data=hm.reorder(data,axes=[axis],permutation=permutation)
        return DTensor(data.reshape(shape),labels=labels)

    def dimcheck(self):
        '''
        Check whether or not the dimensions of the labels and the data match each other.

        Returns
        --------
        logical
            True for match and False for not.
        '''
        return all([label.dim==dim for label,dim in zip(self.labels,self.data.shape)])

    def qngenerate(self,flow,axes,qnses,flows,tol=None):
        '''
        Generate the quantum numbers of a tensor.

        Parameters
        ----------
        flow : +1/-1
            The flow of the unknown axis.
        axes : list of Label/int
            The labels/axes whose quantum number collections are known.
        qnses : list of QuantumNumbers
            The quantum number collections of the known axes.
        flows : tuple of int
            The flows of the quantum numbers of the known axes.
        tol : float64, optional
            The tolerance of the non-zeros.
        '''
        axes=[self.axis(axis) if isinstance(axis,Label) else axis for axis in axes]
        assert flow in {-1,1} and len(axes)==len(qnses)==len(flows)==self.ndim-1
        type=next(iter(qnses)).type
        for axis,qns,f in zip(axes,qnses,flows):
            self.labels[axis].qns=qns
            self.labels[axis].flow=f
            assert qns.type is type
        unkownaxis=(set(xrange(self.ndim))-set(axes)).pop()
        expansions=[qns.expansion() for qns in qnses]
        contents=[None]*self.shape[unkownaxis]
        for index in sorted(np.argwhere(np.abs(self.data)>hm.TOL if tol is None else tol),key=lambda index: index[unkownaxis]):
            qn=type.regularization(sum([expansions[i][index[axis]]*f*(-flow) for i,(axis,f) in enumerate(zip(axes,flows))]))
            if contents[index[unkownaxis]] is None:
                contents[index[unkownaxis]]=qn
            else:
                assert (contents[index[unkownaxis]]==qn).all()
        self.labels[unkownaxis].qns=QuantumNumbers('G',(type,contents,np.arange(len(contents)+1)),protocol=QuantumNumbers.INDPTR)
        self.labels[unkownaxis].flow=flow

    def qnsort(self,history=False):
        '''
        Sort the quantum numbers of all the dimensions of the tensor.

        Returns
        -------
        list of 1d ndarray, optional
            The permutation arrays of the dimensions of the tensor.
            Returned only when ``history`` is True.
        '''
        permutations=[]
        for axis,label in enumerate(self.labels):
            assert label.qnon
            qns,permutation=(label.qns,None) if label.qns.form=='C' else label.qns.sorted(history=True)
            self.data=hm.reorder(self.data,axes=[axis],permutation=permutation)
            self.labels[axis]=label.replace(qns=qns)
            if history: permutations.append(permutation)
        if history: return permutations

    def tostensor(self):
        '''
        Convert to sparse tensor.

        Returns
        -------
        STensor
            The converted sparse tensor.
        '''
        assert all(label.qnon for label in self.labels) and self.ndim>1
        ods=[label.to_ordereddict() for label in self.labels]
        data={}
        for qns in it.product(*ods[:-1]):
            key=tuple(it.chain(qns,[-sum(qns)]))
            if key[-1] in ods[-1]: data[key]=self.data[[od[qn] for od,qn in zip(ods,key)]]
        return STensor(data,labels=self.labels)

class STensor(TensorBase,Arithmetic):
    '''
    Sparse tensor class with labeled axes.

    Attributes
    ----------
    data : dict in the form {key:block}
        The data of the tensor, with

            * key: tuple of QuantumNumber, the quantum number of the block
            * block: ndarray, the data of the block

    labels : list of Label
        The labels of the axes of the tensor.
    '''

    def __init__(self,data,labels):
        '''
        Constructor.

        Parameters
        ----------
        data : dict in the form {key:block}
            The data of the tensor, with

                * key: tuple of QuantumNumber, the quantum number of the block
                * block: ndarray, the data of the block

        labels : list of Label
            The labels of the axes of the tensor.
        '''
        assert len(labels)>1 and all(label.qnon and label.qns.form in 'UC' for label in labels)
        self.data=data
        self.labels=labels

    @property
    def dtype(self):
        '''
        The data type of the tensor.
        '''
        return next(self.data.itervalues()).dtype

    @property
    def norm(self):
        '''
        The norm of the tensor.
        '''
        return np.linalg.norm([np.sum(block*block) for block in self.data.itervalues()])

    @property
    def dagger(self):
        '''
        The dagger of the tensor.
        '''
        return Tensor({key:block.conjugate() for key,block in self.data.iteritems()},labels=[label.P for label in self.labels])

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return "STensor(\nlabels=%s\n%s\n)"%(self.labels,'\n'.join('%s:\n%s'%(key,block) for key,block in self.data.iteritems()))

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        return "STensor(\nlabels=%s\n%s\n)"%(self.labels,'\n'.join('%s: %s'%(key,block.shape) for key,block in self.data.iteritems()))

    def __iadd__(self,other):
        '''
        Overloaded self-addition(+=) operator.
        '''
        if isinstance(other,STensor):
            assert self.ndim==other.ndim and all(l1.equivalent(l2) for l1,l2 in zip(self.labels,other.labels))
            for key in self.data: self.data[key]+=other.data[key]
        else:
            for key in self.data: self.data[key]+=other
        return self

    def __add__(self,other):
        '''
        Overloaded left addition(+) operator.
        '''
        if isinstance(other,STensor):
            assert self.ndim==other.ndim and all(l1.equivalent(l2) for l1,l2 in zip(self.labels,other.labels))
            data={key:self.data[key]+other.data[key] for key in self.data}
        else:
            data={key:self.data[key]+other for key in self.data}
        return STensor(data,self.labels)

    def __imul__(self,other):
        '''
        Overloaded self-multiplication(*=) operator.
        '''
        if isinstance(other,STensor) or isinstance(other,DTensor):
            return contract(self,other)
        else:
            for key in self.data: self.data[key]*=other
            return self

    def __mul__(self,other):
        '''
        Overloaded left multiplication(*) operator.
        '''
        if isinstance(other,STensor) or isinstance(other,DTensor):
            return contract(self,other)
        else:
            return STensor({key:self.data[key]*other for key in self.data},self.labels)

    def __eq__(self,other):
        '''
        Overloaded equivalent(==) operator.
        '''
        if isinstance(other,STensor):
            assert self.ndim==other.ndim and all(l1.equivalent(l2) for l1,l2 in zip(self.labels,other.labels))
            return self.data==other.data
        else:
            return {key:block==other for key,block in self.data.iteritems()}

    def toarray(self):
        '''
        Convert to ndarray.

        Returns
        -------
        ndarray
            The ndarray representation of the tensor.
        '''
        result=np.zeros(self.shape,dtype=self.dtype)
        ods=[label.qns.to_ordereddict() for label in self.labels]
        for key,block in self.data.iteritems():
            result[[ods[i][qn] for i,qn in enumerate(key)]]=block
        return result

    def dotarray(self,axis,array):
        '''
        Multiply a certain axis of a tensor with an array.

        Parameters
        ----------
        axis : int
            The axis of the tensor to be multiplied.
        array : 1d ndarray
            The multiplication array.

        Returns
        -------
        STensor
            The new tensor.
        '''
        if isinstance(axis,Label): axis=self.axis(axis)
        od=self.labels[axis].qns.to_ordereddict()
        data={key:block*array[[od[key[axis]] if i==axis else np.newaxis for i in xrange(self.ndim)]] for key,block in self.data.iteritems()}
        return STensor(data,labels=self.labels)

    def reflow(self,axes=None):
        '''
        Reverse the flows of some axes of the tensor.

        Parameters
        ----------
        axes : list of Label/int, optional
            The labels/axes whose flows to be reversed.
        '''
        axes={self.axis(axis) if isinstance(axis,Label) else axis for axis in (xrange(self.ndim) if axes is None else axes)}
        self.labels=[label.inverse if i in axes else label for i,label in enumerate(self.labels)]
        self.data={tuple(-qn if i in axes else qn for i,qn in enumerate(key)):block for key,block in self.data.iteritems()}

    def transpose(self,axes=None):
        '''
        Change the order of the tensor's axes and return the new tensor.

        Parameters
        ----------
        axes : list of Label/int, optional
            The permutation of the original labels/axes.

        Returns
        -------
        Tensor
            The new tensor with the reordered axes.
        '''
        axes=range(self.ndim-1,-1,-1) if axes is None else [self.axis(axis) if isinstance(axis,Label) else axis for axis in axes]
        labels=[self.labels[axis] for axis in axes]
        data={tuple(key[axis] for axis in axes):block.transpose(axes) for key,block in self.data.iteritems()}
        return STensor(data,labels=labels)

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
        STensor
            The extracted tensor.
        '''
        if isinstance(axis,Label): axis=self.axis(axis)
        qn=self.labels[axis].qns[index]
        index=index-self.labels[axis].qns.to_ordereddict()[qn].start
        data={tuple(key[i] for i in xrange(self.ndim) if i!=axis): block.take(index,axis) for key,block in self.data.iteritems() if key[axis]==qn}
        return STensor(data,labels=[label for i,label in enumerate(self.labels) if i!=axis])

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
        STensor
            The reordered tensor.

        Notes
        -----
        If `qns` is not passed, the new qns will be automatically set according to the permutation array.
        '''
        result=copy(self)
        for arg in args:
            assert len(arg) in (2,3)
            axis,permutation=arg[0],arg[1]
            if permutation is not None:
                axis=self.axis(axis) if isinstance(axis,Label) else axis
                label=result.labels[axis]
                result.labels[axis]=label.replace(qns=arg[2] if len(arg)==3 else label.qns.reorder(permutation,protocol='CONTENTS'))
        return result

    def merge(self,*args):
        '''
        Merge some continuous and ascending labels of a tensor into a new one with a given record.

        Usage: ``tensor.merge((olds,new,record),(olds,new,record),...)``
            * olds: list of Label/int
                The old labels/axes to be merged.
            * new: Label
                The new label.
            * record: dict
                The record of the quantum number collection of the new label.

        Returns
        -------
        STensor
            The new tensor.

        Notes
        -----
        The record is necessary in order to make the quantum numbers of the merged label canonical.
        '''
        signs={}
        keep=OrderedDict([(i,i) for i in xrange(self.ndim)])
        records=OrderedDict([(i,None) for i in xrange(self.ndim)])
        labels=OrderedDict([(i,label) for i,label in enumerate(self.labels)])
        for olds,new,record in args:
            axes=np.array([self.axis(old) if isinstance(old,Label) else old for old in olds])
            if len(axes)!=max(axes)-min(axes)+1 or not all(axes[1:]>axes[:-1]):
                raise ValueError('STensor merge error: the axes to be merged should be continuous and ascending, please call transpose first.')
            signs.update({axis: +1 if new.flow==self.labels[axis] else -1 for axis in axes})
            keep[axes[0]]=slice(axes[0],axes[-1]+1)
            records[axes[0]]=record
            labels[axes[0]]=new
            for axis in axes[1:]:
                keep.pop(axis)
                records.pop(axis)
                labels.pop(axis)
        data={}
        counts=[label.qns.to_ordereddict(protocol=QuantumNumbers.COUNTS) for label in labels.itervalues()]
        for qns,block in self.data.iteritems():
            new=tuple(sum(qns[axis]*signs[axis] for axis in xrange(ax.start,ax.stop)) if isinstance(ax,slice) else qns[ax] for ax in keep.itervalues())
            shape=tuple(np.product(block.shape[ax]) if isinstance(ax,slice) else block.shape[ax] for ax in keep.itervalues())
            slices=[record[qns[ax]] if isinstance(ax,slice) else slice(None) for ax,record in zip(keep.itervalues(),records.itervalues())]
            if new not in data: data[new]=np.zeros(tuple(count[qn] for count,qn in zip(counts,new)),dtype=block.dtype)
            data[new][slices]=block.reshape(shape)
        return STensor(data,labels=labels.values())

    def split(self,*args):
        '''
        Split a label into small ones with a given record.

        Usage: ``tensor.split((old,news,record),(old,news,record),...)``
            * old: Label/int
                The label/axis to be split.
            * news: list of Label
                The new labels.
            * record: dict
                The record of the quantum number collection of the new label.

        Returns
        -------
        STensor
            The new tensor.

        Notes
        -----
        The record is necessary because the quantum numbers of the old label is forced to be canonical/unitary.
        '''
        axes=[self.axis(arg[0]) if isinstance(arg[0],Label) else arg[0] for arg in args]
        records=[arg[2] for arg in args]
        table={axis:seq for seq,axis in enumerate(axes)}
        labels=list(it.chain(*[args[table[axis]][1] if axis in table else [label] for axis,label in enumerate(self.labels)]))
        counts=[label.qns.to_ordereddict(protocol=QuantumNumbers.COUNTS) for label in labels]
        data={}
        for qns,block in self.data.iteritems():
            for content in it.product(*[record[qns[axis]].iteritems() for axis,record in zip(axes,records)]):
                new=tuple(it.chain(*[content[table[axis]][0] if axis in table else [qn] for axis,qn in enumerate(qns)]))
                slices=[content[table[axis]][1] if axis in table else slice(None) for axis in xrange(len(qns))]
                assert new not in data
                data[new]=block[slices].reshape(tuple(count[qn] for count,qn in zip(counts,new)))
        return STensor(data,labels=labels)

    def todtensor(self):
        '''
        Convert to dense tensor.

        Returns
        -------
        DTensor
            The converted dense tensor.
        '''
        return DTensor(self.toarray(),labels=self.labels)

    def to_dict(self,masks=()):
        '''
        Convert the sparse tensor to dict.

        Parameters
        ----------
        masks : tuple of int/Label
            The axis/label of the tensor to be masked.

        Returns
        -------
        dict
            The converted dict.
        '''
        result={}
        masks=[self.axis(label) if isinstance(label,Label) else label for label in masks]
        keeps=[axis for axis in self.ndim if axis not in masks]
        for key,block in self.data.iteritems():
            mask,keep=tuple(key[axis] for axis in masks),tuple(key[axis] for axis in keeps)
            if mask not in result: result[mask]=[]
            result[mask].append((keep,block))
        return result

def Tensor(data,labels):
    '''
    A unified constrcutor for both dense tensors and sparse tensors.

    Parameters
    ----------
    data : ndarray/dict
        The data of the tensor.
    labels : list of Label
        The labels of the tensor.

    Returns
    -------
    DTensor/STensor
        The constructed tensor.
    '''
    return (STensor if isinstance(data,dict) else DTensor)(data,labels)

def contract(a,b,engine=None):
    '''
    The contraction of two tensors.

    Parameters
    ----------
    a,b : DTensor/STensor
        The tensors to be contracted, which should be of the same type or one sparse and another 0d/1d dense.
        Only when both of them are dense tensors, will the parameter `engine` be considered.
    engine : 'block'/'tensordot'/'ftensordot'/'einsum'/None, optional
        * 'block': use the block-structure of tensors to perform the contraction
            This engine is valid only when
                1) `a` and `b` use good quantum numbers,
                2) `a` and `b` share common labels, and
                3) all the flow pairs of the shared labels sum to be zero.
        * 'tensordot': use numpy.tensordot to perform the contraction
            This engine is valid only when no flows of the shared labels are `None`.
        * 'ftensordot': use numpy.tensordot to perform the contraction
            This engine ignores the flows of the shared labels.
        * 'einsum': use numpy.einsum to perform the contraction
            * This engine is valid only when the total number of different labels is <=52.
            * This is the default engine when there exists a shared label whose flow is `None`.

    Returns
    -------
    DTensor/STensor
        The contracted tensor.

    Notes
    -----
    *   Two labels will be contracted if
            1) they share the same immutable part,
            2) neither of their flows is `None`,
            3) the sum of their flows is zero.
    *   When `a` and `b` are dense:
        *   When engine is 'ftensordot', Rule.2 and Rule.3 will be omitted, i.e. as long as Rule.1 is satisfied, the labels will be contracted.
        *   When Rule.1 and Rule.2 is satisfied but Rule.3 is not, an exception will be raised except when engine is 'ftensordot'.
        *   When Rule.1 is satisfied but Rule.2 is not, and when engine is NOT 'ftensordot', an elementwise multiplication will be performed along
            the corresponding axis, and the label whose flow is not `None` will be kept for this axis, i.e. only multiplication yet no summation.
        *   As to the efficiency, usually 'block'>'tensordot'>'einsum'. Therefore, `None` is recommeded and let the program choose the optimal.
    *   When `a` and `b` are sparse or one sparse and another 0d/1d dense:
        *   The dimension of sparse tensors must be greater than 1 and no flows of sparse tensors are allowed to be `None`.
        *   The contraction of an stensor with a 1d dtensor is to multiply the corresponding 1d array to the stensor along the corresponding axis.
    '''
    assert type(a) is type(b) or a.ndim<=1 or b.ndim<=1
    common=set(a.labels)&set(b.labels)
    AL,AC,BL,BC=[],[],[],[]
    for label in a.labels: (AC if label in common else AL).append(label)
    for label in b.labels: (BC if label in common else BL).append(label)
    BC=[BC[BC.index(label)] for label in AC]
    if isinstance(a,DTensor) and isinstance(b,DTensor):
        if engine!='ftensordot':
            keep={}
            for l1,l2 in zip(AC,BC):
                if l1.flow is None or l2.flow is None:
                    keep[l1]=l2 if l1.flow is None else l1
                elif l1.flow+l2.flow!=0:
                    raise ValueError('tensor contraction error: not converged flow for %s and %s'%(repr(l1),repr(l2)))
            if len(keep)>0:
                if engine not in {'einsum',None}: warn("tensor contract warning: None flow encountered and engine will be switched to 'einsum'.")
                if len(AL)+len(BL)+len(common)>52: raise ValueError("tensor contract error: too many(%s) labels for 'einsum'."%(len(AL)+len(BL)+len(common)))
                engine='einsum'
            elif len(common)==0 or len(AL)==0 or len(BL)==0 or not a.qnon or not b.qnon:
                if engine=='block': warn("tensor contraction warning: engine cannot be 'block' and will be switched to 'tensordot'.")
                if engine in {'block',None}: engine='tensordot'
            elif engine is None:
                engine='block'
        if engine=='tensordot' or engine=='ftensordot':
            return Tensor(np.tensordot(a.data,b.data,axes=([a.axis(label) for label in common],[b.axis(label) for label in common]) if len(common)>0 else 0),labels=AL+BL)
        elif engine=='einsum':
            table={key:chr(i+65) if i<26 else chr(i+97) for i,key in enumerate(set(a.labels)|set(b.labels))}
            labels=[keep.pop(label,label) for label in it.chain(a.labels,b.labels) if label in keep or label not in common]
            scripts='%s,%s->%s'%(''.join(table[label] for label in a.labels),''.join(table[label] for label in b.labels),''.join(table[label] for label in labels))
            return Tensor(np.einsum(scripts,a.data,b.data),labels=labels)
        else:
            A=a.transpose(AL+AC).data.reshape((np.product([label.dim for label in AL]),np.product([label.dim for label in AC])))
            B=b.transpose(BC+BL).data.reshape((np.product([label.dim for label in BC]),np.product([label.dim for label in BL])))
            l,lpermutation=Label.union(AL,None,+1,mode=1)
            c,apermutation=Label.union(AC,None,-1,mode=1)
            c,bpermutation=Label.union(BC,None,+1,mode=1)
            r,rpermutation=Label.union(BL,None,-1,mode=1)
            lod,cod,rod=l.qns.to_ordereddict(),c.qns.to_ordereddict(),r.qns.to_ordereddict()
            result=np.zeros((A.shape[0],B.shape[1]),dtype=np.find_common_type([A.dtype,B.dtype],[]))
            for qn in it.ifilter(lambda qn:True if lod.has_key(qn) and rod.has_key(qn) else False,cod):
                linds=lpermutation[lod[qn]]
                ainds=apermutation[cod[qn]]
                binds=bpermutation[cod[qn]]
                rinds=rpermutation[rod[qn]]
                result[linds[:,None],rinds]=A[linds[:,None],ainds].dot(B[binds[:,None],rinds])
            return Tensor(result.reshape(tuple(label.dim for label in it.chain(AL,BL))),labels=AL+BL)
    elif a.ndim<=1:
        return b.dotarray(a.labels[0],a.data) if a.ndim==1 else b*a.data
    elif b.ndim<=1:
        return a.dotarray(b.labels[0],b.data) if b.ndim==1 else a*b.data
    else:
        for l1,l2 in zip(AC,BC):
            if l1.flow+l2.flow!=0: raise ValueError('tensor contraction error: not converged flow for %s and %s'%(repr(l1),repr(l2)))
        data={}
        ad,bd,axes=a.to_dict(AC),b.to_dict(BC),([a.axis(label) for label in common],[b.axis(label) for label in common]) if len(common)>0 else 0
        for qn in it.ifilter(ad.has_key,bd):
            for (qns1,block1),(qns2,block2) in it.product(ad[qn],bd[qn]):
                qns=tuple(it.chain(qns1,qns2))
                if qns not in data: data[qns]=0
                data[qns]+=np.tensordot(block1,block2,axes=axes)
        return STensor(data,labels=AL+BL)
