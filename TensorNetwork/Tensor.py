'''
======
Tensor
======

Labled multi-dimensional tensors, including:
    * classes: Tensor
    * functions: contract
'''

import numpy as np
import itertools as it
import scipy.linalg as sl
import HamiltonianPy.Misc as hm
from collections import Counter
from copy import copy,deepcopy
from HamiltonianPy import Label,QuantumNumbers

__all__=['Tensor','contract']

class Tensor(np.ndarray):
    '''
    Tensor class with labeled axes.

    Attributes
    ----------
    labels : list of Label
        The labels of the axes.
    '''
    autocheck=False

    def __new__(cls,data,labels):
        '''
        Initialize an instance through the explicit construction, i.e. constructor.

        Parameters
        ----------
        data : ndarray like
            The data of the Tensor.
        labels : list of Label
            The labels of the Tensor.
        '''
        if data is None:
            result=np.array(None).view(cls)
            result.labels=labels
        else:
            data=np.asarray(data)
            assert len(labels)==data.ndim
            for label,dim in zip(labels,data.shape):
                assert isinstance(label,Label)
                if label.qns is None:
                    label.qns=dim
                else:
                    assert label.dim==dim
            result=data.view(cls)
            result.labels=labels
        if Tensor.autocheck: assert result.dimcheck()
        return result

    def __array_finalize__(self,obj):
        '''
        Initialize an instance through both explicit and implicit constructions, i.e. construtor, view and slice.
        '''
        if obj is None:
            return
        else:
            self.labels=getattr(obj,'labels',None)

    def __reduce__(self):
        '''
        numpy.ndarray uses __reduce__ to pickle. Therefore this mehtod needs overriding for subclasses.
        '''
        temp=super(Tensor,self).__reduce__()
        return temp[0],temp[1],temp[2]+(self.labels,)

    def __setstate__(self,state):
        '''
        Set the state of the Tensor for pickle and copy.
        '''
        self.labels=state[-1]
        super(Tensor,self).__setstate__(state[0:-1])

    def __deepcopy__(self,demo):
        '''
        Deepcopy of the Tensor.
        '''
        return Tensor(data=super(Tensor,self).__deepcopy__(demo),labels=deepcopy(self.labels))

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        if self.ndim>1:
            return "%s(\nlabels=%s,\ndata=\n%s\n)"%(self.__class__.__name__,self.labels,np.asarray(self))
        else:
            return "%s(\nlabels=%s,\ndata=%s\n)"%(self.__class__.__name__,self.labels,np.asarray(self))

    def dimcheck(self):
        '''
        Check whether or not the dimensions of the labels and the data match each other.

        Returns
        --------
        logical
            True for match and False for not.
        '''
        return all([label.dim==dim for label,dim in zip(self.labels,self.shape)])

    @property
    def qnon(self):
        '''
        True for the labels using good quantum numbers otherwise False.
        '''
        return next(iter(self.labels)).qnon

    def label(self,axis):
        '''
        Return the corresponding label of an axis.

        Parameters
        ----------
        axis : integer
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
        integer
            The corresponding axis.
        '''
        return self.labels.index(label)

    def relabel(self,news,olds=None):
        '''
        Change the labels of the tensor.

        Parameters
        ----------
        news : list of Label
            The new labels of the tensor.
        olds : list of Label/integer, optional
            The old labels/axes of the tensor.
        '''
        if olds is None:
            assert len(news)==self.ndim
            self.labels=news
        else:
            assert len(news)==len(olds)
            for old,new in zip(olds,news):
                self.labels[self.axis(old) if isinstance(old,Label) else old]=new
        if Tensor.autocheck: assert self.dimcheck()

    def transpose(self,axes=None):
        '''
        Change the order of the tensor's axes and return the new tensor.

        Parameters
        ----------
        axes : list of Label/integer, optional
            The permutation of the original labels/axes.

        Returns
        -------
        Tensor
            The new tensor with the reordered axes.
        '''
        if axes is None:
            return self.transpose(axes=range(self.ndim-1,-1,-1))
        else:
            axes=[self.axis(axis) if isinstance(axis,Label) else axis for axis in axes]
            labels=[self.labels[axis] for axis in axes]
            return Tensor(np.asarray(self).transpose(axes),labels=labels)

    def take(self,indices,axis):
        '''
        Take elements from a tensor along an axis.

        Parameters
        ----------
        indices : integer / list of integers
            The indices of the values to extract.
        axis : Label/integer
            The label-of-the-axis/axis along which to take values.

        Returns
        -------
        Tensor
            The extracted parts of the tensor.
        '''
        label=axis if isinstance(axis,Label) else self.labels[axis]
        if type(indices) in (int,long):
            labels=[deepcopy(lb) for lb in self.labels if lb!=label]
        else:
            labels=deepcopy(self.labels)
        return Tensor(np.asarray(self).take(indices,axis=self.axis(label)),labels=labels)

    def copy(self,copy_data=False):
        '''
        Make a copy of the tensor.

        Parameters
        ----------
        copy_data : logical, optional
            True for copy the data of tensor, False for not.

        Returns
        -------
        Tensor
            The copy of the tensor.
        '''
        if copy_data:
            return Tensor(copy(np.asarray(self)),labels=copy(self.labels))
        else:
            return Tensor(self,labels=copy(self.labels))

    def reorder(self,*args):
        '''
        Reorder a dimension of a tensor with a permutation and optionally set a new qns for this dimension.

        Usage: ``tensor.reorder((axis,permutation,<qns>),(axis,permutation,<qns>),...)``
            * axis: integer/Label
                The axis of the dimension to be reordered.
            * permutation: 1d ndarray of integers
                The permutation array.
            * qns: QuantumNumbers, optional
                The new qantum number collection of the dimension if good quantum numbers are used.

        Returns
        -------
        Tensor
            The reordered tensor.
        '''
        result=self.copy(copy_data=False)
        for arg in args:
            assert len(arg) in (2,3)
            axis,permutation=arg[0],arg[1]
            if permutation is not None:
                axis=self.axis(axis) if isinstance(axis,Label) else axis
                if len(arg)==3:
                    result.labels[axis]=result.labels[axis].replace(qns=arg[2])
                else:
                    result.labels[axis]=result.labels[axis].replace(qns=result.labels[axis].qns.reorder(permutation) if self.qnon else len(permutation))
                result=hm.reorder(result,axes=[axis],permutation=permutation)
        return result

    @staticmethod
    def directsum(tensors,labels,axes=[]):
        '''
        The directsum of a couple of tensors.

        Parameters
        ----------
        tensors : list of Tensor
            The tensors to be directsummed.
        labels : list of Label
            The labels of the directsum.
        axes : list of integer, optional
                The axes along which the directsum is block diagonal.

        Returns
        -------
        Tensor
            The directsum of the tensors.
        '''
        for i,tensor in enumerate(tensors):
            if i==0:
                assert tensor.ndim>len(axes)
                ndim,qnon,shps=tensor.ndim,tensor.qnon,[tensor.shape[axis] for axis in axes]
                alters,shape,dtypes=set(xrange(ndim))-set(axes),list(tensor.shape),[tensor.dtype]
            else:
                assert tensor.ndim==ndim and tensor.qnon==qnon and [tensor.shape[axis] for axis in axes]==shps
                for alter in alters: shape[alter]+=tensor.shape[alter]
                dtypes.append(tensor.dtype)
        data=np.zeros(tuple(shape),dtype=np.find_common_type([],dtypes))
        for i,tensor in enumerate(tensors):
            if i==0:
                slices=[slice(0,tensor.shape[axis]) if axis in alters else slice(None,None,None) for axis in xrange(ndim)]
            else:
                for alter in alters:
                    slices[alter]=slice(slices[alter].stop,slices[alter].stop+tensor.shape[alter])
            data[tuple(slices)]=tensor[...]
        if qnon:
            for alter in alters:
                labels[alter].qns=QuantumNumbers.union([tensor.labels[alter].qns for tensor in tensors])
            for axis in axes:
                labels[axis].qns=next(iter(tensor)).labels[axis].qns
        return Tensor(data,labels=labels)

    def merge(self,*args):
        '''
        Merge some continous and accending labels of a tensor into a new one with an optional permutation.

        Usage: ``tensor.merge((olds,new,<permutation>),(olds,new,<permutation>),...)``
            * olds: list of Label/integer
                The old labels/axes to be merged.
            * new: Label
                The new label.
            * permutation: 1d ndarray of integers, optional
                The permutation of the quantum number collection of the new label.

        Returns
        -------
        Tensor
            The new tensor.
        '''
        masks,indptr=set(),range(self.ndim+1)
        for arg in args:
            assert len(arg) in (2,3)
            axes=np.array([self.axis(old) if isinstance(old,Label) else old for old in arg[0]])
            if len(axes)!=axes.max()-axes.min()+1 or not (axes[1:]>axes[:-1]).all():
                raise ValueError('Tensor merge error: the axes to be merged should be continous and accending, please call transpose first.')
            masks.add(axes[0])
            for axis in axes[1:]:
                indptr.remove(axis)
        count,shape,labels,axes,permutations=0,(),[],[],[]
        for axis,(start,stop) in enumerate(zip(indptr[:-1],indptr[1:])):
            if start in masks:
                shape+=(np.product([self.shape[i] for i in xrange(start,stop)]),)
                labels.append(args[count][1])
                axes.append(axis)
                permutations.append(args[count][2] if len(args[count])==3 else None)
                count+=1
            else:
                assert start+1==stop
                shape+=(self.shape[start],)
                labels.append(self.labels[start])
        data=np.asarray(self).reshape(shape)
        for axis,permutation in zip(axes,permutations):
            data=hm.reorder(data,axes=[axis],permutation=permutation)
        return Tensor(data,labels=labels)

    def split(self,*args):
        '''
        Split a label into small ones with an optional permutation.

        Usage: ``tensor.split((old,news,<permutation>),(old,news,<permutation>),...)``
            * old: Label/integer
                The label/axis to be splitted.
            * news: list of Label
                The new labels.
            * permutation: 1d ndarray of integers, optional
                The permutation of the quantum number collection of the old label.

        Returns
        -------
        Tensor
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
        data=np.asarray(self)
        for axis,permutation in zip(axes,permutations):
            data=hm.reorder(data,axes=[axis],permutation=permutation)
        return Tensor(data.reshape(shape),labels=labels)

    def qng(self,axes,qnses,signs=None,tol=None):
        '''
        Generate the quantum numbers of a tensor.

        Parameters
        ----------
        axes : list of Label/integer
            The labels/axes whose quantum numer collections are known.
        qnses : list of QuantumNumbers
            The quantum number collections of the known axes.
        signs : string, optional
            The signs of the quantum numbers to get the unkown one.
        tol : float64, optional
            The tolerance of the non-zeros.
        '''
        axes=[self.axis(axis) if isinstance(axis,Label) else axis for axis in axes]
        signs='+'*len(axes) if signs is None else signs
        assert len(axes)==len(qnses) and len(axes)==len(signs) and len(axes)==self.ndim-1
        type=next(iter(qnses)).type
        for axis,qns in zip(axes,qnses):
            self.labels[axis].qns=qns
            assert qns.type is type
        unkownaxis=next(iter(set(xrange(self.ndim))-set(axes)))
        expansions=[qns.expansion() for qns in qnses]
        contents=[None]*self.shape[unkownaxis]
        for index in sorted(np.argwhere(np.abs(np.asarray(self))>hm.TOL if tol is None else tol),key=lambda index: index[unkownaxis]):
            qn=type.regularization(sum([expansions[i][index[axis]] if sign=='+' else -expansions[i][index[axis]] for i,(axis,sign) in enumerate(zip(axes,signs))]))
            if contents[index[unkownaxis]] is None:
                contents[index[unkownaxis]]=qn
            else:
                assert (contents[index[unkownaxis]]==qn).all()
        self.labels[unkownaxis].qns=QuantumNumbers('G',(type,contents,np.arange(len(contents)+1)),protocal=QuantumNumbers.INDPTR)

    @staticmethod
    def random(shape,labels,signs=None,dtype=np.float64):
        '''
        Construt a random block-stuctured tensor.

        Parameters
        ----------
        shape : tuple of integer
            The shape of the random tensor.
        labels : list of Label
            The labels of the random tensor.
        signs : string, optional
            The signs of the quantum number collections of the labels if good quantum numbers are used.
        dtype : np.float32, np.float64, np.complex64, np.complex128, optional
            The data type of the random tensor.

        Returns
        -------
        Tensor
            A random block-stuctured tensor.
        '''
        np.random.seed()
        assert dtype in (np.float32,np.float64,np.complex64,np.complex128)
        result=Tensor(np.zeros(shape,dtype=dtype),labels=labels)
        if result.qnon:
            assert len(signs)==result.ndim
            paxes,plabels,maxes,mlabels=[],[],[],[]
            for i,(sign,label) in enumerate(zip(signs,labels)):
                if sign=='+':
                    paxes.append(i)
                    plabels.append(label)
                else:
                    maxes.append(i)
                    mlabels.append(label)
            pqns,ppermutation=QuantumNumbers.kron([label.qns for label in plabels]).sort(history=True)
            mqns,mpermutation=QuantumNumbers.kron([label.qns for label in mlabels]).sort(history=True)
            plabel,mlabel=Label('__TENSOR_RANDOM_+__',qns=pqns),Label('__TENSOR_RANDOM_-__',qns=mqns)
            result=result.transpose(axes=paxes+maxes).merge((plabels,plabel,ppermutation),(mlabels,mlabel,mpermutation))
            pod,mod=pqns.to_ordereddict(),mqns.to_ordereddict()
            for qn in it.ifilter(pod.has_key,mod):
                bshape=(pod[qn].stop-pod[qn].start,mod[qn].stop-mod[qn].start)
                if dtype in (np.float32,np.float64):
                    result[pod[qn],mod[qn]]=np.random.random(bshape)
                else:
                    result[pod[qn],mod[qn]]=np.random.random(bshape)+1j*np.random.random(bshape)
            result=result.split((plabel,plabels,np.argsort(ppermutation)),(mlabel,mlabels,np.argsort(mpermutation)))
        else:
            if dtype in (np.float32,np.float64):
                result[...]=np.random.random(shape)
            else:
                result[...]=np.random.random(shape)+1j*np.random.random(shape)
        return result

    def dotarray(self,axis,array):
        '''
        Multiply a certain axis of a tensor with an array.

        Parameters
        ----------
        axis : integer
            The axis of the tensor to be multiplied.
        array : 1d ndarray
            The multiplication array.

        Returns
        -------
        Tensor
            The new tensor.
        '''
        slices=[np.newaxis]*self.ndim
        slices[axis]=slice(None)
        return self*array[slices]

    def eigh(self,row,new,col,row_signs=None,col_signs=None,return_dagger=False):
        '''
        Eigenvalue decomposition of a tensor.

        Parameters
        ----------
        row/col : list of Label or Integer
            The axes or labels to be merged as the row/column dimension during the eigenvalue decomposition.
            The positive direction is IN for row and OUT for col if good quantum numbers are used.
        new : Label
            The label for the eigenvalues.
        row_signs/col_signs : string, optional
            The signs of the quantum number collections of the row/col labels if good quantum numbers are used.
        return_dagger : logical, optional
            True for returning the Hermitian conjugate of the eigenvalue matrix and False for not.

        Returns
        -------
        E,U : Tensor
            The eigenvalue decomposition of the tensor, such taht `self=U*E*U^{\daager}`.
        UD : Tensor, optional
            The Hermitian conjugate of the tensor `U`, with the column split in accordance with `self`.

        Notes
        -----
        The tensor to be decomposed must be Hermitian after the merge of its dimensions.
        '''
        assert len(row)+len(col)==self.ndim
        row=[r if isinstance(r,Label) else self.label(r) for r in row]
        col=[c if isinstance(c,Label) else self.label(c) for c in col]
        if self.qnon:
            row_qns,row_permutation=QuantumNumbers.kron([lb.qns for lb in row],row_signs).sort(history=True)
            col_qns,col_permutation=QuantumNumbers.kron([lb.qns for lb in col],col_signs).sort(history=True)
            assert len(row_qns)==len(col_qns)
            row_label=Label('__TENSOR_EIGH_ROW__',qns=row_qns)
            col_label=Label('__TENSOR_EIGH_COL__',qns=col_qns)
            m=np.asarray(self.merge((row,row_label,row_permutation),(col,col_label,col_permutation)))
            row_od,col_od=row_qns.to_ordereddict(),col_qns.to_ordereddict()
            E=np.zeros(len(row_qns),dtype=np.float64)
            U=np.zeros((len(row_qns),len(row_qns)),dtype=self.dtype)
            for qn in row_od:
                assert row_od[qn]==col_od[qn]
                e,u=sl.eigh(m[row_od[qn],row_od[qn]],check_finite=False)
                E[row_od[qn]]=e
                U[row_od[qn],row_od[qn]]=u
            new=new.replace(qns=row_qns)
            E=Tensor(E,labels=[new])
            U=Tensor(U,labels=[row_label,new]).split((row_label,row,np.argsort(row_permutation)))
            if return_dagger:
                UD=Tensor(hm.dagger(np.asarray(U)),labels=[new,col_label]).split((col_label,col,np.argsort(col_permutation)))
                return E,U,UD
            else:
                return E,U
        else:
            row_label=Label('__TENSOR_EIGH_ROW__',qns=np.product([label.dim for label in row]))
            col_label=Label('__TENSOR_EIGH_COL__',qns=np.product([label.dim for label in col]))
            assert len(row_qns)==len(col_qns)
            e,u=sl.eigh(np.asarray(self.merge((row,row_label),(col,col_label))),check_finite=False)
            new=new.replace(qns=len(row_label))
            E=Tensor(e,labels=[new])
            U=Tensor(u,labels=[row_label,new]).split((row_label,row))
            if return_dagger:
                UD=Tensor(hm.dagger(np.asarray(U)),labels=[new,col_label]).split((col_label,col))
                return E,U,UD
            else:
                return E,U

    def partitioned_svd(self,L,new,R,nmax=None,tol=None,return_truncation_err=False,**karg):
        '''
        Partition a 1d-tensor according to L and R and then perform the Schmitt decomposition.

        Parameters
        ----------
        L/R : Label
            The left/right part of the partition.
        new : Label
            The label for the singular values.
        nmax,tol,return_truncation_err :
            Please refer to HamiltonianPy.Misc.Linalg.truncated_svd for details.

        Returns
        -------
        U,S,V : Tensor
            The Schmitt decomposition of the 1d tensor.
        err : float64, optional
            The truncation error.
        '''
        assert self.ndim==1
        if self.qnon:
            data,qns=np.asarray(self),self.labels[0].qns
            assert qns.num==1 and sl.norm(qns.contents)<10**-6
            lod=L.qns.to_ordereddict()
            rod=R.qns.to_ordereddict()
            us,ss,vs,qns,count=[],[],[],[],0
            for qn in it.ifilter(lod.has_key,rod):
                s1,s2=lod[qn],rod[qn]
                n1,n2=s1.stop-s1.start,s2.stop-s2.start
                u,s,v=sl.svd(data[count:count+n1*n2].reshape((n1,n2)),full_matrices=False,lapack_driver='gesvd')
                us.append(u)
                ss.append(s)
                vs.append(v)
                qns.append(qn)
                count+=n1*n2
            temp=np.sort(np.concatenate([-s for s in ss]))
            nmax=len(temp) if nmax is None else min(nmax,len(temp))
            tol=temp[nmax-1] if tol is None else min(-tol,temp[nmax-1])
            Us,Ss,Vs,QNS,counts=[],[],[],[],[]
            for u,s,v,qn in zip(us,ss,vs,qns):
                cut=np.searchsorted(-s,tol,side='right')
                if cut>0:
                    Us.append(u[:,0:cut])
                    Ss.append(s[0:cut])
                    Vs.append(v[0:cut,:])
                    QNS.append(qn)
                    counts.append(cut)
            new=new.replace(qns=QuantumNumbers('U',(L.qns.type,QNS,counts),QuantumNumbers.COUNTS))
            nod=new.qns.to_ordereddict()
            U=np.zeros((L.dim,new.dim),dtype=self.dtype)
            S=np.concatenate(Ss)
            V=np.zeros((new.dim,R.dim),dtype=self.dtype)
            for u,v,qn in zip(Us,Vs,nod):
                U[lod[qn],nod[qn]]=u
                V[nod[qn],rod[qn]]=v
            U=Tensor(U,labels=[L,new])
            S=Tensor(S,labels=[new])
            V=Tensor(V,labels=[new,R])
            if return_truncation_err:
                err=(temp[nmax:]**2).sum()
                return U,S,V,err
            else:
                return U,S,V
        else:
            data=hm.truncated_svd(np.asarray(self).reshape((L.dim,R.dim)),full_matrices=False,nmax=nmax,tol=tol,return_truncation_err=return_truncation_err)
            new=new.replace(qns=len(data[1]))
            U=Tensor(data[0],labels=[L,new])
            S=Tensor(data[1],labels=[new])
            V=Tensor(data[2],labels=[new,R])
            if return_truncation_err:
                err=data[3]
                return U,S,V,err
            else:
                return U,S,V

    def svd(self,row,new,col,row_signs=None,col_signs=None,nmax=None,tol=None,return_truncation_err=False,**karg):
        '''
        Perform the svd.

        Parameters
        ----------
        row/col : list of Label or integer
            The labels or axes to be merged as the row/column dimension during the svd.
            The positive direction is IN for row and OUT for col if good quantum numbers are used.
        new : Label
            The label for the singular values.
        row_signs/col_signs : string, optional
            The signs of the quantum number collections of the row/col labels if good quantum numbers are used.
        nmax,tol,return_truncation_err :
            Please refer to HamiltonianPy.Misc.Linalg.truncated_svd for details.

        Returns
        -------
        U,S,V : Tensor
            The result tensor.
        err : float64, optional
            The truncation error.
        '''
        assert len(row)+len(col)==self.ndim
        row=[r if isinstance(r,Label) else self.label(r) for r in row]
        col=[c if isinstance(c,Label) else self.label(c) for c in col]
        if self.qnon:
            row_qns,row_permutation=QuantumNumbers.kron([lb.qns for lb in row],row_signs).sort(history=True)
            col_qns,col_permutation=QuantumNumbers.kron([lb.qns for lb in col],col_signs).sort(history=True)
            row_label=Label('__TENSOR_SVD_ROW__',qns=row_qns)
            col_label=Label('__TENSOR_SVD_COL__',qns=col_qns)
            m=np.asarray(self.merge((row,row_label,row_permutation),(col,col_label,col_permutation)))
            row_od,col_od=row_qns.to_ordereddict(),col_qns.to_ordereddict()
            us,ss,vs,qns=[],[],[],[]
            for qn in it.ifilter(row_od.has_key,col_od):
                u,s,v=sl.svd(m[row_od[qn],col_od[qn]],full_matrices=False,lapack_driver='gesvd')
                us.append(u)
                ss.append(s)
                vs.append(v)
                qns.append(qn)
            temp=np.sort(np.concatenate([-s for s in ss]))
            nmax=len(temp) if nmax is None else min(nmax,len(temp))
            tol=temp[nmax-1] if tol is None else min(-tol,temp[nmax-1])
            Us,Ss,Vs,contents=[],[],[],(row_qns.type,[],[])
            for u,s,v,qn in zip(us,ss,vs,qns):
                cut=np.searchsorted(-s,tol,side='right')
                if cut>0:
                    Us.append(u[:,0:cut])
                    Ss.append(s[0:cut])
                    Vs.append(v[0:cut,:])
                    contents[1].append(qn)
                    contents[2].append(cut)
            S=np.concatenate(Ss)
            new=new.replace(qns=QuantumNumbers('U',contents,protocal=QuantumNumbers.COUNTS))
            od=new.qns.to_ordereddict()
            U=np.zeros((len(row_qns),len(new.qns)),dtype=self.dtype)
            V=np.zeros((len(new.qns),len(col_qns)),dtype=self.dtype)
            for u,v,qn in zip(Us,Vs,od):
                U[row_od[qn],od[qn]]=u
                V[od[qn],col_od[qn]]=v
            U=Tensor(U,labels=[row_label,new]).split((row_label,row,np.argsort(row_permutation)))
            S=Tensor(S,labels=[new])
            V=Tensor(V,labels=[new,col_label]).split((col_label,col,np.argsort(col_permutation)))
            if return_truncation_err:
                err=(temp[nmax:]**2).sum()
                return U,S,V,err
            else:
                return U,S,V
        else:
            row_label=Label('__TENSOR_SVD_ROW__',qns=np.product([label.dim for label in row]))
            col_label=Label('__TENSOR_SVD_COL__',qns=np.product([label.dim for label in col]))
            m=np.asarray(self.merge((row,row_label),(col,col_label)))
            temp=hm.truncated_svd(m,full_matrices=False,nmax=nmax,tol=tol,return_truncation_err=return_truncation_err,**karg)
            u,s,v=temp[0],temp[1],temp[2]
            new=new.replace(qns=len(s))
            U=Tensor(u,labels=[row_label,new]).split((row_label,row))
            S=Tensor(s,labels=[new])
            V=Tensor(v,labels=[new,col_label]).split((col_label,col))
            if return_truncation_err:
                err=temp[3]
                return U,S,V,err
            else:
                return U,S,V

    def expanded_svd(self,L,S,R,E,I,ls=None,rs=None,es=None,cut=0,nmax=None,tol=None):
        '''
        Expand a label of a tensor and perform a sequential svd.

        Parameters
        ----------
        L/R : list of Label or integer
            The labels or axes to be merged as the left/right dimension during the expanded svd.
            The positive direction is IN for L and OUT for R if good quantum numbers are used.
        S : Label/integer
            The label/axis to be expanded.
            The positive direction is IN for S if good quantum numbers are used.
        E : list of Label
            The expansion of the merge of S labels.
            The positive direction is IN for E if good quantum numbers are used.
        I : list of Label
            The labels of the newly generated internal legs during the expanded svd.
        ls/rs/es : string, optional
            The signs of the quantum number collections of the L/R/E labels if good quantum numbers are used.
        cut : integer, optional
            The labels in E whose sequences are less than cut will be tied with the u matrices of the svds from the left;
            The labels in E whose sequences are equal to or greater than cut will be tied with the v matrices of the svds from the right.
        namx : integer, optional
            The maximum number of singular values to be kept.
        tol : float64, optional
            The tolerance of the singular values.

        Returns
        -------
        list of Tensor
            The results of the expanded svd.
        '''
        assert len(L)+len(R)==self.ndim-1 and cut>=0 and cut<=len(E)
        L=[l if isinstance(l,Label) else self.label(l) for l in L]
        S=S if isinstance(S,Label) else self.label(S)
        R=[r if isinstance(r,Label) else self.label(r) for r in R]
        es='+'*len(E) if es is None else es
        llabel=Label('__TENSOR_EXPANDED_SVD_L__',qns=QuantumNumbers.kron([lb.qns for lb in L],signs=ls) if self.qnon else None)
        rlabel=Label('__TENSOR_EXPANDED_SVD_R__',qns=QuantumNumbers.kron([lb.qns for lb in R],signs=rs) if self.qnon else None)
        data=self.merge((L,llabel),(R,rlabel)).split((S,E))
        ms=[]
        if cut==len(E):
            assert len(E)==len(I)
            for i in xrange(cut):
                if i>0: data=contract([s,v],engine='einsum',reserve=s.labels)
                row_signs,col_signs='+'+es[i],''.join(('-' if sign=='+' else '+') for sign in es[i+1:])+'+'
                u,s,v=data.svd(row=data.labels[:2],new=I[i],col=data.labels[2:],row_signs=row_signs,col_signs=col_signs,nmax=nmax,tol=tol)
                ms.append(u)
            ms[+0]=ms[+0].split((llabel,L))
            v=v.split((rlabel,R))
            return ms,s,v
        elif cut==0:
            assert len(E)==len(I)
            for i in xrange(len(E)-1,-1,-1):
                if i<len(E)-1: data=contract([u,s],engine='einsum',reserve=s.labels)
                row_signs,col_signs='+'+es[cut:i],('-' if es[i]=='+' else '+')+'+'
                u,s,v=data.svd(row=data.labels[:-2],new=I[i],col=data.labels[-2:],row_signs=row_signs,col_signs=col_signs,nmax=nmax,tol=tol)
                ms.insert(0,v)
            u=u.split((llabel,L))
            ms[-1]=ms[-1].split((rlabel,R))
            return u,s,ms
        else:
            assert len(E)==len(I)+1
            for i in xrange(cut):
                if i>0: data=contract((s,v),engine='einsum',reserve=s.labels)
                new=I[i] if i<cut-1 else Label('__TENSOR_EXPANDED_SVD_LINNER__')
                row_signs,col_signs='+'+es[i],''.join(('-' if sign=='+' else '+') for sign in es[i+1:])+'+'
                u,s,v=data.svd(row=data.labels[:2],new=new,col=data.labels[2:],row_signs=row_signs,col_signs=col_signs,nmax=nmax,tol=tol)
                ms.append(u)
            ls,data=s,v
            for i in xrange(len(E)-1,cut-1,-1):
                if i<len(E)-1: data=contract((u,s),engine='einsum',reserve=s.labels)
                new=I[i-1] if i>cut else Label('__TENSOR_EXPANDED_SVD_RINNER__')
                row_signs,col_signs='+'+es[cut:i],('-' if es[i]=='+' else '+')+'+'
                u,s,v=data.svd(row=data.labels[:-2],new=new,col=data.labels[-2:],row_signs=row_signs,col_signs=col_signs,nmax=nmax,tol=tol)
                ms.insert(cut,v)
            data,rs=u,s
            u,s,v=contract([ls,data,rs],engine='einsum',reserve=ls.labels+rs.labels).svd(row=[0],new=I[cut-1],col=[1])
            ms[cut-1]=contract([ms[cut-1],u],engine='tensordot')
            ms[cut]=contract([v,ms[cut]],engine='tensordot')
            Lambda=s
            ms[+0]=ms[+0].split((llabel,L))
            ms[-1]=ms[-1].split((rlabel,R))
            return ms,Lambda

    def deparallelization(self,row,new,col,mode='R',zero=10**-8,tol=10**-6):
        '''
        Deparallelize a tensor.

        Parameters
        ----------
        row/col : list of Label or integer
            The labels or axes to be merged as the row/column dimension during the deparallelization.
        new : Label
            The label for the new axis after the deparallelization.
        mode : 'R', 'C', optional
            'R' for the deparallelization of the row dimension;
            'C' for the deparallelization of the col dimension.
        zero : np.float64, optional
            The absolute value to identity zero vectors.
        tol : np.float64, optional
            The relative tolerance for rows or columns that can be considered as paralleled.

        Returns
        -------
        M : Tensor
            The deparallelized tensor.
        T : Tensor
            The coefficient matrix that satisfies T*M==m('R') or M*T==m('C').
        '''
        assert len(row)+len(col)==self.ndim
        row=[r if isinstance(r,Label) else self.label(r) for r in row]
        col=[c if isinstance(c,Label) else self.label(c) for c in col]
        qnsgenerator=QuantumNumbers.kron if self.qnon else np.product
        rlabel=Label('__TENSOR_DEPARALLELIZATION_ROW__',qns=qnsgenerator([lb.qns for lb in row]))
        clabel=Label('__TENSOR_DEPARALLELIZATION_COL__',qns=qnsgenerator([lb.qns for lb in col]))
        data=np.asarray(self.merge((row,rlabel),(col,clabel)))
        m1,m2,indices=hm.deparallelization(data,mode=mode,zero=zero,tol=tol,return_indices=True)
        if mode=='R':
            new=new.replace(qns=rlabel.qns.reorder(permutation=indices,protocal='EXPANSION') if self.qnon else len(indices))
            T=Tensor(m1,labels=[rlabel,new]).split((rlabel,row))
            M=Tensor(m2,labels=[new,clabel]).split((clabel,col))
            return T,M
        else:
            new=new.replace(qns=clabel.qns.reorder(permutation=indices,protocal='EXPANSION') if self.qnon else len(indices))
            M=Tensor(m1,labels=[rlabel,new]).split((rlabel,row))
            T=Tensor(m2,labels=[new,clabel]).split((clabel,col))
            return M,T

def contract(tensors,engine='einsum',**karg):
    '''
    Contract a collection of tensors.

    Parameters
    ----------
    tensors : list of Tensor
        The tensors to be contracted.
    engine : 'tensordot', 'einsum' or 'block', optional
        The engine to implement the contract of tensors, 'tensordot' for np.tensordot, 'einsum' for np.einsum and 'block' for blockcontract.
    karg : dict, optional
        See tensordotcontract, einsumcontract or blockcontract for details.

    Returns
    -------
    Tensor
        The contracted tensor.
    '''
    assert len(tensors)>0
    if engine=='tensordot':
        return tensordotcontract(tensors,**karg)
    elif engine=='einsum':
        return einsumcontract(tensors,**karg)
    elif engine=='block':
        return blockcontract(tensors,**karg)
    else:
        raise ValueError('contract error: engine(%s) not supported.'%engine)

def tensordotcontract(tensors):
    '''
    Use np.tensordot to implement the contraction of tensors.

    Parameters
    ----------
    tensors : list of Tensor
        The tensors to be contracted.

    Returns
    -------
    Tensor
        The contracted tensor.
    '''
    def _contract_(a,b):
        common=set(a.labels)&set(b.labels)
        aaxes=[a.axis(label) for label in common]
        baxes=[b.axis(label) for label in common]
        labels=[label for label in it.chain(a.labels,b.labels) if label not in common]
        axes=(aaxes,baxes) if len(common)>0 else 0
        return Tensor(np.tensordot(a,b,axes=axes),labels=labels)
    result=tensors[0]
    for i in xrange(1,len(tensors)):
        result=_contract_(result,tensors[i])
    return result

def einsumcontract(tensors,sequential=False,reserve=None):
    '''
    Use np.einsum to implement the contraction of tensors.
    
    Parameters
    ----------
    tensors : list of Tensor
        The tensors to be contracted.
    sequential : logical, optional
        True for sequential contraction and False for single complete contraction..
    reserve : list of Label, optional
        The labels that are repeated but not summed over.
    
    Returns
    -------
    Tensor
        The contracted tensor.
    '''
    def _contract_(tensors,reserve=None):
        reserve=[] if reserve is None else reserve
        replace={key:key for key in reserve}
        keep={key:True for key in reserve}
        lists=[tensor.labels for tensor in tensors]
        alls=[replace.get(label,label) for labels in lists for label in labels]
        counts=Counter(alls)
        table={key:i+65 if i<26 else i+97 for i,key in enumerate(counts)}
        subscripts=[''.join(chr(table[label]) for label in labels) for labels in lists]
        contracted_labels=[label for label in alls if (counts[label]==1 or keep.pop(label,False))]
        contracted_subscript=''.join(chr(table[label]) for label in contracted_labels)
        return Tensor(np.einsum('%s->%s'%(','.join(subscripts),contracted_subscript),*tensors),labels=contracted_labels)
    if sequential:
        result=tensors[0]
        for i in xrange(1,len(tensors)):
            result=_contract_((result,tensors[i]),reserve=reserve)
        return result
    else:
        return _contract_(tensors,reserve=reserve)

def blockcontract(tensors,signs=None):
    '''
    Use the block information of the tensors to implement their contraction.

    Parameters
    ----------
    tensors : list of Tensor
        The tensors to be contracted.
    signs : list of string, optional
        The signs of the quantum numbers of the tensors' labels.
        '+' for IN and '-' for OUT.

    Returns
    -------
    Tensor
        The contracted tensor.
    '''
    def _contract_(A,B,ASIGNS,BSIGNS):
        common=set(A.labels)&set(B.labels)
        if len(common)==0 or not A.qnon or not B.qnon:
            aaxes=[A.axis(label) for label in common]
            baxes=[B.axis(label) for label in common]
            labels=[label for label in it.chain(A.labels,B.labels) if label not in common]
            axes=(aaxes,baxes) if len(common)>0 else 0
            return Tensor(np.tensordot(A,B,axes=axes),labels=labels),None
        else:
            laxes,aaxes,baxes,raxes=[],[],[],[]
            lqnses,aqnses,bqnses,rqnses=[],[],[],[]
            lsigns,asigns,bsigns,rsigns=[],[],[],[]
            lshape,ashape,bshape,rshape=[],[],[],[]
            labels,signs=[],[]
            for i,label in enumerate(A.labels):
                if label not in common:
                    laxes.append(i)
                    lqnses.append(label.qns)
                    lsigns.append(ASIGNS[i])
                    lshape.append(label.dim)
                    labels.append(label)
                    signs.append(ASIGNS[i])
                else:
                    aaxes.append(i)
                    aqnses.append(label.qns)
                    asigns.append('+' if ASIGNS[i]=='-' else '-')
                    ashape.append(label.dim)
            for i,label in enumerate(B.labels):
                if label in common:
                    baxes.append(i)
                    bqnses.append(label.qns)
                    bsigns.append(BSIGNS[i])
                    bshape.append(label.dim)
                else:
                    raxes.append(i)
                    rqnses.append(label.qns)
                    rsigns.append('+' if BSIGNS[i]=='-' else '-')
                    rshape.append(label.dim)
                    labels.append(label)
                    signs.append(BSIGNS[i])
            A=np.asarray(A).transpose(laxes+aaxes).reshape((np.product(lshape),np.product(ashape)))
            B=np.asarray(B).transpose(baxes+raxes).reshape((np.product(bshape),np.product(rshape)))
            lqns,lpermutation=QuantumNumbers.kron(lqnses,signs=lsigns).sort(history=True)
            aqns,apermutation=QuantumNumbers.kron(aqnses,signs=asigns).sort(history=True)
            bqns,bpermutation=QuantumNumbers.kron(bqnses,signs=bsigns).sort(history=True)
            rqns,rpermutation=QuantumNumbers.kron(rqnses,signs=rsigns).sort(history=True)
            lod,cod,rod=lqns.to_ordereddict(),aqns.to_ordereddict(),rqns.to_ordereddict()
            result=np.zeros((A.shape[0],B.shape[1]),dtype=np.find_common_type([A.dtype,B.dtype],[]))
            for qn in it.ifilter(lambda qn: True if lod.has_key(qn) and rod.has_key(qn) else False,cod):
                linds=lpermutation[lod[qn]]
                ainds=apermutation[cod[qn]]
                binds=bpermutation[cod[qn]]
                rinds=rpermutation[rod[qn]]
                result[linds[:,None],rinds]=A[linds[:,None],ainds].dot(B[binds[:,None],rinds])
            result=Tensor(result.reshape(lshape+rshape),labels=labels)
            return result,signs
    signs=['+'*tensor.ndim for tensor in tensors] if signs is None else signs
    assert len(signs)==len(tensors)
    result,SIGNS=tensors[0],signs[0]
    for i in xrange(1,len(tensors)):
        result,SIGNS=_contract_(A=result,B=tensors[i],ASIGNS=SIGNS,BSIGNS=signs[i])
    return result
