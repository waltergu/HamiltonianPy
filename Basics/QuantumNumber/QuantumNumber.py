'''
-------------------
Good quantum number
-------------------

Quantum number, including:
    * classes: QuantumNumber, QuantumNumbers
'''

from collections import OrderedDict
from copy import copy,deepcopy
from fpermutation import fpermutation
import numpy as np
import numpy.linalg as nl
import random

__all__=['QuantumNumber','QuantumNumbers']

class QuantumNumber(np.ndarray):
    '''
    Quantum number, which itself is a 1d ndarray with the elements being the values of the quantum number.

    Attributes
    ----------
    names : tuple of string
        The names of the quantum number.
    periods : tuple of integer
        The periods of the quantum number.
    '''
    names=()
    periods=()

    def __new__(cls,values):
        '''
        Constructor.

        Parameters
        ----------
        values : iterable of numbers
            The values of the quantum number.
        '''
        self=np.asarray(values).reshape(-1).view(cls)
        assert len(self)==len(cls.names)
        return self

    @classmethod
    def __set_names_and_periods__(cls,names,periods):
        '''
        Set the names and periods of the quantum number.

        Parameters
        ----------
        names : list of str
            The names of the quantum number.
        periods : list of None/posotive integer
            The periods of the quantum number.
        '''
        assert len(names)==len(periods)
        for name,period in zip(names,periods):
            assert isinstance(name,str)
            assert (period is None) or (type(period) in (long,int) and period>0)
        cls.names=tuple(names)
        cls.periods=tuple(periods)

    @classmethod
    def regularization(cls,array):
        '''
        Regularize the elements of an array so that it can represent quantum numbers.

        Parameters
        ----------
        array : 1d/2d ndarray
            * When 1d array, it is a potential representation of a quantum number;
            * When 2d array, it is a potential representation of a collection of quantum numbers with the rows being the individual ones.

        Returns
        -------
        1d/2d ndarray
            The input array after the regularization.
        '''
        assert array.shape[-1]==len(cls.periods)
        if array.ndim==1:
            for i,t in enumerate(cls.periods):
                if t is not None: array[i]%=t
        elif array.ndim==2:
            for i,t in enumerate(cls.periods):
                if t is not None: array[:,i]%=t
        else:
            raise ValueError('%s regularization error: array should be 1d or 2d.'%(cls.__name__))
        return array

    def __getattr__(self,key):
        '''
        Overloaded dot(.) operator.
        '''
        try:
            return self[self.__class__.names.index(key)]
        except ValueError:
            raise AttributeError("%s has no attribute '%s'."%(self.__class__.__name__,key))

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        return ''.join(['QN(',','.join(['%s']*len(self)),')'])%tuple(self)

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        temp=[]
        for name,value,period in zip(self.__class__.names,self,self.__class__.periods):
            temp.append(name)
            temp.append(value)
            temp.append('U1' if period is None else 'Z%s'%(period))
        return ''.join(['QN','(',','.join(['%s=%r(%s)']*len(self)),')'])%tuple(temp)

    @classmethod
    def zero(cls):
        '''
        Return a new quantum number with all the values equal to zero.
        '''
        return cls([0 for i in xrange(len(cls.names))])

    def __hash__(self):
        '''
        Return the hash value of the quantum number.
        '''
        return hash(tuple(self))

    def __neg__(self):
        '''
        Overloaded negative(-) operator.
        '''
        return self.__class__(self.__class__.regularization(-np.asarray(self)))

    def __add__(self,other):
        '''
        Overloaded left addition(+) operator, which supports the left addition by an instance of QuantumNumber/QuantumNumbers..
        '''
        if isinstance(other,QuantumNumbers):
            return other+self
        else:
            assert self.__class__ is other.__class__
            return self.__class__(self.__class__.regularization(np.asarray(self)+np.asarray(other)))

    __iadd__=__add__

    def __sub__(self,other):
        '''
        Overloaded subtraction(-) operator, which supports the subtraction of two quantum numbers.
        '''
        if isinstance(other,QuantumNumbers):
            return -other+self
        else:
            assert self.__class__ is other.__class__
            return self.__class__(self.__class__.regularization(np.asarray(self)-np.asarray(other)))

    __isub__=__sub__

    def __mul__(self):
        '''
        Overloaded multiplication(*) operator.
        '''
        raise NotImplementedError()

    __imul__=__mul__

    def __rmul__(self,other):
        '''
        Overloaded multiplication(*) operator.
        '''
        return self.__mul__(other)

    def __div__(self,other):
        '''
        Overloaded left division(/) operator.
        '''
        raise NotImplementedError()

    __idiv__=__div__

    def replace(self,**karg):
        '''
        Return a new QuantumNumber object with specified fields replaced with new values.
        '''
        result=self.__class__(map(karg.pop,self.__class__.names,self))
        if karg:
            raise ValueError('%s replace error: it got unexpected field names: %r'%(self.__class__.__name__,karg.keys()))
        return result

    @classmethod
    def directsum(cls,self,other):
        '''
        The directsum of two quantum numbers.

        Parameters
        ----------
        cls : class
            The class of the result.
        self,other : QuantumNumber
            The quantum numbers to be direct summed.

        Returns
        -------
        cls
            The new quantum number.
        '''
        assert cls.names==self.__class__.names+other.__class__.names
        assert cls.periods==self.__class__.periods+other.__class__.periods
        return cls(np.concatenate([np.asarray(self),np.asarray(other)]))

class QuantumNumbers(object):
    '''
    A collection of quantum numbers in a stroage format similiar to that of compressed-sparse-row vectors.

    Attributes
    ----------
    form : 'G', 'U' or 'C'
        * 'G': general form
            No restriction for the contents of the collection
        * 'U': unitary form
            The contents of the collection have no duplicates with respect to the rows.
        * 'C': canonical form
            The contents of the collection must be arranged in a accending order with no duplicates by the rows.
    type : class
        The class of the quantum numbers contained in the collection.
    contents : 2d ndarray
        The ndarray representation of the collection with the rows representing the set of its quantum numbers.
    indptr : 1d ndarray of integers
        The index pointer array of the set of the quantum numbers.
    '''
    COUNTS,INDPTR=0,1

    def __init__(self,form,data,protocal=COUNTS):
        '''
        Constructor, supporting the following usages:
            * ``QuantumNumbers(form,(qns,counts),QuantumNumbers.COUNTS)``, with
                form: 'G'/'g', 'U'/'u' or 'C'/'c'
                    'G'/'g' for general form, 'U'/'u' for unitary form and 'C'/'c' for canonical form.
                qns: list of QuantumNumber
                    The quantum numbers contained in the collection.
                counts: list of integer
                    The counts of the duplicates of the quantum numbers.
            * ``QuantumNumbers(form,(qns,indptr),QuantumNumbers.INDPTR)``, with
                form: 'G'/'g', 'U'/'u' or 'C'/'c'
                    'G'/'g' for general form, 'U'/'u' for unitary form and 'C'/'c' for canonical form.
                qns: list of QuantumNumber
                    The quantum numbers contained in the collection.
                indptr: list of integer
                    The indptr of the collection.
            * ``QuantumNumbers(form,(type,contents,counts),QuantumNumbers.COUNTS)``, with
                form: 'G'/'g', 'U'/'u' or 'C'/'c'
                    'G'/'g' for general form, 'U'/'u' for unitary form and 'C'/'c' for canonical form.
                type: class
                    The class of the quantum numbers contained in the collection.
                contents: 2d ndarray
                    The contents of the collection.
                counts: list of integer
                    The counts of the duplicates of the quantum numbers.
            * ``QuantumNumbers(form,(type,contents,indptr),QuantumNumbers.INDPTR)``, with
                form: 'G'/'g', 'U'/'u' or 'C'/'c'
                    'G'/'g' for general form, 'U'/'u' for unitary form and 'C'/'c' for canonical form.
                type: class
                    The class of the quantum numbers contained in the collection.
                contents: 2d ndarray
                    The contents of the collection.
                indptr: list of integer
                    The indptr of the collection.
        '''
        assert form in ('G','g','U','u','C','c') and len(data) in (2,3) and protocal in (QuantumNumbers.COUNTS,QuantumNumbers.INDPTR)
        self.form=form.upper()
        if protocal==QuantumNumbers.COUNTS:
            if len(data)==2:
                self.type=next(iter(data[0])).__class__
                self.contents=np.asarray(data[0])
                counts=data[1]
            else:
                self.type=data[0]
                self.contents=np.asarray(data[1])
                counts=data[2]
            assert np.all(counts>=0)
            self.indptr=np.concatenate(([0],np.cumsum(counts)))
        else:
            if len(data)==2:
                self.type=next(iter(data[0])).__class__
                self.contents=np.asarray(data[0])
                self.indptr=np.asarray(data[1])
            else:
                self.type=data[0]
                self.contents=np.asarray(data[1])
                self.indptr=np.asarray(data[2])
        assert self.contents.ndim==2 and self.indptr.ndim==1

    @staticmethod
    def mono(qn,count=1):
        '''
        Construct a collection composed of only one quantum number.

        Parameters
        ----------
        qn : QuantumNumber
            The solitary quantum number the collection contains.
        count : positive integer, optional
            The count of the duplicates of the quantum number.

        Returns
        -------
        QuantumNumbers
            The constructed collection.
        '''
        return QuantumNumbers('C',((qn,),[0,count]),protocal=QuantumNumbers.INDPTR)

    def sort(self,history=False):
        '''
        Sort the contents of the collection and convert it to the canonical form.

        Parameters
        ----------
        history : logical, optional
            When True, the permutation to make the quantum numbers in order will be recorded.

        Returns
        -------
        self : QuantumNumbers
            The collection after the sort.
        permutation : 1d ndarray of integer, optional
            The permutation array to sort the collection.
        '''
        self.form='C'
        contents,indptr,counts=self.contents,self.indptr,self.indptr[1:]-self.indptr[:-1]
        permutation=np.lexsort(contents.T[::-1])
        mask=np.concatenate(([True],np.any(contents[permutation[1:]]!=contents[permutation[:-1]],axis=1)))
        self.contents=contents[permutation][mask]
        counts=counts[permutation]
        indices=np.concatenate((np.argwhere(mask).reshape((-1)),[len(mask)]))
        self.indptr=np.zeros(self.contents.shape[0]+1,dtype=np.int64)
        for i in xrange(len(indices)-1):
            self.indptr[i+1]=self.indptr[i]+counts[indices[i]:indices[i+1]].sum()
        if history:
            return self,fpermutation(indptr[:-1][permutation],counts,len(self))
        else:
            return self

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        return 'QNS(%s,%s)'%(self.num,len(self))

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return '\n'.join(['QNS(','\n'.join(['%s:(%s:%s)'%(qn,self.indptr[i],self.indptr[i+1]) for i,qn in enumerate(self)]),')'])

    def __len__(self):
        '''
        The total number of quantum numbers when duplicates are counted duplicately.
        '''
        return self.indptr[-1]

    @property
    def num(self):
        '''
        The number of unduplicate quantum numbers of the collection.
        '''
        return self.contents.shape[0]

    def __iter__(self):
        '''
        Return an iterator over the quantum number it contains.
        '''
        for values in self.contents:
            yield self.type(values)

    def __getitem__(self,n):
        '''
        Overloaded method ``self[n]``, where `n` should be a integer.
        '''
        index=len(self)+n if n<0 else n
        if index<0 or index>=len(self): raise IndexError('QuantumNumbers index(%s) out of range.'%n)
        return self.type(self.contents[np.searchsorted(self.indptr,index,side='right')-1])

    def indices(self,qn):
        '''
        Find the indices of a quantum number.

        Parameters
        ----------
        qn : QuantumNumber
            The quantum number whose index is inquired.

        Returns
        -------
        1d ndarray of integers
            The corresponding indices of the quantum number.
        '''
        if self.form=='C':
            L,R,pos=0,self.num,self.num/2
            qn,current=tuple(qn),tuple(self.contents[pos])
            while current!=qn:
                if current>qn:
                    R=pos
                else:
                    L=pos
                pos,last=(L+R)/2,pos
                if pos==last:
                    raise ValueError('QuantumNumbers index error: %s not in QuantumNumbers.'%(qn))
                current=tuple(self.contents[pos])
            else:
                return [pos]
        else:
            return np.argwhere(np.all(self.contents==np.asarray(qn),axis=1)).reshape((-1))

    def __contains__(self,qn):
        '''
        Judge whether a quantum number is contained in the collection.
        '''
        try:
            self.indices(qn)
            return True
        except ValueError:
            return False

    def __pos__(self):
        '''
        Overloaed positive(+) operator.
        '''
        return self

    def __neg__(self):
        '''
        Overloaded negative(-) operator.
        '''
        return QuantumNumbers(
                form=       'U' if self.form=='C' else self.form,
                data=       (self.type,self.type.regularization(-self.contents),self.indptr),
                protocal=   QuantumNumbers.INDPTR
                )

    def __add__(self,other):
        '''
        Overloaded left addition(+) operator, which supports the left addition by an instance of QuantumNumber.
        '''
        assert self.type is other.__class__ or not issubclass(other.__class__,QuantumNumber)
        return QuantumNumbers(
                form=       'U' if self.form=='C' else self.form,
                data=       (self.type,self.type.regularization(self.contents+np.asarray(other)[np.newaxis,:]),self.indptr),
                protocal=   QuantumNumbers.INDPTR
                )

    __iadd__=__add__

    def __radd__(self,other):
        '''
        Overloaded right addition(+) operator, which supports the right addition by an instance of QuantumNumber.
        '''
        return self+other

    def __sub__(self,other):
        '''
        Overloaded subtraction(-) operator.
        '''
        assert self.type is other.__class__ or not issubclass(other.__class__,QuantumNumber)
        return QuantumNumbers(
                form=       'U' if self.form=='C' else self.form,
                data=       (self.type,self.type.regularization(self.contents-np.asarray(other)[np.newaxis,:]),self.indptr),
                protocal=   QuantumNumbers.INDPTR
                )

    __isub__=__sub__

    def __mul__(self,other):
        '''
        Overloaded left multiplication(*) operator.
        '''
        assert type(other) in (int,long)
        return QuantumNumbers.kron([self]*other)

    __imul__=__mul__

    def __rmul__(self,other):
        '''
        Overloaded right multiplication(*) operator.
        '''
        return self*other

    def __eq__(self,other):
        '''
        Overloaded equivalent(==) operator.
        '''
        return self.type is other.type and nl.norm(self.contents-other.contents)==0 and nl.norm(self.indptr-other.indptr)==0

    def __ne__(self,other):
        '''
        Overloaded not-equivalent(!=) operator.
        '''
        return not self==other

    def subset(self,targets=()):
        '''
        A subsets of the collection of the quantum numbers.

        Parameters
        ----------
        targets : list of QuantumNumber, optional
            The quantum numbers of the subset.

        Returns
        -------
        QuantumNumbers
            The subset.
        '''
        indices=np.concatenate([self.indices(target) for target in targets])
        return QuantumNumbers(
                    form=       'G' if self.form=='G' else 'U',
                    data=       (self.type,self.contents[indices],self.indptr[indices+1]-self.indptr[indices]),
                    protocal=   QuantumNumbers.COUNTS
                    )

    def subslice(self,targets=()):
        '''
        The subslice with the corresponding quantum numbers in targets.

        Parameters
        ----------
        targets : list of QuantumNumber, optional
            The quantum numbers whose slices wanted to be extracted.

        Returns
        -------
        1d ndarray of integer
            The subslice.
        '''
        indices=np.concatenate([self.indices(target) for target in targets])
        return np.concatenate([xrange(self.indptr[index],self.indptr[index+1]) for index in indices])

    def expansion(self,targets=None):
        '''
        The expansion of the quantum number collection.

        Parameters
        ----------
        targets : list of QuantumNumber, optional
            The quantum numbers to be expanded.

        Returns
        -------
        2d ndarray
            The ndarray representation of the expanded quantum numbers.
        '''
        if targets is None:
            return np.repeat(self.contents,self.indptr[1:]-self.indptr[:-1],axis=0)
        else:
            indices=np.concatenate([self.indices(target) for target in targets])
            return np.repeat(self.contents[indices],self.indptr[indices+1]-self.indptr[indices],axis=0)

    @staticmethod
    def union(args,signs=None):
        '''
        The union of several quantum number collections.

        Parameters
        ----------
        args : list of QuantumNumbers
            The collections of quantum numbers to be unioned.
        signs : string with each element being '+' or '-'
            The signs for the collections of the quantum numbers.

        Returns
        -------
        QuantumNumbers
            The union of the input collections.
        '''
        signs=len(args)*'+' if signs is None else signs
        assert len(signs)==len(args)
        cumsums=np.concatenate(([0],np.cumsum([len(arg) for arg in args])))
        contents=np.concatenate([arg.contents if sign=='+' else -arg.contents for arg,sign in zip(args,signs)])
        indptr=np.concatenate([arg.indptr[:-1]+cumsum for arg,cumsum in zip(args,cumsums[:-1])]+[(cumsums[-1],)])
        return QuantumNumbers('G',(next(iter(args)).type,contents,indptr),protocal=QuantumNumbers.INDPTR)

    @staticmethod
    def kron(args,signs=None):
        '''
        Tensor dot of several quantum number collections.

        Parameters
        ----------
        args : list of QuantumNumbers
            The collections of quantum numbers to be tensor dotted.
        signs : string with each element being '+' or '-'
            The signs for the collections of the quantum numbers.

        Returns
        -------
        QuantumNumbers
            The tensor dot of the input collections.
        '''
        signs=len(args)*'+' if signs is None else signs
        assert len(signs)==len(args)
        contents=np.zeros((1))
        for i,(qns,sign) in enumerate(zip(args,signs)):
            if i==len(args)-1:
                type=qns.type
                temp=qns.contents
                counts=np.tile(qns.indptr[1:]-qns.indptr[:-1],len(contents))
            else:
                temp=qns.expansion() 
            if sign=='+':
                contents=(contents[:,np.newaxis,...]+temp[np.newaxis,:,...]).reshape((len(contents)*len(temp),-1))
            else:
                contents=(contents[:,np.newaxis,...]-temp[np.newaxis,:,...]).reshape((len(contents)*len(temp),-1))
        assert all([qns.type is type for qns in args])
        return QuantumNumbers('G',(type,type.regularization(contents),counts),protocal=QuantumNumbers.COUNTS)

    def to_ordereddict(self,protocal=INDPTR):
        '''
        Convert a canonical collection of quantum numbers to an OrderedDict.

        Parameters
        ----------
        protocal : QuantumNumbers.INDPTR, QuantumNumbers.COUNTS, optional
            * When QuantumNumbers.INDPTR, the values of the result are the slices of the quantum numbers in the collection.
            * When QuantumNumbers.COUNTS, the values of the result are the counts of the duplicates of the quantum numbers.

        Returns
        -------
        OrderedDict
            The converted OrderedDict.
        '''
        assert protocal in (QuantumNumbers.INDPTR,QuantumNumbers.COUNTS) and self.form in ('U','C')
        result=OrderedDict()
        if protocal==QuantumNumbers.INDPTR:
            for i,qn in enumerate(self.contents):
                result[tuple(qn)]=slice(self.indptr[i],self.indptr[i+1],None)
        else:
            for i,qn in enumerate(self.contents):
                result[tuple(qn)]=self.indptr[i+1]-self.indptr[i]
        return result

    @staticmethod
    def from_ordereddict(type,ordereddict,protocal=INDPTR):
        '''
        Convert an ordered dict to a quantum number collection.

        Parameters
        ----------
        type : class
            The class of the quantum numbers.
        ordereddict : OrderedDict
            The OrderedDict that contains the contents of the collection.
        protocal : QuantumNumbers.INDPTR, QuantumNumbers.COUNTS, optional
            * When QuantumNumbers.INDPTR, the values of the ordereddict are the slices of the quantum numbers in the collection;
            * When QuantumNumbers.COUNTS, the values of the ordereddict are the counts of the duplicates of the quantum numbers.

        Returns
        -------
        QuantumNumbers
            The converted collection.
        '''
        assert protocal in (QuantumNumbers.COUNTS,QuantumNumbers.INDPTR)
        contents=np.asarray(ordereddict.keys())
        if protocal==QuantumNumbers.COUNTS:
            counts=ordereddict.values()
            return QuantumNumbers('U',(type,contents,counts),protocal=QuantumNumbers.COUNTS)
        else:
            indptr=np.concatenate(([0],[slice.stop for slice in ordereddict.itervalues()]))
            return QuantumNumbers('U',(type,contents,indptr),protocal=QuantumNumbers.INDPTR)

    def reorder(self,permutation,protocal='EXPANSION'):
        '''
        Reorder the quantum numbers of the collection and return the new one.

        Parameters
        ----------
        permutation : 1d ndarray
            The permutation array.
        protocal : 'EXPANSION', 'CONTENTS'
            * 'EXPANSION' for the reorder of the expansion of the collection;
            * 'CONTENTS' for the reorder of the contents of the collection.

        Returns
        -------
        QuantumNumbers
            The reordered collection.
        '''
        if permutation is None:
            return self
        else:
            assert protocal in ('EXPANSION','CONTENTS')
            if protocal=='EXPANSION':
                return QuantumNumbers(
                        form=       'G',
                        data=       (self.type,self.expansion()[permutation],range(len(permutation)+1)),
                        protocal=   QuantumNumbers.INDPTR
                        )
            else:
                return QuantumNumbers(
                        form=       'G' if self.form=='G' else 'U',
                        data=       (self.type,self.contents[permutation],(self.indptr[1:]-self.indptr[:-1])[permutation]),
                        protocal=   QuantumNumbers.COUNTS
                        )

    @staticmethod
    def decomposition(qnses,target,signs=None,method='exhaustion',nmax=None):
        '''
        Find the a couple of decompositions of target with respect to qnses.

        Parameters
        ----------
        qnses : list of QuantumNumbers
            The decomposition set of quantum number collections.
        target : QuantumNumber
            The target of the decomposition.
        signs : string, optional
            The signs of qnses when they are tensordotted.
        method : 'exhaustion' or 'monte carlo', optional
            The method used to find the decompositions.
        nmax : integer, optional
            The maximum number of decompositions.

        Returns
        -------
        list of tuple
            For each tuple, it satisfies the decomposition rule:
                * ``sum([qns.expansion()[index] if sign=='+' else -qns.expansion()[index] for qns,sign,index in zip(qnses,signs,tuple)])==target``
        '''
        assert method in ('exhaustion','monte carlo')
        result=set()
        if method== 'exhaustion':
            for pos in QuantumNumbers.kron(qnses,signs=signs).subslice(targets=(target,)):
                indices=[]
                for qns in reversed(qnses):
                    indices.append(pos%len(qns))
                    pos/=len(qns)
                result.add(tuple(reversed(indices)))
            if nmax is not None and nmax>len(result): 
                random.seed()
                result=random.sample(result,nmax)
        else:
            assert type(nmax) in (int,long)
            random.seed()
            target=np.array(target)
            qnses=[qns.expansion() if sign=='+' else -qns.expansion() for qns,sign in zip(qnses,'+'*len(qnses) if signs is None else signs)]
            diff=lambda indices: nl.norm(sum(qns[index] for qns,index in zip(qnses,indices))-target)
            old,new=np.array([random.randint(0,len(qns)-1) for qns in qnses]),np.zeros(len(qnses),dtype=np.int64)
            count,olddiff=0,diff(old)
            while 1:
                new[...]=old[...]
                pos=random.randint(0,len(qnses)-1)
                index=random.randint(0,len(qnses[pos])-2)
                new[pos]=index if index<old[pos] else index+1
                newdiff=diff(new)
                if newdiff<=olddiff or np.exp(olddiff-newdiff)>random.random():
                    old[...]=new[...]
                    olddiff=newdiff
                if newdiff==0:
                    count+=1
                    result.add(tuple(new))
                if len(result)>=nmax or count>nmax*5: break
        return list(result)
