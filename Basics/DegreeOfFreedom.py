'''
Degrees of freedom in a lattice, including:
1) classes: Table, Index, Internal, IDFConfig, Label, DegFreTree, IndexPack, IndexPackList
'''

__all__=['Table','Index','Internal','IDFConfig','Label','DegFreTree','IndexPack','IndexPackList']

import numpy as np
from Geometry import PID
from copy import copy,deepcopy
from collections import OrderedDict
from QuantumNumber import QuantumNumberCollection
from ..Math import Tree

class Table(dict):
    '''
    This class provides the methods to get an index from its sequence number or vice versa.
    '''

    def __init__(self,indices=[],key=None):
        '''
        Constructor.
        Parameters:
            indices: list of any hashable object
                The indices that need to be mapped to sequences.
            key: function, optional
                The function used to sort the indices.
            NOTE: The final order of the index in indices will be used as its sequence number.
        '''
        for i,v in enumerate(indices if key is None else sorted(indices,key=key)):
            self[v]=i

    @staticmethod
    def union(tables,key=None):
        '''
        This function returns the union of index-sequence tables.
        Parameters:
            tables: list of Table
                The tables to be unioned.
            key: callable, optional
                The function used to compare different indices in tables.
                When it is None, the sequence of an index will be naturally ordered by the its sequence in the input tables.
        Returns: Table
            The union of the input tables.
        '''
        result=Table()
        if key is None:
            sum=0
            for table in tables:
                if isinstance(table,Table):
                    count=0
                    for k,v in table.iteritems():
                        result[k]=v+sum
                        count+=1
                    sum+=count
        else:
            for table in tables:
                result.update(table)
            for i,k in enumerate(sorted(result,key=key)):
                result[k]=i
        return result

    def subset(self,select):
        '''
        This function returns a certain subset of an index-sequence table according to the select function.
        Parameters:
            select: callable
                A certain subset of table is extracted according to the return value of this function on the index in the table.
                When the return value is True, the index will be included and the sequence is naturally determined by its order in the mother table.
        Returns:
            The subset table.
        '''
        result=Table()
        for k,v in self.iteritems():
            if select(k):
                result[k]=v
        buff={}
        for i,k in enumerate(sorted([key for key in result.keys()],key=result.get)):
            buff[k]=i
        result.update(buff)
        return result

    @property
    def reversed_table(self):
        '''
        This function returns the sequence-index table for a reversed lookup.
        Returns: Table
            The reversed table whose key is the sequence and value the index.
        '''
        result=Table()
        for k,v in self.iteritems():
            result[v]=k
        return result

class Index(tuple):
    '''
    This class provides an index for a microscopic degree of freedom, including the spatial part and interanl part.
    '''

    def __new__(cls,pid,iid):
        '''
        Constructor.
        Parameters:
            pid: PID
                The point index, i.e. the spatial part in a lattice of the index
            iid: namedtuple
                The internal index, i.e. the internal part in a point of the index.
        '''
        self=super(Index,cls).__new__(cls,pid+iid)
        self.__dict__=OrderedDict()
        self.__dict__.update(pid._asdict())
        self.__dict__.update(iid._asdict())
        return self

    def __copy__(self):
        '''
        Copy.
        '''
        return self.replace(**self.__dict__)

    def __deepcopy__(self,memo):
        '''
        Deep copy.
        '''
        return self.replace(**self.__dict__)

    @property
    def pid(self):
        '''
        The pid part of the index.
        '''
        return PID(**{key:getattr(self,key) for key in PID._fields})

    def iid(self,cls):
        '''
        The iid part of the index.
        '''
        return cls(**{key:value for key,value in self.__dict__.items() if key not in PID._fields})

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        return ''.join(['Index(','=%r, '.join(self.__dict__.keys()),'=%r)'])%self

    def replace(self,**karg):
        '''
        Return a new Index object with specified fields replaced with new values.
        '''
        result=tuple.__new__(Index,map(karg.pop,self.__dict__.keys(),self))
        if karg:
            raise ValueError('Index replace error: it got unexpected field names: %r'%karg.keys())
        result.__dict__=OrderedDict()
        for key,value in zip(self.__dict__.keys(),result):
            result.__dict__[key]=value
        return result

    def mask(self,*arg):
        '''
        Mask some attributes of the index to None and return the new one.
        Parameters:
            arg: list of string
                The attributes to be masked.
        Returns: Index
            The masked index.
        '''
        return self.replace(**{key:None for key in arg})

    def select(self,*arg):
        '''
        Select some attributes of the index, mask the others to None and return the new one.
        Parameters:
            arg: list of string
                The attributes to be selected.
        Returns: Index
            The selected index.
        '''
        return self.mask(*[key for key in self.__dict__ if key not in arg])

    def to_tuple(self,priority):
        '''
        Convert an instance to tuple according to the parameter priority.
        Parameters:
            priority: list of string
                Every element of this list should correspond to a name of an attribute of self.
                The elements should have no duplicates and its length should be equal to the number of self's attributes.
        Returns: tuple
            The elements of the returned tuple are equal to the attributes of self. 
            Their orders are determined by the orders they appear in priority.
        '''
        if len(set(priority))<len(priority):
            raise ValueError('Index to_tuple error: the priority has duplicates.')
        if len(priority)!=len(self.__dict__):
            raise ValueError("Index to_tuple error: the priority doesn't cover all the attributes.")
        return tuple(map(self.__dict__.get,priority))

class Internal(object):
    '''
    This class is the base class for all internal degrees of freedom in a single point.
    '''

    def indices(self,pid,mask=[]):
        '''
        Return a list of all the masked indices within this internal degrees of freedom combined with an extra spatial part.
        Parameters:
            pid: PID
                The extra spatial part of the indices.
            mask: list of string, optional
                The attributes that will be masked to None.
        Returns: list of Index
            The indices.
        '''
        raise NotImplementedError("%s indices error: it is not implemented."%self.__class__.__name__)

class IDFConfig(dict):
    '''
    Configuration of the internal degrees of freedom in a lattice.
    For each of its (key,value) pairs,
        key: PID
            The pid of the lattice point where the interanl degrees of freedom live.
        value: subclasses of Internal
            The internal degrees of freedom on the corresponding point.
    Attributes:
        priority: list of string 
            The sequence priority of the allowed indices that can be defined on a lattice.
        map: function
            This function maps the pid of a lattice point to the interanl degrees of freedom on it.
    '''

    def __init__(self,priority,pids=[],map=None):
        '''
        Constructor.
        Parameters:
            priority: list of string
                The sequence priority of the allowed indices that can be defined on the lattice.
            pids: list of PID, optional
                The pids of the lattice points where the interanl degrees of freedom live.
            map: function, optional
                This function maps the pid of a lattice point to the interanl degrees of freedom on it.
        '''
        self.reset(priority=priority,pids=pids,map=map)

    def reset(self,priority=None,pids=[],map=None):
        '''
        Reset the idfconfig.
        Parameters:
            pids: list of PID, optional
                The pids of the lattice points where the interanl degrees of freedom live.
            map: function, optional
                This function maps the pid of a lattice point to the interanl degrees of freedom on it.
            priority: list of string, optional
                The sequence priority of the allowed indices that can be defined on the lattice.
        '''
        self.clear()
        if priority is not None: self.priority=priority
        if map is not None: self.map=map
        for pid in pids:
            self[pid]=self.map(pid)

    def __setitem__(self,key,value):
        '''
        Set the value of an item.
        Parameters:
            key: FID
                The pid of the lattice point where the internal degrees of freedom live.
            value: subclasses of Internal
                The internal degrees of freedom on the corresponding point.
        '''
        assert isinstance(key,PID)
        assert isinstance(value,Internal)
        dict.__setitem__(self,key,value)

    def table(self,mask=[]):
        '''
        Return a Table instance that contains all the allowed indices which can be defined on a lattice.
        '''
        return Table([index for key,value in self.items() for index in value.indices(key,mask)],key=lambda index: index.to_tuple(priority=self.priority))

class Label(tuple):
    '''
    The label of one dimension of a tensor.
    Attributes:
        immutable: ('identifier','_prime_')
            The names of the immutable part of the label.
        qnc: integer or QuantumNumberCollection
            When integer, it is the dimension of the label;
            When QuantumNumberCollection, it is the quantum number collection of the label.
    '''
    repr_form=1

    def __new__(cls,identifier,prime=False,qnc=None):
        '''
        Parameters:
            identifier: any hashable object
                The index of the label
            prime: logical, optional
                When True, the label is in the prime form;
                otherwise not.
            qnc: integer or QuantumNumberCollection, optional
                When integer, it is the dimension of the label;
                When QuantumNumberCollection, it is the quantum number collection of the label.
        '''
        self=tuple.__new__(cls,(identifier,prime))
        tuple.__setattr__(self,'immutable',('identifier','_prime_'))
        tuple.__setattr__(self,'qnc',qnc)
        return self

    def replace(self,**karg):
        '''
        Return a new label with some of its attributes parts replaced.
        Parameters:
            karg: dict in the form (key,value), with
                key: string
                    The attributes of the label
                value: any object
                    The corresponding value.
        Returns: Label
            The new label.
        '''
        result=tuple.__new__(self.__class__,[karg.pop(key,value) for key,value in zip(self.immutable,self)])
        for key,value in self.__dict__.iteritems():
            tuple.__setattr__(result,key,karg.pop(key,value))
        if karg:
            raise ValueError("Label replace error: %s are not the attributes of the label."%karg.keys())
        return result

    @classmethod
    def repr_qnc_on(cls,id=False):
        '''
        Turn on the qnc part in the repr, and optionally, the id of the qnc.
        '''
        if id:
            cls.repr_form=2
        else:
            cls.repr_form=1

    @classmethod
    def repr_qnc_off(cls):
        '''
        Turn off the qnc part in the repr.
        '''
        cls.repr_form=0

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        if self.repr_form==0:
            if self[-1]:
                return "Label%s%s"%((tuple.__repr__(self[0:-1])),"'")
            else:
                return "Label%s"%(tuple.__repr__(self[0:-1]))
        elif self.repr_form==1:
            if self[-1]:
                return "Label%s%s,with qnc=%s"%((tuple.__repr__(self[0:-1])),"'",self.qnc)
            else:
                return "Label%s,with qnc=%s"%(tuple.__repr__(self[0:-1]),self.qnc)
        else:
            if self[-1]:
                return "Label%s%s,with qnc(id=%s)=%s"%((tuple.__repr__(self[0:-1])),"'",id(self.qnc),self.qnc)
            else:
                return "Label%s,with qnc(id=%s)=%s"%(tuple.__repr__(self[0:-1]),id(self.qnc),self.qnc)

    def __getattr__(self,key):
        '''
        Overloaded operator(.).
        '''
        return self[self.immutable.index(key)]

    def __setattr__(self,key,value):
        '''
        The setter of the overloaded operator(.).
        '''
        if key not in self.immutable:
            tuple.__setattr__(self,key,value)
        else:
            raise TypeError("Label __setattr__ error: %s is immutable."%key)

    def __copy__(self):
        '''
        Copy.
        '''
        result=tuple.__new__(self.__class__,self)
        tuple.__setattr__(result,'__dict__',copy(self.__dict__))
        return result

    def __deepcopy__(self,memo):
        '''
        Deepcopy.
        '''
        result=tuple.__new__(self.__class__,self)
        tuple.__setattr__(result,'__dict__',deepcopy(self.__dict__))
        return result

    @property
    def prime(self):
        '''
        The prime of the label.
        '''
        temp=list(self)
        temp[-1]=not temp[-1]
        result=tuple.__new__(self.__class__,temp)
        for key,value in self.__dict__.iteritems():
            tuple.__setattr__(result,key,value)
        return result

class DegFreTree(Tree):
    '''
    The tree of the layered degrees of freedom.
    For each (node,data) pair of the tree,
        node: Index
            The selected index which can represent a couple of indices.
        data: integer of QuantumNumberCollection
            When an integer, it is the number of degrees of freedom that the index represents;
            When a QuantumNumberCollection, it is the quantum number collection that the index is associated with.
    Attributes:
        mode: 'QN' or 'NB'
            The mode of the DegFreTree.
        layers: list of tuples of string
            The tag of each layer of indices.
        priority: lsit of string
            The sequence priority of the allowed indices.
        map: function
            This function maps a leaf (bottom index) of the DegFreTree to its corresponding data.
        cache: dict
            The cache of the degfretree.
    '''

    def __init__(self,mode,layers,priority,leaves=[],map=None):
        '''
        Constructor.
        Parameters:
            mode: 'QN' or 'NB'
                The mode of the DegFreTree.
            layers: list of tuples of string
                The tag of each layer of indices.
            priority: lsit of string
                The sequence priority of the allowed indices.
            leaves: list of Index, optional
                The leaves (bottom indices) of the DegFreTree.
            map: function, optional
                This function maps a leaf (bottom index) of the DegFreTree to its corresponding data.
        '''
        self.reset(mode=mode,layers=layers,priority=priority,leaves=leaves,map=map)

    def reset(self,mode=None,layers=None,priority=None,leaves=[],map=None):
        '''
        Reset the DegFreTree.
        Parameters:
            mode: 'QN' or 'NB', optional
                The mode of the DegFreTree.
            layers: list of tuples of string, optional
                The tag of each layer of indices.
            priority: lsit of string, optional
                The sequence priority of the allowed indices.
            leaves: list of Index, optional
                The leaves (bottom indices) of the DegFreTree.
            map: function, optional
                This function maps a leaf (bottom index) of the DegFreTree to its corresponding data.
        '''
        self.clear()
        Tree.__init__(self)
        assert mode in (None,'QN','NB')
        if mode is not None: self.mode=mode
        if layers is not None: self.layers=layers
        if priority is not None: self.priority=priority
        if map is not None: self.map=map
        self.cache={}
        if len(leaves)>0:
            temp=[key for layer in self.layers for key in layer]
            assert set(range(len(PID._fields)))==set([temp.index(key) for key in PID._fields])
            temp=set(temp)
            self.add_leaf(parent=None,leaf=tuple([None]*len(leaves[0])),data=None)
            for layer in self.layers:
                temp-=set(layer)
                indices=set([index.replace(**{key:None for key in temp}) for index in leaves])
                self.cache[('indices',layer)]=sorted(indices,key=lambda index: index.to_tuple(priority=self.priority))
                self.cache[('table',layer)]=Table(self.indices(layer=layer))
                self.cache[('reversed_table',layer)]=self.table(layer=layer).reversed_table
                for index in self.indices(layer=layer):
                    self.add_leaf(parent=index.replace(**{key:None for key in layer}),leaf=index,data=None)
            if self.mode=='QN':
                for i,layer in enumerate(reversed(self.layers)):
                    key=('qnc_evolutions',layer)
                    self.cache[key]={}
                    if i==0:
                        for index in self.indices(layer):
                            self[index]=self.map(index)
                            self.cache[key][index]=[self[index]]
                    else:
                        for index in self.indices(layer):
                            buff=[]
                            for j,child in enumerate(self.children(index)):
                                if j==0:
                                    buff.append(self[child])
                                else:
                                    buff.append(buff[j-1].kron(self[child],'+',history=True))
                            self.cache[key][index]=buff
                            self[index]=buff[-1]
            else:
                for i,layer in enumerate(reversed(self.layers)):
                   for index in self.indices(layer):
                        self[index]=self.map(index) if i==0 else np.product([self[child] for child in self.children(index)])

    def ndegfre(self,index):
        '''
        The number of degrees of freedom reprented by index.
        Parameters:
            index: Index
                The index of the degrees of freedom.
        Returns: integer
            The number of degrees of freedom.
        '''
        if self.mode=='NB':
            return self[index]
        else:
            return self[index].n

    def indices(self,layer):
        '''
        The indices in a layer.
        Parameters:
            layer: tuple of string
                The layer where the indices are restricted.
        Returns: list of Index
            The indices in the requested layer.
        '''
        return self.cache[('indices',layer)]

    def table(self,layer):
        '''
        Return a index-sequence table with the index restricted on a specific layer.
        Parameters:
            layer: tuple of string
                The layer where the indices are restricted.
        Returns: Table
            The index-sequence table.
        '''
        return self.cache[('table',layer)]

    def reversed_table(self,layer):
        '''
        Return the reversed sequence-index table with the index restricted on a specific layer.
        Parameters:
            layer: tuple of string
                The layer where the indices are restricted.
        Returns: Table
            The sequence-index table.
        '''
        return self.cache[('reversed_table',layer)]

    def qnc_evolutions(self,layer):
        '''
        Retrun a dict in the form {branch:qncs}
            branch: index
                A branch on a specific layer.
            qncs: list of QuantumNumberCollection
                The kron path of the quantum number collection corresponding to the branch.
        Parameters:
            layer: tuple of string
                The layer where the branches are restricted.
        Returns: as above.
        '''
        return self.cache.get(('qnc_evolutions',layer),None)

    def labels(self,layer,full_labels=True):
        '''
        Return a OrderedDict of labels on a specific layer.
        Parameters:
            layer: tuple of string
                The layer where the labels are restricted.
            full_labels: logical, optional
                When True the full labels including the bond labels and physical labels will be returned;
                When Flase, only the physical labels will be returned.
        Returns: 
            When full_labels is True: OrderedDict of 3-tuple (L,S,R)
            When full_labels is False: OrderedDict of Label S
            L,S,R are in the following form:
                L=Label(identifier=i,qnc=None)
                S=Label(identifier=index,qnc=self[index])
                R=Label(identifier=(i+1)%len(indices),qnc=None)
            with,
                identifier: Index
                    The physical degrees of freedom of the labels restricted on this layer.
                i: integer
                    The sequence of the labels restricted on this layer.
                N: integer or QuantumNumberCollection
                    When self.mode=='QN', it is a QuantumNumberCollection, the quantum number collection of the physical degrees of freedom;
                    When self.mode=='NB', it is a integer, the number of the physical degrees of freedom.
        '''
        if ('labels',layer,full_labels) not in self.cache:
            result=OrderedDict()
            indices=sorted(self.indices(layer),key=lambda index: index.to_tuple(priority=self.priority))
            for i,index in enumerate(indices):
                S=Label(identifier=index,qnc=self[index])
                if full_labels:
                    L=Label(identifier=i,qnc=None)
                    R=Label(identifier=(i+1)%len(indices),qnc=None)
                    result[index]=(L,S,R)
                else:
                    result[index]=S
            self.cache[('labels',layer,full_labels)]=result
        return self.cache[('labels',layer,full_labels)]

class IndexPack(object):
    '''
    This class packs several degrees of freedom as a whole for convenience.
    Attributes:
        value: float64 or complex128
            The overall coefficient of the IndexPack.
    '''

    def __init__(self,value):
        '''
        Constructor.
        Parameters:
            value: float64/complex128
                The overall coefficient of the IndexPack.
        '''
        self.value=value

    def __add__(self,other):
        '''
        Overloaded operator(+), which supports the addition of an IndexPack instance with an IndexPack/IndexPackList instance.
        '''
        result=IndexPackList()
        result.append(self)
        if issubclass(other.__class__,IndexPack):
            result.append(other)
        elif isinstance(other,IndexPackList):
            result.extend(other)
        else:
            raise ValueError("IndexPack '+' error: the 'other' parameter must be of class IndexPack or IndexPackList.")
        return result

    def __rmul__(self,other):
        '''
        Overloaded operator(*).
        '''
        return self.__mul__(other)

class IndexPackList(list):
    '''
    This class packs several IndexPack as a whole for convenience.
    '''

    def __init__(self,*arg):
        for buff in arg:
            if issubclass(buff.__class__,IndexPack):
                self.append(buff)
            else:
                raise ValueError("IndexPackList init error: the input parameters must be of IndexPack's subclasses.")

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return 'IndexPackList('+', '.join([str(obj) for obj in self])

    def __add__(self,other):
        '''
        Overloaded operator(+), which supports the addition of an IndexPackList instance with an IndexPack/IndexPackList instance.
        '''
        result=IndexPackList(*self)
        if isinstance(other,IndexPack):
            result.append(other)
        elif isinstance(other,IndexPackList):
            result.extend(other)
        else:
            raise ValueError("IndexPackList '+' error: the 'other' parameter must be of class IndexPack or IndexPackList.")
        return result

    def __radd__(self,other):
        '''
        Overloaded operator(+).
        '''
        return self.__add__(other)

    def __mul__(self,other):
        '''
        Overloaded operator(*).
        '''
        result=IndexPackList()
        for buff in self:
            temp=buff*other
            if isinstance(temp,IndexPackList):
                result.extend(temp)
            elif issubclass(temp.__class__,IndexPack):
                result.append(temp)
            else:
                raise ValueError("IndexPackList *' error: the element(%s) in self multiplied by other is not of IndexPack/IndexPackList."%(buff))
        return result

    def __rmul__(self,other):
        '''
        Overloaded operator(*).
        '''
        return self.__mul__(other)
