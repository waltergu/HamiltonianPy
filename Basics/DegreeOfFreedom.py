'''
============================================================
Descriptions of the interanl degrees of freedom on a lattice
============================================================

This modulate defines several classes to define the way to describe the interanl degrees of freedom on a lattice, including
    * classes: Status,Table, Index, Internal, IDFConfig, Label, DegFreTree, IndexPack, IndexPacks
'''

__all__=['Status','Table','Index','Internal','IDFConfig','Label','DegFreTree','IndexPack','IndexPacks']

import numpy as np
from numpy.linalg import norm
from Constant import RZERO
from Geometry import PID
from QuantumNumber import QuantumNumbers
from ..Misc import Arithmetic,Tree
from copy import copy,deepcopy
from collections import OrderedDict

class Status(object):
    '''
    This class provides an object with a stauts.

    Attributes
    ----------
    name : any hashable object
        The name of the object.
    info : any object
        Additional information of the object.
    data : OrderedDict
        The data of the object.
        In current version, these are the parameters of the object.
    _const_ : OrderedDict
        The constant parameters of the object.
    _alter_ : OrderedDict
        The alterable parameters of the object.
    '''

    def __init__(self,name='',info='',const=None,alter=None):
        '''
        Constructor.

        Parameters
        ----------
        name : any hashable object
            The name of the object.
        info : any object
            Additional information of the object.
        const,alter : OrderedDict, optional
            The constant/alterable parameters of the object.
        '''
        self.name=name
        self.info=info
        self._const_=OrderedDict() if const is None else const
        self._alter_=OrderedDict() if alter is None else alter
        self.data=OrderedDict()
        self.data.update(self._const_)
        self.data.update(self._alter_)

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        result=[]
        if len(str(self.name))>0:result.append(str(self.name))
        if len(self._const_)>0:result.append('_'.join([str(v) for v in self._const_.values()]))
        if len(self._alter_)>0:result.append('_'.join([str(v) for v in self._alter_.values()]))
        if len(str(self.info))>0:result.append(str(self.info))
        return '_'.join(result)

    def update(self,const=None,alter=None):
        '''
        Update the parameters of the object.

        Parameters
        ----------
        const,alter : dict, optional
            The new parameters.
        '''
        if const is not None:
            self._const_.update(const)
            self.data.update(const)
        if alter is not None:
            self._alter_.update(alter)
            self.data.update(alter)

    def __getitem__(self,key):
        '''
        Overloaded operator([]).
        '''
        return self.data[key]

    @property
    def const(self):
        '''
        This method returns a string representation of the status containing only the constant parameters.
        '''
        result=[]
        if len(str(self.name))>0:result.append(str(self.name))
        if len(self._const_)>0:result.append('_'.join([str(v) for v in self._const_.values()]))
        if len(str(self.info))>0:result.append(str(self.info))
        return '_'.join(result)

    @property
    def alter(self):
        '''
        This method returns a string representation of the status containing only the alterable parameters.
        '''
        result=[]
        if len(str(self.name))>0:result.append(str(self.name))
        if len(self._alter_)>0:result.append('_'.join([str(v) for v in self._alter_.values()]))
        if len(str(self.info))>0:result.append(str(self.info))
        return '_'.join(result)

    def __le__(self,other):
        '''
        Overloaded operator(<=).
        If ``self.data`` is a subset of ``other.data``, return True. Otherwise False.
        '''
        try:
            for key,value in self.data.iteritems():
                if norm(np.array(value)-np.array(other.data[key]))>RZERO:
                    return False
            else:
                return True
        except KeyError:
            return False

    def __ge__(self,other):
        '''
        Overloaded operator(>=).
        If ``other.data`` is a subset of ``self.data``, return True. Otherwise False.
        '''
        return other.__le__(self)

class Table(dict):
    '''
    This class provides the methods to get an index from its sequence number or vice versa.
    '''

    def __init__(self,indices=[],key=None):
        '''
        Constructor.

        Parameters
        ----------
        indices : list of any hashable object
            The indices that need to be mapped to sequences.
        key : function, optional
            The function used to sort the indices.

        Notes
        -----
        The final order of the index in `indices` will be used as its sequence number.
        '''
        for i,v in enumerate(indices if key is None else sorted(indices,key=key)):
            self[v]=i

    @staticmethod
    def union(tables,key=None):
        '''
        This function returns the union of index-sequence tables.

        Parameters
        ----------
        tables : list of Table
            The tables to be unioned.
        key : callable, optional
            The function used to compare different indices in tables.
            When it is None, the sequence of an index will be naturally ordered by the its sequence in the input tables.

        Returns
        -------
        Table
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

        Parameters
        ----------
        select : callable
            The select function whose argument is the index of the mother table.
            When its returned value is True, the index will be included in the subset.
            The sequence is naturally determined by its order in the mother table.

        Returns
        -------
        Table
            The subset table.
        '''
        return Table(sorted([key for key in self if select(key)],key=self.get))

    @property
    def reversed_table(self):
        '''
        This function returns the sequence-index table for a reversed lookup.

        Returns
        -------
        Table
            The reversed table whose key is the sequence and value the index.
        '''
        result=Table()
        for k,v in self.iteritems():
            result[v]=k
        return result

class Index(tuple):
    '''
    This class provides an index for a microscopic degree of freedom, including the spatial part and interanl part.

    Attributes
    ----------
    names : tuple of string
        The names of the microscopic degrees of freedom.
    icls : Class
        The class of the interanl part of the index.
    '''

    def __new__(cls,pid,iid):
        '''
        Constructor.

        Parameters
        ----------
        pid : PID
            The point index, i.e. the spatial part in a lattice of the index
        iid : namedtuple
            The internal index, i.e. the internal part in a point of the index.
        '''

        self=super(Index,cls).__new__(cls,pid+iid)
        self.names=pid._fields+iid._fields
        self.icls=iid.__class__
        return self

    def __getnewargs__(self):
        '''
        Return the arguments for Index.__new__, required by copy and pickle.
        '''
        return (self.pid,self.iid)

    def __getstate__(self):
        '''
        Since Index.__new__ constructs everything, self.__dict__ can be omitted for copy and pickle.
        '''
        pass

    def __getattr__(self,key):
        '''
        Overloaded dot(.) operator.
        '''
        try:
            return self[self.names.index(key)]
        except ValueError:
            raise AttributeError("'Index' has no attribute %s."%(key))

    @property
    def pid(self):
        '''
        The pid part of the index.
        '''
        return PID(**{key:getattr(self,key) for key in PID._fields})

    @property
    def iid(self):
        '''
        The iid part of the index.
        '''
        return self.icls(**{key:getattr(self,key) for key in self.names if key not in PID._fields})

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return ''.join(['Index(','=%r, '.join(self.names),'=%r)'])%self

    def replace(self,**karg):
        '''
        Return a new Index object with specified fields replaced with new values.
        '''
        result=tuple.__new__(Index,map(karg.pop,self.names,self))
        if karg:
            raise ValueError('Index replace error: it got unexpected field names: %r'%karg.keys())
        result.names=self.names
        result.icls=self.icls
        return result

    def mask(self,*arg):
        '''
        Mask some attributes of the index to None and return the new one.

        Parameters
        ----------
        arg : list of string
            The attributes to be masked.

        Returns
        -------
        Index
            The masked index.
        '''
        return self.replace(**{key:None for key in arg})

    def select(self,*arg):
        '''
        Select some attributes of the index, mask the others to None and return the new one.

        Parameters
        ----------
        arg : list of string
            The attributes to be selected.

        Returns
        -------
        Index
            The selected index.
        '''
        return self.mask(*[key for key in self.names if key not in arg])

    def to_tuple(self,priority):
        '''
        Convert an instance to tuple according to the parameter priority.

        Parameters
        ----------
        priority : list of string
            Every element of this list should correspond to a name of an attribute of self.
            The elements should have no duplicates and its length should be equal to the number of self's attributes.

        Returns
        -------
        tuple
            The elements of the returned tuple are equal to the attributes of self. 
            Their orders are determined by the orders they appear in priority.
        '''
        if len(set(priority))<len(priority):
            raise ValueError('Index to_tuple error: the priority has duplicates.')
        if len(priority)!=len(self.names):
            raise ValueError("Index to_tuple error: the priority doesn't cover all the attributes.")
        return tuple([getattr(self,name) for name in priority])

class Internal(object):
    '''
    This class is the base class for all internal degrees of freedom in a single point.
    '''

    def indices(self,pid,mask=[]):
        '''
        Return a list of all the masked indices within this internal degrees of freedom combined with an extra spatial part.

        Parameters
        ----------
        pid : PID
            The extra spatial part of the indices.
        mask : list of string, optional
            The attributes that will be masked to None.

        Returns
        -------
        list of Index
            The indices.
        '''
        raise NotImplementedError("%s indices error: it is not implemented."%self.__class__.__name__)

class IDFConfig(dict):
    '''
    Configuration of the internal degrees of freedom in a lattice. For each of its (key,value) pairs,
        * key: PID
            The pid of the lattice point where the interanl degrees of freedom live.
        * value: subclasses of Internal
            The internal degrees of freedom on the corresponding point.

    Attributes
    ----------
    priority : list of string 
        The sequence priority of the allowed indices that can be defined on a lattice.
    map : function
        This function maps the pid of a lattice point to the interanl degrees of freedom on it.
    '''

    def __init__(self,priority,pids=[],map=None):
        '''
        Constructor.

        Parameters
        ----------
        priority : list of string
            The sequence priority of the allowed indices that can be defined on the lattice.
        pids : list of PID, optional
            The pids of the lattice points where the interanl degrees of freedom live.
        map : function, optional
            This function maps the pid of a lattice point to the interanl degrees of freedom on it.
        '''
        self.reset(priority=priority,pids=pids,map=map)

    def reset(self,priority=None,pids=[],map=None):
        '''
        Reset the idfconfig.

        Parameters
        ----------
        pids : list of PID, optional
            The pids of the lattice points where the interanl degrees of freedom live.
        map : function, optional
            This function maps the pid of a lattice point to the interanl degrees of freedom on it.
        priority : list of string, optional
            The sequence priority of the allowed indices that can be defined on the lattice.
        '''
        self.clear()
        self.priority=getattr(self,'priority',None) if priority is None else priority
        self.map=getattr(self,'map',None) if map is None else map
        for pid in pids:
            self[pid]=self.map(pid)

    def __setitem__(self,key,value):
        '''
        Set the value of an item.

        Parameters
        ----------
        key : FID
            The pid of the lattice point where the internal degrees of freedom live.
        value : subclasses of Internal
            The internal degrees of freedom on the corresponding point.
        '''
        assert isinstance(key,PID)
        assert isinstance(value,Internal)
        dict.__setitem__(self,key,value)

    def subset(self,select):
        '''
        This function returns a certain subset of an IDFConfig according to the select function.

        Parameters
        ----------
        select : callable
            The select function whose argument is the pid of the mother IDFConfig.
            When its returned value is True, the pid will be included in the subset.

        Returns
        -------
        IDFConfig
            The subset IDFConfig.
        '''
        result=IDFConfig(priority=self.priority,map=self.map)
        for pid,interanl in self.iteritems():
            if select(pid): result[pid]=interanl
        return result

    def table(self,mask=[]):
        '''
        Return a Table instance that contains all the allowed indices which can be defined on a lattice.
        '''
        return Table([index for key,value in self.items() for index in value.indices(key,mask)],key=lambda index: index.to_tuple(priority=self.priority))

class Label(tuple):
    '''
    The label for a set of degrees of freedom.

    Attributes
    ----------
    names : ('identifier','_prime_')
        The names of the immutable part of the label.
    qns : integer or QuantumNumbers
        * When integer, it is the dimension of the label;
        * When QuantumNumbers, it is the collection of the quantum numbers of the label.
    '''
    names=('identifier','_prime_')

    def __new__(cls,identifier,prime=False,qns=None):
        '''
        Constructor.

        Parameters
        ----------
        identifier : Label
            The index of the label
        prime : logical, optional
            When True, the label is in the prime form; otherwise not.
        qns : integer or QuantumNumbers, optional
            * When integer, it is the dimension of the label;
            * When QuantumNumbers, it is the collection of the quantum numbers of the label.
        '''
        self=tuple.__new__(cls,(identifier,prime))
        self.qns=qns
        return self

    def __getnewargs__(self):
        '''
        Return the arguments for Label.__new__, required by copy and pickle.
        '''
        return tuple(self)+(self.qns,)

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
            raise AttributeError("'Label' object has no attribute %s."%(key))

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        if self[-1]:
            return "Label(%s)%s<%s>"%(self[0],"'",self.qns)
        else:
            return "Label(%s)<%s>"%(self[0],self.qns)

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        if self[-1]:
            return "Label(%s)%s<%s>"%(self[0],"'",repr(self.qns))
        else:
            return "Label(%s)<%s>"%(self[0],repr(self.qns))

    def replace(self,**karg):
        '''
        Return a new label with some of its attributes replaced.

        Parameters
        ----------
        karg : dict in the form (key,value), with
            * key: string
                The attributes of the label
            * value: any object
                The corresponding value.

        Returns
        -------
        Label
            The new label.
        '''
        result=tuple.__new__(self.__class__,map(karg.pop,type(self).names,self))
        for key,value in self.__dict__.iteritems():
            setattr(result,key,karg.pop(key,value))
        if karg:
            raise ValueError("Label replace error: %s are not the attributes of the label."%karg.keys())
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

    @property
    def dim(self):
        '''
        The length of the dimension this label labels.
        '''
        if isinstance(self.qns,QuantumNumbers):
            return len(self.qns)
        else:
            return self.qns

    @property
    def qnon(self):
        '''
        True for qns is an instance of QuantumNumbers otherwise False.
        '''
        return isinstance(self.qns,QuantumNumbers)

class DegFreTree(Tree):
    '''
    The tree of the layered degrees of freedom.
    For each (node,data) pair of the tree,
        * node: Index
            The selected index which can represent a couple of indices.
        * data: integer or QuantumNumbers
            When an integer, it is the number of degrees of freedom that the index represents;
            When a QuantumNumbers, it is the quantum number collection that the index is associated with.

    Attributes
    ----------
    mode : 'QN' or 'NB'
        The mode of the DegFreTree.
    layers : list of tuples of string
        The tag of each layer of indices.
    priority : lsit of string
        The sequence priority of the allowed indices.
    map : function
        This function maps a leaf (bottom index) of the DegFreTree to its corresponding data.
    cache : dict
        The cache of the degfretree.
    '''

    def __init__(self,mode,layers,priority,leaves=[],map=None):
        '''
        Constructor.

        Parameters
        ----------
        mode : 'QN' or 'NB'
            The mode of the DegFreTree.
        layers : list of tuples of string
            The tag of each layer of indices.
        priority : lsit of string
            The sequence priority of the allowed indices.
        leaves : list of Index, optional
            The leaves (bottom indices) of the DegFreTree.
        map : function, optional
            This function maps a leaf (bottom index) of the DegFreTree to its corresponding data.
        '''
        self.reset(mode=mode,layers=layers,priority=priority,leaves=leaves,map=map)

    def reset(self,mode=None,layers=None,priority=None,leaves=[],map=None):
        '''
        Reset the DegFreTree.

        Parameters
        ----------
        mode,layers,priority,leaves,map :
            Please see DegFreTree.__init__ for details.
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
                for index in self.indices(layer=layer):
                    self.add_leaf(parent=index.replace(**{key:None for key in layer}),leaf=index,data=None)
            for i,layer in enumerate(reversed(self.layers)):
                if i==0:
                    for index in self.indices(layer):
                        self[index]=self.map(index)
                else:
                    for index in self.indices(layer):
                        if self.mode=='QN':
                            self[index]=QuantumNumbers.kron([self[child] for child in self.children(index)])
                        else:
                            self[index]=np.product([self[child] for child in self.children(index)])

    def ndegfre(self,index):
        '''
        The number of degrees of freedom represented by index.

        Parameters
        ----------
        index : Index
            The index of the degrees of freedom.

        Returns
        -------
        integer
            The number of degrees of freedom.
        '''
        if self.mode=='NB':
            return self[index]
        else:
            return len(self[index])

    def indices(self,layer=0):
        '''
        The indices in a layer.

        Parameters
        ----------
        layer : integer/tuple-of-string, optional
            The layer where the indices are restricted.

        Returns
        -------
        list of Index
            The indices in the requested layer.
        '''
        return self.cache[('indices',self.layers[layer] if type(layer) in (int,long) else layer)]

    def table(self,layer=0):
        '''
        Return a index-sequence table with the index restricted on a specific layer.

        Parameters
        ----------
        layer : integer/tuple-of-string
            The layer where the indices are restricted.

        Returns
        -------
        Table
            The index-sequence table.
        '''
        return self.cache[('table',self.layers[layer] if type(layer) in (int,long) else layer)]

    def labels(self,mode,layer=0):
        '''
        Return the inquired labels.

        Parameters
        ----------
        mode : 'B','S','O'
            * 'B' for bond labels of an mps;
            * 'S' for site labels of an mps or an mpo;
            * 'O' for bond labels of an mpo.
        layer : integer/tuple-of-string, optional
            The layer information of the inquired labels.

        Returns
        -------
        list of Label
            The inquired labels.
        '''
        mode,layer=mode.upper(),self.layers[layer] if type(layer) in (int,long) else layer
        assert mode in ('B','S','O')
        if ('labels',mode,layer) not in self.cache:
            if mode in ('B','O'):
                result=[Label(identifier='%s%s-%s'%(mode,self.layers.index(layer),i),qns=None) for i in xrange(len(self.indices(layer))+1)]
            else:
                result=[Label(identifier=index,qns=self[index]) for index in self.indices(layer)]
            self.cache[('labels',mode,layer)]=result
        return self.cache[('labels',mode,layer)]

class IndexPack(Arithmetic):
    '''
    This class packs several degrees of freedom as a whole for convenience.

    Attributes
    ----------
    value : float64 or complex128
        The overall coefficient of the IndexPack.
    '''

    def __init__(self,value):
        '''
        Constructor.

        Parameters
        ----------
        value: float64/complex128
            The overall coefficient of the IndexPack.
        '''
        self.value=value

    def __add__(self,other):
        '''
        Overloaded left addition(+) operator, which supports the left addition by an IndexPack/IndexPacks.
        '''
        result=IndexPacks()
        result.append(self)
        if issubclass(other.__class__,IndexPack):
            result.append(other)
        elif isinstance(other,IndexPacks):
            result.extend(other)
        else:
            assert norm(other)==0
        return result

class IndexPacks(Arithmetic,list):
    '''
    This class packs several IndexPack as a whole for convenience.
    '''

    def __init__(self,*arg):
        for obj in arg:
            if issubclass(obj.__class__,IndexPack):
                self.append(obj)
            else:
                raise ValueError("IndexPacks init error: the input parameters must be of IndexPack's subclasses.")

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return 'IndexPacks('+', '.join([str(obj) for obj in self])

    def __iadd__(self,other):
        '''
        Overloaded self-addition(+=) operator, which supports the self-addition by an IndexPack/IndexPacks.
        '''
        if isinstance(other,IndexPack):
            self.append(other)
        elif isinstance(other,IndexPacks):
            self.extend(other)
        else:
            assert norm(other)==0
        return self

    def __add__(self,other):
        '''
        Overloaded left addition(+) operator, which supports the left addition by an IndexPack/IndexPacks.
        '''
        result=IndexPacks(*self)
        if isinstance(other,IndexPack):
            result.append(other)
        elif isinstance(other,IndexPacks):
            result.extend(other)
        else:
            assert norm(other)==0
        return result

    def __mul__(self,other):
        '''
        Overloaded left multiplication(*) operator, which supports the left-multiplication by a scalar/IndexPack/IndexPacks.
        '''
        result=IndexPacks()
        for obj in self:
            temp=obj*other
            if isinstance(temp,IndexPacks):
                result.extend(temp)
            elif issubclass(temp.__class__,IndexPack):
                result.append(temp)
            else:
                raise ValueError("IndexPacks *' error: the element(%s) in self multiplied by other is not of IndexPack/IndexPacks."%(obj))
        return result
    
    __imul__=__mul__
