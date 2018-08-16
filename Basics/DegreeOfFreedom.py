'''
============================================================
Descriptions of the internal degrees of freedom on a lattice
============================================================

This modulate defines several classes to define the way to describe the internal degrees of freedom on a lattice, including
    * classes: Table, Index, Internal, IDFConfig, QNSConfig, IndexPack, IndexPacks
'''

__all__=['Table','Index','Internal','IDFConfig','QNSConfig','IndexPack','IndexPacks']

import numpy as np
import itertools as it
from numpy.linalg import norm
from .Utilities import RZERO,Arithmetic,decimaltostr
from .Geometry import PID
from .QuantumNumber import QuantumNumbers
from collections import OrderedDict

class Table(dict):
    '''
    This class provides the methods to get an index from its sequence number or vice versa.
    '''

    def __init__(self,indices=(),key=None):
        '''
        Constructor.

        Parameters
        ----------
        indices : list of any hashable object
            The indices that need to be mapped to sequences.
        key : callabel, optional
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
        return Table(it.chain(*(sorted(table,key=table.get) for table in tables)),key=key)

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
    def reversal(self):
        '''
        This function returns the sequence-index table for a reversed lookup.

        Returns
        -------
        Table
            The reversed table whose key is the sequence and value the index.
        '''
        result=Table()
        for k,v in self.items():
            result[v]=k
        return result

class Index(tuple):
    '''
    This class provides an index for a microscopic degree of freedom, including the spatial part and internal part.

    Attributes
    ----------
    names : tuple of string
        The names of the microscopic degrees of freedom.
    icls : Class
        The class of the internal part of the index.
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
        return self.pid,self.iid

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
            raise AttributeError("'Index' has no attribute %s."%key)

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        return (':'.join(['%s']*len(self)))%self

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return ''.join(['Index(','=%r, '.join(self.names),'=%r)'])%self

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

    @property
    def masks(self):
        '''
        The masks of the index's attributes.
        '''
        return tuple(name for name in self.names if getattr(self,name) is None)

    def replace(self,**karg):
        '''
        Return a new Index object with specified fields replaced with new values.
        '''
        result=super(Index,type(self)).__new__(type(self),map(karg.pop,self.names,self))
        if karg: raise ValueError('Index replace error: it got unexpected field names: %r'%list(karg.keys()))
        result.names=self.names
        result.icls=self.icls
        return result

    def mask(self,*arg):
        '''
        Mask some attributes of the index to None and return the new one.

        Parameters
        ----------
        arg : list of str
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
        arg : list of str
            The attributes to be selected.

        Returns
        -------
        Index
            The selected index.
        '''
        return self.mask(*[key for key in self.names if key not in arg])

    def totuple(self,priority):
        '''
        Convert an instance to tuple according to the parameter priority.

        Parameters
        ----------
        priority : list of str
            Every element of this list should correspond to a name of an attribute of self.
            The elements should have no duplicates and its length should be equal to the number of self's attributes.

        Returns
        -------
        tuple
            The elements of the returned tuple are equal to the attributes of self. 
            Their orders are determined by the orders they appear in priority.
        '''
        if len(set(priority))<len(priority):
            raise ValueError('Index totuple error: the priority has duplicates.')
        if len(priority)!=len(self.names):
            raise ValueError("Index totuple error: the priority doesn't cover all the attributes.")
        return tuple([getattr(self,name) for name in priority])

class Internal(object):
    '''
    This class is the base class for all internal degrees of freedom in a single point.
    '''

    def indices(self,pid,mask=()):
        '''
        Return a list of all the masked indices within this internal degrees of freedom combined with an extra spatial part.

        Parameters
        ----------
        pid : PID
            The extra spatial part of the indices.
        mask : list of str, optional
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
            The pid of the lattice point where the internal degrees of freedom live.
        * value: subclasses of Internal
            The internal degrees of freedom on the corresponding point.

    Attributes
    ----------
    priority : list of str
        The sequence priority of the allowed indices that can be defined on a lattice.
    map : callable
        This function maps the pid of a lattice point to the internal degrees of freedom on it.
    '''

    def __init__(self,priority,pids=(),map=None):
        '''
        Constructor.

        Parameters
        ----------
        priority : list of str
            The sequence priority of the allowed indices that can be defined on the lattice.
        pids : list of PID, optional
            The pids of the lattice points where the internal degrees of freedom live.
        map : callable, optional
            This function maps the pid of a lattice point to the internal degrees of freedom on it.
        '''
        self.reset(priority=priority,pids=pids,map=map)

    def reset(self,priority=None,pids=(),map=None):
        '''
        Reset the idfconfig.

        Parameters
        ----------
        pids : list of PID, optional
            The pids of the lattice points where the internal degrees of freedom live.
        map : callable, optional
            This function maps the pid of a lattice point to the internal degrees of freedom on it.
        priority : list of str, optional
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
        for pid,interanl in self.items():
            if select(pid): result[pid]=interanl
        return result

    def table(self,mask=()):
        '''
        Return a Table instance that contains all the allowed indices which can be defined on a lattice.
        '''
        return Table([index for key,value in self.items() for index in value.indices(key,mask)],key=lambda index: index.totuple(priority=self.priority))

class QNSConfig(dict):
    '''
    Configuration of the quantum numbers. For each of its (key,value) pairs,
        * key: Index
            The index representing a number of degrees of freedom.
        * value: QuantumNumbers
            The quantum numbers associated with the index.

    Attributes
    ----------
    priority : list of str
        The sequence priority of the allowed indices.
    map : callable
        This function maps a index of the QNSConfig to its corresponding quantum numbers.
    '''

    def __init__(self,priority,indices=None,map=None):
        '''
        Constructor.

        Parameters
        ----------
        priority : list of str
            The sequence priority of the allowed indices.
        indices : list of Index, optional
            The indices of the QNSConfig.
        map : callable, optional
            This function maps a index of the QNSConfig to its corresponding quantum numbers.
        '''
        self.reset(priority=priority,indices=indices,map=map)

    def reset(self,priority,indices=(),map=None):
        '''
        Reset the QNSConfig.

        Parameters
        ----------
        priority : list of str
            The sequence priority of the allowed indices.
        indices : list of Index, optional
            The indices of the QNSConfig.
        map : callable, optional
            This function maps a index of the QNSConfig to its corresponding quantum numbers.
        '''
        self.clear()
        self.priority=priority or getattr(self,'priority',None)
        self.map=map or getattr(self,'map',None)
        for index in indices:
            self[index]=self.map(index)

    def __setitem__(self,key,value):
        '''
        Set the value of an item.

        Parameters
        ----------
        key : Index
            The index of the QNSConfig.
        value : QuantumNumbers
            The corresponding quantum numbers.
        '''
        assert isinstance(key,Index)
        assert isinstance(value,QuantumNumbers)
        dict.__setitem__(self,key,value)

    def subset(self,select):
        '''
        This function returns a certain subset of a QNSConfig according to the select function.

        Parameters
        ----------
        select : callable
            The select function whose argument is the index of QNSConfig. True for including.

        Returns
        -------
        QNSConfig
            The subset QNSConfig.
        '''
        result=QNSConfig(priority=self.priority,map=self.map)
        for index,qns in self.items():
            if select(index): result[index]=qns
        return result

class IndexPack(Arithmetic):
    '''
    This class packs several internal degrees of freedom as a whole for convenience.

    Attributes
    ----------
    value : float or complex
        The overall coefficient of the IndexPack.
    '''

    def __init__(self,value):
        '''
        Constructor.

        Parameters
        ----------
        value: float/complex
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
        '''
        Constructor.
        '''
        for obj in arg:
            if issubclass(obj.__class__,IndexPack):
                self.append(obj)
            else:
                raise ValueError("IndexPacks init error: the input parameters must be of IndexPack's subclasses.")

    def __str__(self):
        '''
        Convert an instance to string.
        '''
        return ''.join(['IndexPacks(',','.join([repr(obj) for obj in self]),')'])

    def __repr__(self):
        '''
        Convert an instance to string.
        '''
        result=[]
        for i,obj in enumerate(self):
            rep=repr(obj)
            if i>0: result.append('' if rep[0]=='-' else '+')
            result.append(rep)
        return ''.join(result)

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
                raise ValueError("IndexPacks *' error: the element(%s) in self multiplied by other is not of IndexPack/IndexPacks."%obj)
        return result

    __imul__=__mul__

    __eq__=list.__eq__