'''
Degrees of freedom in a lattice, including:
1) classes: Status,Table, Index, Internal, IDFConfig, DegFreTree, IndexPack, IndexPackList
'''

__all__=['Status','Table','Index','Internal','IDFConfig','IndexPack','IndexPackList']

import numpy as np
from numpy.linalg import norm
from Constant import RZERO
from Geometry import PID
from copy import copy,deepcopy
from collections import OrderedDict

class Status(object):
    '''
    This class provides an object with a stauts.
    Attributes:
        name: any hashable object
            The name of the object.
        info: any object
            Additional information of the object.
        data: OrderedDict
            The data of the object.
            In current version, these are the parameters of the object.
        _const_: OrderedDict
            The constant parameters of the object.
        _alter_: OrderedDict
            The alterable parameters of the object.
    '''

    def __init__(self,name='',info='',const=None,alter=None):
        '''
        Constructor.
        Parameters:
            name: any hashable object
                The name of the object.
            info: any object
                Additional information of the object.
            const,alter: OrderedDict, optional
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
        Parameters:
            const, alter: dict, optional
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
        If self.data is a subset of other.data, return True. Otherwise False.
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
        If other.data is a subset of self.data, return True. Otherwise False.
        '''
        return other.__le__(self)

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
                The select function whose argument is the index of the mother table.
                When its returned value is True, the index will be included in the subset.
                The sequence is naturally determined by its order in the mother table.
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
    Attributes:
        names: tuple of string
            The names of the microscopic degrees of freedom.
        icls: Class
            The class of the interanl part of the index.
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

    def __repr__(self):
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
        return self.mask(*[key for key in self.names if key not in arg])

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
        self.priority=getattr(self,'priority',None) if priority is None else priority
        self.map=getattr(self,'map',None) if map is None else map
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

    def subset(self,select):
        '''
        This function returns a certain subset of an IDFConfig according to the select function.
        Parameters:
            select: callable
                The select function whose argument is the pid of the mother IDFConfig.
                When its returned value is True, the pid will be included in the subset.
         Returns:
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
