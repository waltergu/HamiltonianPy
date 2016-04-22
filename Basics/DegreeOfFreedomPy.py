'''
Degrees of freedom in a lattice, including:
1) functions: union, subset, reversed_table
2) classes: Table, Index, Internal, Configuration
'''

__all__=['union','subset','reversed_table','Table','Index','Internal','Configuration']

from collections import OrderedDict

class Table(dict):
    '''
    This class provides the methods to get an index from its sequence number or vice versa.
    '''
    def __init__(self,indices=[],dicts=[],f=None):
        '''
        Constructor.
        Parameters:
            indices: list of any hashable object
                The indices that need to be mapped to sequences.
            dict: dict, optional
                An already constructed index-sequence table.
            f: function, optional
                The function used to map an index to a sequence.
                If it is None, the order of the index in indices will be used as its sequence number.
        '''
        for i,v in enumerate(indices):
            if f is None:
                self[v]=i
            else:
                self[v]=f(v)
        for dict in dicts:
            self.update(dict)

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
        buff={}
        for i,k in enumerate(sorted([k for k in result.keys()],key=key)):
            buff[k]=i
        result=buff
    return result

def subset(table,mask):
    '''
    This function returns a certain subset of an index-sequence table according to the mask function.
    Parameters:
        table: Table
            The mother table.
        mask: callable
            A certain subset of table is extracted according to the return value of this function on the index in the table.
            When the return value is True, the index will be included and the sequence is naturally determined by its order in the mother table.
    Returns:
        The subset table.
    '''
    result=Table()
    for k,v in table.iteritems():
        if mask(k):
            result[k]=v
    buff={}
    for i,k in enumerate(sorted([key for key in result.keys()],key=result.get)):
        buff[k]=i
    result.update(buff)
    return result

def reversed_table(table):
    '''
    This function returns the sequence-index table for a reversed lookup.
    Parameters:
        table: Table
            The original table.
    Returns: Table
        The reversed table whose key is the sequence and value the index.
    '''
    result=Table()
    for k,v in table.iteritems():
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
        self.__dict__=pid._asdict()
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
        if len(priority)<len(self.__dict__):
            raise ValueError("Index to_tuple error: the priority doesn't cover all the attributes.")
        return tuple(map(self.__dict__.get,priority))

class Internal(object):
    '''
    This class is the base class for all internal degrees of freedom in a single point.
    '''
    
    def table(self,pid,**karg):
        '''
        Return a Table instance that contains all the allowed indices constructed from an input pid and the internal degrees of freedom.
        Parameters:
            pid: PID
                The spatial part of the indices.
        Returns: Table
            The index-sequence table.
        Note: this method must be overridden by its child class if it is to be used.
        '''
        raise ValueError("%s table error: it is not implemented."%self.__class__.__name__)

class Configuration(dict):
    '''
    Configuration of the degrees of freedom in a lattice.
    Attributes:
        priority: list of string
            The sequence priority of the allowed indices that can be defined on a lattice.
    '''

    def __init__(self,dict=None,priority=None):
        '''
        Constructor.
        Parameters:
            dict: dict with key PID and value Internal respectively
                The key is the pid of the lattice point and the value is the internal degrees of freedom of that point.
            priority: list of string
                The sequence priority of the allowed indices that can be defined on the lattice.
        '''
        if dict is not None:
            self.update(dict)
        self.priority=priority

    def table(self,**karg):
        '''
        Return a Table instance that contains all the allowed indices which can be defined on a lattice.
        '''
        return union([value.table(key,**karg) for key,value in self.iteritems()],key=lambda index: index.to_tuple(priority=self.priority))

    def enlarged(self,map):
        '''
        Return an enlarged configuration (several copies of the values with new keys) according to map.
        Parameters:
            map: dict
                Its items (key,value) has the following meaning:
                    key: new key in the enlarged configuration;
                    value: old key in the original configuration.
        Returns: Configuration
            The enlarged configuration.
        '''
        result=copy(self)
        for key,value in map.iteritems():
            result[key]=result[value]
        return result            
