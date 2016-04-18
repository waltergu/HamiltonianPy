'''
Table, including
1) functions: union, subset, reversed_table
2) classes: Table
'''

__all__=['union','subset','reversed_table','Table']

class Table(dict):
    '''
    This class provides the methods to get an index from its sequence number or vice versa.
    '''
    def __init__(self,indices=[],dicts=[],f=None):
        '''
        Constructor.
        Parameters:
            indices: list of Index
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
        result.update(buff)
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
