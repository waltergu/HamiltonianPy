'''
Table.
'''
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

def reverse_table(table):
    '''
    This function returns the sequence-index table for a reversed lookup.
    '''
    result=Table()
    for k,v in table.iteritems():
        result[v]=k
    return result
